#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, Image, NavSatFix
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import math
import re
from cv_bridge import CvBridge


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # Parameters
        self.declare_parameter('detection_topic', '/yolo/detections')
        self.declare_parameter('pointcloud_topic', '/sensing/lidar/top/outlier_filtered/pointcloud')
        self.declare_parameter('inference_image_topic', '/yolo/inference_image')
        self.declare_parameter('gnss_topic', '/sensing/gnss/fix')
        self.declare_parameter('lidar_offset_x', -0.5)
        self.declare_parameter('lidar_offset_y', 0.0)
        self.declare_parameter('lidar_offset_z', 2.0)
        self.declare_parameter('camera_hfov_deg', 90.0)
        self.declare_parameter('match_angle_padding_deg', 5.0)

        detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        pointcloud_topic = self.get_parameter('pointcloud_topic').get_parameter_value().string_value
        inference_image_topic = self.get_parameter('inference_image_topic').get_parameter_value().string_value
        gnss_topic = self.get_parameter('gnss_topic').get_parameter_value().string_value
        self.lidar_offset = np.array([
            self.get_parameter('lidar_offset_x').get_parameter_value().double_value,
            self.get_parameter('lidar_offset_y').get_parameter_value().double_value,
            self.get_parameter('lidar_offset_z').get_parameter_value().double_value,
        ])
        self.hfov_rad = math.radians(self.get_parameter('camera_hfov_deg').get_parameter_value().double_value)
        self.match_angle_padding = math.radians(self.get_parameter('match_angle_padding_deg').get_parameter_value().double_value)

        # Subscribers
        self.detection_sub = self.create_subscription(String, detection_topic, self.detection_callback, 10)
        self.pc_sub = self.create_subscription(PointCloud2, pointcloud_topic, self.pointcloud_callback, 10)
        self.image_sub = self.create_subscription(Image, inference_image_topic, self.image_callback, 10)
        self.gnss_sub = self.create_subscription(NavSatFix, gnss_topic, self.gnss_callback, 10)

        # Publisher
        self.pub = self.create_publisher(String, '/fusion/detections', 10)

        # State
        self.latest_detections = []  # list of dicts with class, x1 y1 x2 y2, conf
        self.latest_pc = None
        self.latest_image_width = None
        self.latest_gnss = None
        self.bridge = CvBridge()

        # Timer to run fusion at 10Hz
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(f'Fusion node started. Listening to {detection_topic} and {pointcloud_topic}')

    def detection_callback(self, msg: String):
        text = msg.data.strip()
        parsed = self.parse_detection_text(text)
        if parsed:
            self.latest_detections = parsed
        else:
            self.latest_detections = []

    def image_callback(self, msg: Image):
        # Only need width to map bbox center to angle
        self.latest_image_width = msg.width

    def gnss_callback(self, msg: NavSatFix):
        self.latest_gnss = msg

    def pointcloud_callback(self, msg: PointCloud2):
        # store latest pointcloud message
        self.latest_pc = msg

    def timer_callback(self):
        if not self.latest_detections or self.latest_pc is None or self.latest_image_width is None:
            return
        # Convert pointcloud to numpy array of points
        points = []
        for p in pc2.read_points(self.latest_pc, field_names=('x','y','z'), skip_nans=True):
            points.append(p)
        if not points:
            return
        pts = np.array(points)  # Nx3
        # apply lidar offset (pointcloud reported in lidar frame -> transform to ego frame)
        pts = pts + self.lidar_offset.reshape((1,3))

        results = []
        for det in self.latest_detections:
            cls = det['class']
            x1,y1,x2,y2 = det['bbox']
            conf = det['conf']
            cx = (x1 + x2)/2.0
            w = (x2 - x1)
            img_w = self.latest_image_width
            # map pixel center to horizontal angle
            rel = (cx - img_w/2.0)/img_w  # -0.5..0.5
            angle_center = rel * self.hfov_rad
            # width based angular tolerance
            angle_tol = max(abs(w)/img_w * self.hfov_rad * 0.5, self.match_angle_padding)

            # compute azimuth for each point (atan2(y,x))
            az = np.arctan2(pts[:,1], pts[:,0])
            # select points roughly in front (x>0)
            mask_front = pts[:,0] > 0.2
            mask_angle = np.abs(self.angle_diff(az, angle_center)) < angle_tol
            mask = mask_front & mask_angle
            if not np.any(mask):
                # fallback: take nearest point in front
                front_pts = pts[mask_front]
                if front_pts.size == 0:
                    continue
                # choose point with smallest x (closest)
                idx = np.argmin(front_pts[:,0])
                centroid = front_pts[idx]
            else:
                selected = pts[mask]
                centroid = np.mean(selected, axis=0)

            # compute local coordinates (x forward, y left)
            local_x, local_y, local_z = float(centroid[0]), float(centroid[1]), float(centroid[2])

            # compute global lat/lon if GNSS available
            lat, lon, alt = None, None, None
            if self.latest_gnss is not None and self.latest_gnss.status.status >= 0:
                lat0 = self.latest_gnss.latitude
                lon0 = self.latest_gnss.longitude
                alt0 = self.latest_gnss.altitude if hasattr(self.latest_gnss, 'altitude') else 0.0
                # map vehicle frame to north/east: assume x -> north, y -> left -> west, so east = -y
                north = local_x
                east = -local_y
                R = 6378137.0
                lat = lat0 + (north / R) * (180.0 / math.pi)
                lon = lon0 + (east / (R * math.cos(math.radians(lat0)))) * (180.0 / math.pi)
                alt = alt0 + local_z

            results.append({
                'class': cls,
                'local': {'x': local_x, 'y': local_y, 'z': local_z},
                'global': {'lat': lat, 'lon': lon, 'alt': alt},
                'conf': conf
            })

        # publish results as formatted string
        lines = []
        for r in results:
            g = r['global']
            if g['lat'] is not None:
                lines.append(f"{r['class']}: local(x={r['local']['x']:.2f}, y={r['local']['y']:.2f}, z={r['local']['z']:.2f}), global(lat={g['lat']:.7f}, lon={g['lon']:.7f}, alt={g['alt']:.2f}), conf={r['conf']:.2f}")
            else:
                lines.append(f"{r['class']}: local(x={r['local']['x']:.2f}, y={r['local']['y']:.2f}, z={r['local']['z']:.2f}), global(none), conf={r['conf']:.2f}")

        if lines:
            msg = "; ".join(lines)
            self.pub.publish(String(data=msg))

    @staticmethod
    def angle_diff(a,b):
        d = a - b
        d = (d + math.pi) % (2*math.pi) - math.pi
        return d

    def parse_detection_text(self, text):
        """
        Parse detection text messages emitted by PerceptionNode.

        PerceptionNode currently emits detections in a compact CSV-like format:
          "ClassName,x1,y1,x2,y2,conf; ClassName2,x1,y1,x2,y2,conf"

        Older/alternative format (kept for compatibility) is:
          "Class [x1, y1, x2, y2] (conf); ..."

        This parser supports both formats and returns a list of dicts:
          {'class': <str>, 'bbox': (x1,y1,x2,y2), 'conf': <float>}
        """
        if not text or text.strip().lower() == 'no detections':
            return []

        parts = [p.strip() for p in text.split(';') if p.strip()]
        detections = []

        # First, try to parse the compact CSV-like entries produced by PerceptionNode
        for p in parts:
            # split by commas - the expected CSV format has at least 6 fields
            csv_fields = [f.strip() for f in p.split(',')]
            if len(csv_fields) >= 6:
                try:
                    cls = csv_fields[0]
                    x1 = float(csv_fields[1])
                    y1 = float(csv_fields[2])
                    x2 = float(csv_fields[3])
                    y2 = float(csv_fields[4])
                    conf = float(csv_fields[5])
                    detections.append({'class': cls, 'bbox': (x1, y1, x2, y2), 'conf': conf})
                    continue
                except ValueError:
                    # fall through to regex parser if conversion fails
                    pass

            # Fallback: try the bracketed pattern "Class [x1, y1, x2, y2] (conf)"
            m = re.search(r'(?P<class>[\w ][\w ]*?)\s*\[\s*(?P<x1>[-+]?\d*\.?\d+)\s*,\s*(?P<y1>[-+]?\d*\.?\d+)\s*,\s*(?P<x2>[-+]?\d*\.?\d+)\s*,\s*(?P<y2>[-+]?\d*\.?\d+)\s*\]\s*\(?\s*(?P<conf>[-+]?\d*\.?\d+)\s*\)?', p)
            if m:
                try:
                    cls = m.group('class').strip()
                    x1 = float(m.group('x1'))
                    y1 = float(m.group('y1'))
                    x2 = float(m.group('x2'))
                    y2 = float(m.group('y2'))
                    conf = float(m.group('conf'))
                    detections.append({'class': cls, 'bbox': (x1, y1, x2, y2), 'conf': conf})
                except ValueError:
                    continue

        return detections


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

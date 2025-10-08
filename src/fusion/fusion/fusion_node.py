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
        # Robustly extract text from incoming String messages. Accept a few common
        # accidental prefixes (e.g. "data: ...") or surrounding quotes that may
        # appear if the topic output has been forwarded into another pipeline.
        raw = msg.data if msg is not None else ''
        text = str(raw).strip()

        # remove common "data:" prefix if someone pasted the topic echo output
        if text.lower().startswith('data:'):
            text = text.split(':', 1)[1].strip()

        # strip surrounding quotes if present
        if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
            text = text[1:-1].strip()

        self.get_logger().info(f"[detection_callback] raw message: '{text}'")
        print(f"[detection_callback] raw message: '{text}'")
        parsed = self.parse_detection_text(text)
        self.get_logger().info(f"[detection_callback] parsed {len(parsed)} detections: {parsed}")
        print(f"[detection_callback] parsed: {parsed}")

        # Always update latest_detections with the parsed result (may be empty).
        self.latest_detections = parsed
    
    def image_callback(self, msg: Image):
        # Only need width to map bbox center to angle
        self.latest_image_width = msg.width
        self.get_logger().info(f"[image_callback] image width set to {self.latest_image_width}")
        print(f"[image_callback] image width: {self.latest_image_width}")
    
    def gnss_callback(self, msg: NavSatFix):
        self.latest_gnss = msg
        st = getattr(msg, "status", None)
        status_val = st.status if st is not None else None
        self.get_logger().info(f"[gnss_callback] GNSS received status={status_val}, lat={msg.latitude:.7f}, lon={msg.longitude:.7f}, alt={getattr(msg,'altitude',0.0):.2f}")
        print(f"[gnss_callback] GNSS status={status_val}, lat={msg.latitude}, lon={msg.longitude}, alt={getattr(msg,'altitude',0.0)}")
    
    def pointcloud_callback(self, msg: PointCloud2):
        # store latest pointcloud message
        self.latest_pc = msg
        # log some header info to help debug
        try:
            frame = msg.header.frame_id
            width = getattr(msg, "width", None)
            height = getattr(msg, "height", None)
            point_step = getattr(msg, "point_step", None)
            row_step = getattr(msg, "row_step", None)
            self.get_logger().info(f"[pointcloud_callback] frame={frame}, width={width}, height={height}, point_step={point_step}, row_step={row_step}")
            print(f"[pointcloud_callback] frame={frame}, width={width}, height={height}, point_step={point_step}, row_step={row_step}")
        except Exception as e:
            self.get_logger().info(f"[pointcloud_callback] header read error: {e}")
            print(f"[pointcloud_callback] header read error: {e}")
    
    def timer_callback(self):
        self.get_logger().info("[timer_callback] triggered")
        print("[timer_callback] triggered")
        self.get_logger().info(f"[timer_callback] latest_detections_count={len(self.latest_detections)}, latest_pc_set={self.latest_pc is not None}, latest_image_width={self.latest_image_width}")
        print(f"[timer_callback] detections={self.latest_detections}")
    
        if not self.latest_detections:
            self.get_logger().info("[timer_callback] no detections to process")
            print("[timer_callback] no detections to process")
            return
        if self.latest_pc is None:
            self.get_logger().info("[timer_callback] no pointcloud available")
            print("[timer_callback] no pointcloud available")
            return
        if self.latest_image_width is None:
            self.get_logger().info("[timer_callback] no image width available")
            print("[timer_callback] no image width available")
            return
    
        # Convert pointcloud to numpy array of points
        points = []
        try:
            for p in pc2.read_points(self.latest_pc, field_names=('x','y','z'), skip_nans=True):
                points.append(p)
        except Exception as e:
            self.get_logger().info(f"[timer_callback] error reading points: {e}")
            print(f"[timer_callback] error reading points: {e}")
            return
    
        if not points:
            self.get_logger().info("[timer_callback] pointcloud contained no points after read_points")
            print("[timer_callback] pointcloud empty")
            return
        pts = np.array(points)  # Nx3
        self.get_logger().info(f"[timer_callback] pointcloud loaded, pts.shape={pts.shape}")
        print(f"[timer_callback] pts.shape={pts.shape}")
    
        # apply lidar offset (pointcloud reported in lidar frame -> transform to ego frame)
        try:
            pts = pts + self.lidar_offset.reshape((1,3))
        except Exception as e:
            self.get_logger().info(f"[timer_callback] error applying lidar offset: {e}")
            print(f"[timer_callback] error applying lidar offset: {e}")
            return
    
        results = []
        for idx_det, det in enumerate(self.latest_detections):
            try:
                cls = det['class']
                x1,y1,x2,y2 = det['bbox']
                conf = det['conf']
            except Exception as e:
                self.get_logger().info(f"[timer_callback] invalid detection format at index {idx_det}: {e}")
                print(f"[timer_callback] invalid detection format at index {idx_det}: {e}")
                continue
    
            cx = (x1 + x2)/2.0
            w = (x2 - x1)

            # Prefer the width from the latest annotated image. If not available,
            # try to estimate it from detection bboxes (useful when annotated
            # image messages are not always published).
            img_w = self.latest_image_width
            if not img_w or img_w == 0:
                max_x2 = max((d['bbox'][2] for d in self.latest_detections), default=0)
                if max_x2 > 0:
                    img_w = max_x2 + 1.0
                    self.get_logger().info(f"[timer_callback] estimated image width from detections: {img_w}")
                    print(f"[timer_callback] estimated image width: {img_w}")
                else:
                    self.get_logger().info("[timer_callback] no image width available and cannot estimate, skipping detection")
                    print("[timer_callback] no image width available")
                    continue
    
            # map pixel center to horizontal angle
            rel = (cx - img_w/2.0)/img_w  # -0.5..0.5
            angle_center = rel * self.hfov_rad
            # width based angular tolerance
            angle_tol = max(abs(w)/img_w * self.hfov_rad * 0.5, self.match_angle_padding)
    
            self.get_logger().info(f"[timer_callback] det#{idx_det} class={cls} bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) conf={conf:.2f} cx={cx:.1f} angle_center={math.degrees(angle_center):.2f}deg angle_tol={math.degrees(angle_tol):.2f}deg")
            print(f"[timer_callback] det#{idx_det} class={cls} cx={cx} angle_center(deg)={math.degrees(angle_center):.2f} angle_tol(deg)={math.degrees(angle_tol):.2f}")
    
            # compute azimuth for each point (atan2(y,x))
            az = np.arctan2(pts[:,1], pts[:,0])
            # select points roughly in front (x>0)
            mask_front = pts[:,0] > 0.2
            mask_angle = np.abs(self.angle_diff(az, angle_center)) < angle_tol
            mask = mask_front & mask_angle
            self.get_logger().info(f"[timer_callback] det#{idx_det} mask_front_count={np.count_nonzero(mask_front)} mask_angle_count={np.count_nonzero(mask_angle)} mask_count={np.count_nonzero(mask)}")
            print(f"[timer_callback] mask_front_count={np.count_nonzero(mask_front)} mask_angle_count={np.count_nonzero(mask_angle)} mask_count={np.count_nonzero(mask)}")
    
            if not np.any(mask):
                # fallback: take nearest point in front
                front_pts = pts[mask_front]
                if front_pts.size == 0:
                    self.get_logger().info(f"[timer_callback] det#{idx_det} no front points available, skipping")
                    print(f"[timer_callback] det#{idx_det} no front points available")
                    continue
                # choose point with smallest x (closest)
                idx = np.argmin(front_pts[:,0])
                centroid = front_pts[idx]
                self.get_logger().info(f"[timer_callback] det#{idx_det} fallback centroid chosen from front_pts idx={idx} centroid={centroid}")
                print(f"[timer_callback] det#{idx_det} fallback centroid={centroid}")
            else:
                selected = pts[mask]
                centroid = np.mean(selected, axis=0)
                self.get_logger().info(f"[timer_callback] det#{idx_det} selected {selected.shape[0]} pts centroid={centroid}")
                print(f"[timer_callback] det#{idx_det} centroid={centroid}")
    
            # compute local coordinates (x forward, y left)
            local_x, local_y, local_z = float(centroid[0]), float(centroid[1]), float(centroid[2])
            self.get_logger().info(f"[timer_callback] det#{idx_det} local coords x={local_x:.2f}, y={local_y:.2f}, z={local_z:.2f}")
            print(f"[timer_callback] det#{idx_det} local x={local_x}, y={local_y}, z={local_z}")
    
            # compute global lat/lon if GNSS available
            lat, lon, alt = None, None, None
            if self.latest_gnss is not None and getattr(self.latest_gnss, "status", None) is not None and self.latest_gnss.status.status >= 0:
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
                self.get_logger().info(f"[timer_callback] det#{idx_det} global approx lat={lat:.7f}, lon={lon:.7f}, alt={alt:.2f}")
                print(f"[timer_callback] det#{idx_det} global lat={lat}, lon={lon}, alt={alt}")
            else:
                self.get_logger().info(f"[timer_callback] det#{idx_det} no valid GNSS to compute global coords")
                print(f"[timer_callback] det#{idx_det} no GNSS")
    
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
            self.get_logger().info(f"[timer_callback] publishing {len(lines)} fused results")
            print(f"[timer_callback] publishing message: {msg}")
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

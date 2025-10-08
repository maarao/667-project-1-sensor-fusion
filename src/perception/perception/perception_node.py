import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2



class PerceptionNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # Create CvBridge
        self.bridge = CvBridge()

        # Load YOLO model
        #self.model = YOLO("src/models/yolo11n.engine")
        self.model = YOLO("src/models/yolo11n.pt")
        
        # Allowed classes to report/draw (normalized to lowercase with underscores replaced)
        self.allowed_class_names = {
            "person", "car", "truck", "bus", "motorbike", "motorcycle",
            "bicycle", "van", "vehicle", "stop sign", "stop_sign", "stop"
        }
        
        # Declare parameters for topic names
        #self.declare_parameter("image_topic", "/camera/camera/color/image_raw") # For Real Camera
        self.declare_parameter("image_topic", "/sensing/camera/camera0/image_rect_color") # For Simulation
        self.declare_parameter("inference_image_topic", "/yolo/inference_image")
        self.declare_parameter("detection_topic", "/yolo/detections")

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT
        inference_image_topic = self.get_parameter("inference_image_topic").get_parameter_value().string_value
        detection_topic = self.get_parameter("detection_topic").get_parameter_value().string_value

        # Subscribers
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile
        )

        # Publishers
        self.image_publisher = self.create_publisher(
            Image,
            inference_image_topic,
            10
        )
        self.detection_publisher = self.create_publisher(
            String,
            detection_topic,
            10
        )

        self.get_logger().info(
            f"YOLO Inference Node Started. Subscribed to {image_topic}, "
            f"publishing annotated image to {inference_image_topic} and detections to {detection_topic}"
        )

    def image_callback(self, msg):
        self.get_logger().info("Received image for inference.")
        try:
            # Convert ROS Image message to OpenCV image
            if msg.encoding == '8UC4': # For Simulation
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # Optionally convert to BGR if needed:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
            else: # For Real Camera
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge exception: {e}")
            return
        
        # Run YOLO inference
        results = self.model(cv_image)
        annotated_image = cv_image.copy()
        detected_info = []
        if results:
            for result in results:
                if result.boxes and result.boxes.cls is not None:
                    cls_ids = result.boxes.cls.cpu().numpy()
                    bboxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()

                    for cls_id, bbox, conf in zip(cls_ids, bboxes, confs):
                        class_name = self.model.names[int(cls_id)]
                        x1, y1, x2, y2 = map(int, bbox)
                        # normalize class name for matching
                        normalized_name = str(class_name).lower().replace('_', ' ')
                        if normalized_name in self.allowed_class_names:
                            # Draw rectangle
                            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
                            
                            # Put label
                            label = f"{class_name}: {conf:.2f}"
                            cv2.putText(annotated_image, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 0, 0), 2)
                            
                            # Add detected info to the list
                            detected_info.append(
                                f"{class_name},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{conf:.2f}"
                            )
                # Publish annotated image
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                annotated_msg.header.frame_id = "annotated_frame"
                self.image_publisher.publish(annotated_msg)
                self.get_logger().info("Published annotated inference image.")

        # 2️⃣ Prepare detection text message for control logic
        # detected_info = []
        # if results:
        #     for result in results:
        #         if result.boxes and result.boxes.cls is not None:
        #             cls_ids = result.boxes.cls.cpu().numpy()
        #             bboxes = result.boxes.xyxy.cpu().numpy()
        #             confs = result.boxes.conf.cpu().numpy()

        #             for cls_id, bbox, conf in zip(cls_ids, bboxes, confs):
        #                 class_name = self.model.names[int(cls_id)]
        #                 x1, y1, x2, y2 = bbox
        #                 normalized_name = str(class_name).lower().replace('_', ' ')
        #                 if normalized_name in self.allowed_class_names:
        #                     detected_info.append(
        #                         f"{class_name},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{conf:.2f}"
        #                     )

        if detected_info:
            detection_msg = "; ".join(detected_info)
            self.detection_publisher.publish(String(data=detection_msg))
            self.get_logger().info(f"Published detections: {detection_msg}")
        else:
            detection_msg = "no detections"
            self.detection_publisher.publish(String(data=detection_msg))
            self.get_logger().info(f"No detections")

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

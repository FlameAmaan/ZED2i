#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('model_path', '/home/flame/ros2_ws/src/object_Detector/best.pt')
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('show', True)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf = float(self.get_parameter('conf').value)
        self.show = bool(self.get_parameter('show').value)

        self.model = YOLO(model_path)
        self.class_names = self.model.names

        self.bridge = CvBridge()
        self.depth_frame = None

        # Subscriptions
        self.create_subscription(Image, image_topic, self.callback, 10)
        self.create_subscription(Image, depth_topic, self.depth_callback, 10)

        self.pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)

        det_array = Detection2DArray()
        det_array.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.class_names[cls]

                det = Detection2D()
                det.bbox.center.position.x = (x1 + x2) / 2.0
                det.bbox.center.position.y = (y1 + y2) / 2.0
                det.bbox.size_x = float(x2 - x1)
                det.bbox.size_y = float(y2 - y1)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = label
                hyp.hypothesis.score = conf

                if self.depth_frame is not None:
                    cx, cy = int(det.bbox.center.position.x), int(det.bbox.center.position.y)
                    if 0 <= cy < self.depth_frame.shape[0] and 0 <= cx < self.depth_frame.shape[1]:
                        depth_val = float(self.depth_frame[cy, cx])
                        hyp.pose.pose.position.z = depth_val  # meters

                det.results.append(hyp)
                det_array.detections.append(det)

                if self.show:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    display_text = f"{label} {conf:.2f}"
                    if hyp.pose.pose.position.z:
                        display_text += f" {hyp.pose.pose.position.z:.2f}m"
                    cv2.putText(frame, display_text, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        self.pub.publish(det_array)

        if self.show:
            cv2.imshow("YOLOv8 Detections with Depth", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()

def main():
    rclpy.init()
    node = YOLOv8Node()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

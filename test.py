
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import pyzed.sl as sl
import os

# Open the camera
init_params = sl.InitParameters()
cam = sl.Camera()
status = cam.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open failed:", status)
    exit(1)

# Get camera info
cam_info = cam.get_camera_information()

# Get intrinsics for left camera
calib = cam_info.camera_configuration.calibration_parameters
left = calib.left_cam
right = calib.right_cam
leftFx=left.fx
leftFy=left.fy
rightFx=right.fx

CxRight=left.cx
CyRight=left.cy
CxLeft=left.cx
CyLeft=left.cy

cam.close()


import time
print("sleeping for 5 secs")
time.sleep(5)



class YOLOv8Node(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # Parameters
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/camera_info')
        self.declare_parameter('model_path', '/home/flameamaan/ros2_ws/src/ZED2i/yolov8_best_FineTune.pt')
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('show', True)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf = float(self.get_parameter('conf').value)
        self.show = bool(self.get_parameter('show').value)

        # Load YOLO model
        self.model = YOLO(model_path)
        self.class_names = self.model.names

        # Utils
        self.bridge = CvBridge()
        self.depth_frame = None

        # Intrinsics
        self.fx = leftFx
        self.fy = leftFy
        self.cx = CxLeft
        self.cy = CyLeft

        # Subscriptions
        self.create_subscription(Image, image_topic, self.callback, 10)
        self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 1)

        # Publisher
        self.pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)

    def camera_info_callback(self, msg: CameraInfo):
        """Store intrinsics from ZED camera info"""
        self.fx = leftFx
        self.fy = leftFy
        self.cx = CxLeft
        self.cy = CyLeft

    def depth_callback(self, msg):
        """Convert depth image to cv2 format"""
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def callback(self, msg:Image):
        """Main RGB callback for detection"""
        if self.fx is None:
            self.get_logger().warn("Waiting for camera intrinsics...")
            return

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

                # Bounding box
                det = Detection2D()
                det.bbox.center.position.x = (x1 + x2) / 2.0
                det.bbox.center.position.y = (y1 + y2) / 2.0
                det.bbox.size_x = float(x2 - x1)
                det.bbox.size_y = float(y2 - y1)

                # Hypothesis with pose
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = label
                hyp.hypothesis.score = conf

                # Depth to 3D
                if self.depth_frame is not None:
                    u, v = int(det.bbox.center.position.x), int(det.bbox.center.position.y)
                    if 0 <= v < self.depth_frame.shape[0] and 0 <= u < self.depth_frame.shape[1]:
                        Z = float(self.depth_frame[v, u])  # meters
                        if Z > 0:  # valid depth
                            X = (u - self.cx) * Z / self.fx
                            Y = (v - self.cy) * Z / self.fy
                            hyp.pose.pose.position.x = X
                            hyp.pose.pose.position.y = Y
                            hyp.pose.pose.position.z = Z

                det.results.append(hyp)
                det_array.detections.append(det)

                # GUI drawing
                if self.show:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if hyp.pose.pose.position.z:  # Show 3D coords if available
                        X = hyp.pose.pose.position.x
                        Y = hyp.pose.pose.position.y
                        Z = hyp.pose.pose.position.z
                        display_text = f"{label}  Z={Z:.2f}m  ({X:.2f},{Y:.2f},{Z:.2f})"
                        #front 
                    else:
                        display_text = f"{label}  Z=?"

                    cv2.putText(frame, display_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Publish detection array
        self.pub.publish(det_array)

        # Show GUI
        if self.show:
            cv2.imshow("YOLOv8 Detections with Depth & 3D", frame)
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

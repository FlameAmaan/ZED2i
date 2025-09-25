#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from sensor_msgs.msg import Image, CameraInfo, Imu
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import math
from nav_msgs.msg import Odometry

def quaternion_to_euler(x, y, z, w):
    """Convert quaternion (x,y,z,w) to roll, pitch, yaw in radians"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class YOLOv8IMUNode(Node):
    def __init__(self):
        super().__init__('yolov8_imu_detector')

        # Parameters
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/camera_info')
        self.declare_parameter('imu_topic', '/zed/zed_node/imu/data')
        self.declare_parameter('model_path', '/home/flameamaan/ros2_ws/src/ZED2i/yolov8_best_FineTune.pt')
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('show', True)

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf = float(self.get_parameter('conf').value)
        self.show = bool(self.get_parameter('show').value)
        self.camera_position = np.zeros(3)  # t_cam: x, y, z
        self.create_subscription(Odometry, '/zed/zed_node/odom', self.odom_callback, 10)
        # YOLO model
        self.model = YOLO(model_path)
        self.class_names = self.model.names

        self.bridge = CvBridge()
        self.depth_frame = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # IMU orientation
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Subscribers
        self.create_subscription(Image, self.image_topic, self.callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 1)
        self.create_subscription(Imu, self.imu_topic, self.imu_callback, 10)

        # Publisher
        self.pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)

    def imu_callback(self, msg: Imu):
        self.roll, self.pitch, self.yaw = quaternion_to_euler(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )
    def odom_callback(self, msg: Odometry):
    # Camera position in world frame
        self.camera_position[0] = msg.pose.pose.position.x
        self.camera_position[1] = msg.pose.pose.position.y
        self.camera_position[2] = msg.pose.pose.position.z

    # Optional: combine rotation from odometry if you prefer

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.p[0]
        self.fy = msg.p[5]
        self.cx = msg.p[2]
        self.cy = msg.p[6]

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def callback(self, msg: Image):
        if self.fx is None:
            self.get_logger().warn("Waiting for camera intrinsics...")
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)

        det_array = Detection2DArray()
        det_array.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)

        # Build rotation matrix from IMU
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.roll), -np.sin(self.roll)],
            [0, np.sin(self.roll), np.cos(self.roll)]
        ])
        Ry = np.array([
            [np.cos(self.pitch), 0, np.sin(self.pitch)],
            [0, 1, 0],
            [-np.sin(self.pitch), 0, np.cos(self.pitch)]
        ])
        Rz = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), 0],
            [np.sin(self.yaw), np.cos(self.yaw), 0],
            [0, 0, 1]
        ])
        R_imu = Rz @ Ry @ Rx

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

                # Depth to 3D (camera frame)
                if self.depth_frame is not None:
                    u, v = int(det.bbox.center.position.x), int(det.bbox.center.position.y)
                    if 0 <= v < self.depth_frame.shape[0] and 0 <= u < self.depth_frame.shape[1]:
                        dist = float(self.depth_frame[v, u])
                        if dist > 0:
                            X = (u - self.cx) / self.fx
                            Y = (v - self.cy) / self.fy
                            Z = 1.0
                            point_cam = np.array([X, Y, Z])
                            point_cam = point_cam / np.linalg.norm(point_cam) * dist

                            # Transform to world frame using IMU
                            point_world = R_imu @ point_cam + self.camera_position
                            a_x, a_y, a_z = point_world

                            hyp.pose.pose.position.x = a_x
                            hyp.pose.pose.position.y = a_y
                            hyp.pose.pose.position.z = a_z

                det.results.append(hyp)
                det_array.detections.append(det)

                if self.show:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if hyp.pose.pose.position.z: 
                        X, Y, Z = hyp.pose.pose.position.x, hyp.pose.pose.position.y, hyp.pose.pose.position.z
                        display_text = f"{label}  Z={Z:.2f}m  ({X:.2f},{Y:.2f},{Z:.2f})"
                    else:
                        display_text = f"{label}  Z=?"
                    cv2.putText(frame, display_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.pub.publish(det_array)

        if self.show:
            cv2.imshow("YOLOv8 Detections with Depth & 3D", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()


def main():
    rclpy.init()
    node = YOLOv8IMUNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

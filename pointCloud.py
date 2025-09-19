#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import pyzed.sl as sl
import sensor_msgs_py.point_cloud2 as pc2
import time

# --- Initialize ZED once to grab calibration
init_params = sl.InitParameters()
cam = sl.Camera()
status = cam.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open failed:", status)
    exit(1)

cam_info = cam.get_camera_information()
calib = cam_info.camera_configuration.calibration_parameters
left = calib.left_cam
leftFx, leftFy, CxLeft, CyLeft = left.fx, left.fy, left.cx, left.cy
cam.close()

print("Sleeping 5s")
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
        self.pointcloud = None

        # Intrinsics
        self.fx, self.fy, self.cx, self.cy = leftFx, leftFy, CxLeft, CyLeft

        # Subscriptions
        self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 1)
        self.create_subscription(PointCloud2, '/zed/zed_node/point_cloud/cloud_registered',self.pointcloud_callback, 10)

        # Publishers
        self.pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.cone_pub = self.create_publisher(PointStamped, '/yolo/cone_positions', 10)


    def pointcloud_callback(self, msg: PointCloud2):
        self.pointcloud = msg

    def camera_info_callback(self, msg: CameraInfo):
        """Update intrinsics (if needed)"""
        self.fx, self.fy, self.cx, self.cy = leftFx, leftFy, CxLeft, CyLeft

    def depth_callback(self, msg: Image):
        """Convert depth image to cv2 format"""
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def image_callback(self, msg: Image):   
        """Main RGB callback for detection + point cloud lookup"""
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

                # Hypothesis
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = label
                hyp.hypothesis.score = conf

                # -------- Get 3D coords from PointCloud2 --------
                if self.pointcloud is not None:
                    u, v = int(det.bbox.center.position.x), int(det.bbox.center.position.y)

                    # Small region around bbox center (5x5)
                    region = [(int(u + dx), int(v + dy)) for dx in range(-2, 3) for dy in range(-2, 3)]
                    flat_region = [coord for uv in region for coord in uv]
                    points = list(pc2.read_points(
                        self.pointcloud,
                        field_names=("x", "y", "z"),
                        skip_nans=True,
                        uvs=flat_region
                    ))

                    if points:
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        zs = [p[2] for p in points]

                        if xs and ys and zs:
                            X = float(np.mean(xs))
                            Y = float(np.mean(ys))
                            Z = float(np.mean(zs))

                        hyp.pose.pose.position.x = float(X)
                        hyp.pose.pose.position.y = float(Y)
                        hyp.pose.pose.position.z = float(Z)

                        if label.lower() == "cone":
                            self.get_logger().info(f"CONE at (X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}) m")

                            # Publish cone coordinates
                            cone_msg = PointStamped()
                            cone_msg.header = det_array.header
                            cone_msg.point.x = float(X)
                            cone_msg.point.y = float(Y)
                            cone_msg.point.z = float(Z)
                            self.cone_pub.publish(cone_msg)
                    else:
                        self.get_logger().warn(f"No valid 3D point at ({u},{v})")

                det.results.append(hyp)
                det_array.detections.append(det)

                # -------- GUI Drawing --------
                if self.show:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if hyp.pose.pose.position.z:  # Show 3D coords if available
                        X = hyp.pose.pose.position.x
                        Y = hyp.pose.pose.position.y
                        Z = hyp.pose.pose.position.z
                        display_text = f"{label} {conf:.2f} Z={Z:.2f}m ({X:.2f},{Y:.2f},{Z:.2f})"
                    else:
                        display_text = f"{label} {conf:.2f} Z=?"
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

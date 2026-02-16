
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
import matplotlib.pyplot as plt
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.time import Time



class YOLOv8Node(Node):
    def __init__(self):

        super().__init__('yolov8_detector')
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.world_cones = []

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Z (meters)")
        self.ax.set_title("Cone Map (Top View)")
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(0, 10)
        self.scatter = self.ax.scatter([], [])
        self.cone_positions = []

        self.declare_parameter('image_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/color/rect/image/camera_info')
        self.declare_parameter('model_path', '//home/flame/ZED2i/yolov8_best_FineTune.pt')
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('show', True)
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf = float(self.get_parameter('conf').value)
        self.show = bool(self.get_parameter('show').value)

        self.model = YOLO(model_path)
        self.class_names = self.model.names

        self.bridge = CvBridge()
        self.depth_frame = None

        self.create_subscription(Image, image_topic, self.callback, 10)
        self.create_subscription(Image, depth_topic, self.depth_callback, 10)

        self.pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 1)

    def camera_info_callback(self, msg: CameraInfo):
        #get intrinsics 
        self.fx=msg.p[0]
        self.fy=msg.p[5]
        self.cx=msg.p[2]
        self.cy=msg.p[6]

    def depth_callback(self, msg):
        # Convert depth image to cv2
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def callback(self, msg:Image):
        self.get_logger().info("RGB received")

        #Main RGB callback for detection
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

                # Hypothesis with pose(for detection)
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = label
                hyp.hypothesis.score = conf 

                # Depth to 3D
                if self.depth_frame is not None:
                    u, v = int(det.bbox.center.position.x), int(det.bbox.center.position.y)
                    if 0 <= v < self.depth_frame.shape[0] and 0 <= u < self.depth_frame.shape[1]:
                        dist = float(self.depth_frame[v, u])  # meters
                        if dist > 0:
                            X = (u - self.cx) / self.fx
                            Y = (v - self.cy) / self.fy
                            Z = 1.0
                            Coord = np.array([X, Y, Z])
                            Coord = Coord / np.linalg.norm(Coord) * dist
                            a_x, a_y, a_z = Coord
                            hyp.pose.pose.position.x = float(a_x)
                            hyp.pose.pose.position.y = float(a_y)
                            hyp.pose.pose.position.z = float(a_z)
                            point_cam = PointStamped()
                            point_cam.header.frame_id = "zed_camera_center"
                            point_cam.header.stamp = Time().to_msg()

                            point_cam.point.x = float(a_x)
                            point_cam.point.y = float(a_y)
                            point_cam.point.z = float(a_z)

                            try:
                                point_world = self.tf_buffer.transform(
                                    point_cam,
                                    "map",
                                    timeout=Duration(seconds=0.1)
                                )

                                wx = point_world.point.x
                                wy = point_world.point.y

                                # Avoid duplicates (optional but recommended)
                                if not any((wx - x)**2 + (wy - y)**2 < 1.0**2 for x, y in self.world_cones):
                                    self.world_cones.append((wx, wy))

                            except Exception as e:
                                print("TF failed:", e)


                det.results.append(hyp)
                det_array.detections.append(det)

                if self.show:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if hyp.pose.pose.position.z!=0: 
                        X = hyp.pose.pose.position.x
                        Y = hyp.pose.pose.position.y
                        dist = hyp.pose.pose.position.z
                        display_text = f"{label}  Z={dist:.2f}m  ({X:.2f},{Y:.2f},{dist:.2f})"  #X->left-right, Y->down-up, Z->forward
                    else:
                        display_text = f"{label}  Z=?"

                    cv2.putText(frame, display_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.world_cones:
            xs = [p[0] for p in self.world_cones]
            zs = [p[1] for p in self.world_cones]

            self.ax.clear()

            self.ax.set_xlabel("X (meters)")
            self.ax.set_ylabel("Z (meters)")
            self.ax.set_aspect('equal', 'box')
            self.ax.set_title("Cone Map (Top View)")
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
            self.ax.scatter(zs, xs, c='orange')
            print("gello")
            plt.draw()
            plt.pause(0.001)
            # self.cone_positions.clear()

        self.pub.publish(det_array)

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

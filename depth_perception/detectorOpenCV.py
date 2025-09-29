"""
This code translates the Cone coordinates and displays them on OpenCV,
and plots camera-frame cone positions on demand.
"""

#!/usr/bin/env python3
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ----------------------------
# Initialize ZED
# ----------------------------
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED")
    exit(1)

# Enable positional tracking for camera translation
tracking_params = sl.PositionalTrackingParameters()
tracking_params.enable_imu_fusion = True
status = zed.enable_positional_tracking(tracking_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to enable positional tracking")
    zed.close()
    exit(1)

runtime_params = sl.RuntimeParameters()
left_image = sl.Mat()
depth_map = sl.Mat()

# ----------------------------
# Load YOLO model
# ----------------------------
model = YOLO("/home/gera/perception/Perception-master/zed-2i_ROS2/depth_perception/yolov8_best_FineTune.pt")

# ----------------------------
# Main loop
# ----------------------------
camera_pose = sl.Pose()
py_translation = sl.Translation()

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # Get left frame + depth
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.XYZRGBA)
        frame = left_image.get_data()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Get camera translation (world frame)
        tracking_state = zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
        if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
            translation = camera_pose.get_translation(py_translation)
            world_X, world_Y, world_Z = translation.get()
        else:
            world_X, world_Y, world_Z = 0, 0, 0

        # Run YOLO inference
        results = model(frame_bgr, verbose=False)[0]

        # Store detections for plotting
        cones_cam_coords = []

        for box in results.boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Center of bbox
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Get camera-frame 3D coordinates (depth)
            success, point = depth_map.get_value(cx, cy)
            if success == sl.ERROR_CODE.SUCCESS and point is not None:
                cam_X, cam_Y, cam_Z, _ = point
                X_cm, Z_cm = cam_X * 100, cam_Z * 100
                cones_cam_coords.append((X_cm, Z_cm, label))
            else:
                cam_X, cam_Y, cam_Z = 0, 0, 0

            # Draw bbox + labels
            cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Overlay both coordinates
            text_y = int(y2) + 15
            cv2.putText(frame_bgr, f"Cam Frame: ({cam_X:.2f},{cam_Y:.2f},{cam_Z:.2f})m",
                        (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(frame_bgr, f"World Frame: ({world_X:.2f},{world_Y:.2f},{world_Z:.2f})m",
                        (int(x1), text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # Show result
        cv2.imshow("YOLO + ZED 3D", frame_bgr)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p') and cones_cam_coords:
            xs, zs, colors = [], [], []
            for X_cm, Z_cm, lbl in cones_cam_coords:
                xs.append(X_cm)
                zs.append(Z_cm)
                if "blue" in lbl.lower():
                    colors.append("blue")
                elif "yellow" in lbl.lower():
                    colors.append("gold")
                else:
                    colors.append("gray")

            plt.figure()
            plt.scatter(xs, zs, c=colors, s=40, edgecolors="k")
            plt.xlabel("X (cm) [sideways]")
            plt.ylabel("Z (cm) [forward]")
            plt.title("Detected Cones in Camera Frame")
            plt.grid(True)
            plt.axis("equal")
            plt.show()

        elif key == ord('q'):
            break

# ----------------------------
# Cleanup
# ----------------------------
zed.close()
cv2.destroyAllWindows()

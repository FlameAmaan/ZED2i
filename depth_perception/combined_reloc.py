#!/usr/bin/env python3


import os
import traceback
from pathlib import Path
from threading import Lock, Thread
from time import sleep
import ogl_viewer.viewer as gl
import cv2
import numpy as np
import torch
import argparse
import pyzed.sl as sl
from ultralytics import YOLO
import json
import time
import cv_viewer.tracking_viewer as cv_viewer
import importlib.util
import traceback as _tb
from OpenGL.GL import *

lock = Lock()
run_signal = False
exit_signal = False
detections = []
image_net = None
MODEL_PATH = None
MODEL_NAMES = None
skip_pointcloud_update = False


def get_pose_transform(pose_obj):
    """Return a sl.Transform for a given sl.Pose across pyzed API variations.
    Tries return-value form first, then out-param form.
    Raises RuntimeError on failure with original exception attached."""
    try:
        return pose_obj.pose_data()
    except Exception:
        try:
            tr = sl.Transform()
            pose_obj.pose_data(tr)
            return tr
        except Exception as e:
            raise RuntimeError(f"Could not obtain Transform from Pose: {e}")


def init_yolo_model(model_path: Path):
    model_path_str = str(model_path)
    if not os.path.exists(model_path_str):
        raise FileNotFoundError(f"Model not found at {model_path_str}")
    try:
        model = YOLO(model_path_str)
        if torch.cuda.is_available():
            model.to('cuda')
        else:
            print("CUDA not available: running YOLO on CPU")
        return model
    except Exception as e:
        print(f"Error loading model with ultralytics.YOLO: {e}")
        traceback.print_exc()
        # fallback
        return YOLO('yolov8n.pt')


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))
    x_min = (xywh[0] - 0.5 * xywh[2])
    x_max = (xywh[0] + 0.5 * xywh[2])
    y_min = (xywh[1] - 0.5 * xywh[3])
    y_max = (xywh[1] + 0.5 * xywh[3])
    output[0][0] = x_min
    output[0][1] = y_min
    output[1][0] = x_max
    output[1][1] = y_min
    output[2][0] = x_max
    output[2][1] = y_max
    output[3][0] = x_min
    output[3][1] = y_max
    return output


def detections_to_custom_box(dets, im0, model_names):
    out = []
    for det in dets:
        xywh = det.xywh[0]
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        # prefer readable label name if available
        try:
            cls_idx = int(det.cls)
            obj.label = model_names[cls_idx] if model_names is not None and cls_idx < len(model_names) else cls_idx
        except Exception:
            obj.label = det.cls
        obj.probability = float(det.conf)
        # mark as grounded so the SDK attempts to estimate a 3D position when possible
        obj.is_grounded = True
        out.append(obj)
    return out


def torch_thread(img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections, MODEL_PATH, MODEL_NAMES

    print("Initializing Network thread...")
    try:
        model = init_yolo_model(MODEL_PATH)
        MODEL_NAMES = tuple(model.names.values()) if isinstance(model.names, dict) else tuple(model.names)
        print(f"Model loaded with {len(MODEL_NAMES)} classes")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit_signal = True
        return

    while not exit_signal:
        if run_signal:
            lock.acquire()
            try:
                img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
                # Use ultralytics predict – limit to a single inference per capture
                res = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)
                if len(res) > 0:
                    det = res[0].cpu().numpy().boxes
                    detections = detections_to_custom_box(det, image_net, MODEL_NAMES)
                else:
                    detections = []
            except Exception as e:
                print(f"Detection error: {e}")
                detections = []
            finally:
                lock.release()
                run_signal = False
        sleep(0.005)


def main():
    global image_net, exit_signal, run_signal, detections, MODEL_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8_best_scratch.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file, if not passed, use the plugged camera')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='iou threshold')
    opt = parser.parse_args()

    MODEL_PATH = Path(opt.weights)

    # start detection thread
    try:
        t = Thread(target=torch_thread, kwargs={
            'img_size': opt.img_size,
            'conf_thres': opt.conf_thres,
            'iou_thres': opt.iou_thres
        }, daemon=True)
        t.start()
    except Exception as e:
        print(f"Failed to start detection thread: {e}")
        return

    # Initialize ZED
    zed = sl.Camera()
    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)
    init_params = sl.InitParameters(input_t=input_type)
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 15000
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    if opt.svo is not None:
        init_params.svo_real_time_mode = False

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera open failed:", repr(status))
        return

    # Positional tracking
    pt_params = sl.PositionalTrackingParameters()
    pt_params.set_as_static = False
    zed.enable_positional_tracking(pt_params)

    # Try to enable Area Memory (wrapped - adapt API if needed)
    area_memory = None
    saved_area_ids = []
    try:
        # Preferred API: enable area memory with parameters and get area memory handle
        amp = sl.AreaMemoryParameters()
        zed.enable_area_memory(amp)
        area_memory = zed.get_area_memory()
        print("Area Memory enabled")
    except Exception as e:
        # fallback: some SDK versions may have direct enable without parameters or different names
        try:
            zed.enable_area_memory()
            area_memory = zed.get_area_memory()
            print("Area Memory enabled (fallback call)")
        except Exception:
            print("Area Memory not available or different API on this pyzed build. Area memory operations will be skipped.")
            # print exception for debugging
            print(f"Area memory init error: {e}")

    # Enable object detection (custom boxes ingestion)
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False
    zed.enable_object_detection(obj_param)

    runtime_params = sl.RuntimeParameters()
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    # Viewer (GL)
    try:
        viewer = gl.GLViewer()
        point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
        viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
        print("GL viewer initialized successfully")
    except Exception as e:
        print(f"OpenGL/GLUT initialization failed: {e}")
        viewer = None
        print("GL viewer not available; running with 2D windows only")

    # Try to initialize the positional-tracking viewer (to display global landmarks)
    viewer_pt = None
    try:
        pt_path = "/home/gera/perception/positional tracking/positional tracking/python/ogl_viewer/tracking_viewer.py"
        spec = importlib.util.spec_from_file_location("pt_tracking_viewer", pt_path)
        pt_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pt_mod)
        try:
            viewer_pt = pt_mod.GLViewer()
            viewer_pt.init(camera_infos.camera_model)
            # create a landmarks object on the positional viewer for dynamic points
            viewer_pt.landmarks = pt_mod.Simple3DObject(False)
            viewer_pt.landmarks.set_drawing_type(GL_POINTS)
            viewer_pt.landmarks.push_to_GPU()
            viewer_pt.landmark_point_size = 8.0

            # Monkey-patch draw to render landmarks after normal draw
            orig_draw = viewer_pt.draw
            def patched_draw():
                try:
                    orig_draw()
                except Exception:
                    pass
                try:
                    glUseProgram(viewer_pt.shader_image.get_program_id())
                    vpMatrix = viewer_pt.camera.getViewProjectionMatrixRT(viewer_pt.pose)
                    glUniformMatrix4fv(viewer_pt.shader_MVP, 1, GL_FALSE, (GLfloat * len(vpMatrix))(*vpMatrix))
                    glPointSize(viewer_pt.landmark_point_size)
                    viewer_pt.landmarks.draw()
                    glUseProgram(0)
                except Exception:
                    pass
            viewer_pt.draw = patched_draw
            print("Positional tracking viewer initialized successfully")
        except Exception as e:
            print(f"Failed to instantiate positional tracking viewer: {e}")
            viewer_pt = None
    except Exception as e:
        print(f"Could not load positional tracking viewer module: {e}")
        print(_tb.format_exc())
        viewer_pt = None

    image_left_tmp = sl.Mat()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    cam_w_pose = sl.Pose()
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # compute scale from camera native resolution to display resolution for 2D renderer
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]

    # helper path for a simple shared map of cones
    cone_map_path = '/tmp/cone_map.json'

    # Main loop
    while (viewer.is_available() if viewer else True) and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # get image for detection
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True
            # wait for detection to complete
            while run_signal and not exit_signal:
                sleep(0.001)
            

            # ingest detections into ZED
            lock.acquire()
            try:
                if detections:
                    zed.ingest_custom_box_objects(detections)
            except Exception as e:
                print(f"Failed to ingest custom boxes: {e}")
            lock.release()

            # retrieve objects from ZED SDK
            objects = sl.Objects()
            obj_runtime = sl.ObjectDetectionRuntimeParameters()
            zed.retrieve_objects(objects, obj_runtime)

            # If any object labelled "orange" (case-insensitive) exists -> try to save area memory
            orange_found = False
            for obj in objects.object_list:
                label = getattr(obj, 'label', None)
                # label may be numeric or string; normalize to string
                try:
                    lab_str = str(label).lower()
                except Exception:
                    lab_str = ""
                if "orange" in lab_str:
                    orange_found = True
                    break

            if orange_found and area_memory is not None:
                # attempt to save current area signature (API varies between SDK versions)
                try:
                    # Preferred style if available
                    saved_id = area_memory.save()  # many bindings provide save() -> signature id
                    saved_area_ids.append(saved_id)
                    print(f"Saved area signature id: {saved_id}")
                except Exception as e:
                    try:
                        # Another possible form: area_memory.save_area() or area_memory.save_area(str)
                        saved_id = area_memory.save_area()  # try alternate name
                        saved_area_ids.append(saved_id)
                        print(f"Saved area signature id (alt): {saved_id}")
                    except Exception as e2:
                        print("Could not call Area Memory save API. Please adapt this block to your pyzed version.")
                        print(f"Area memory save errors: {e} | {e2}")

            # 3D retrieve & display
            try:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
                point_cloud_render = sl.Mat()
                point_cloud.copy_to(point_cloud_render)
            except Exception:
                point_cloud_render = None

            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # print object positions for debug and append cone detections to shared map
            try:
                # read current camera world translation once per frame
                cam_translation = (0.0, 0.0, 0.0)
                try:
                    trans = sl.Translation()
                    cam_w_pose.get_translation(trans)
                    cam_translation = tuple(trans.get())
                except Exception:
                    pass

                for obj in objects.object_list:
                    pos = None
                    try:
                        p = obj.position
                        if hasattr(p, 'get'):
                            pv = p.get()
                            pos = (float(pv[0]), float(pv[1]), float(pv[2]))
                        else:
                            pos = (float(p[0]), float(p[1]), float(p[2]))
                    except Exception:
                        pos = None

                    print(f"Object id={getattr(obj,'id','?')} label={getattr(obj,'label','?')} pos={pos}")

                    # Prepare to save an entry for this detection if it looks relevant
                    try:
                        lab = str(getattr(obj, 'label', '')).lower()
                    except Exception:
                        lab = ''
                    prob = getattr(obj, 'probability', None)

                    # decide whether to save: if label hints cone OR probability high enough
                    should_save = False
                    if pos is not None:
                        if any(k in lab for k in ('cone', 'orange', 'traffic')):
                            should_save = True
                        elif prob is None or (opt and getattr(opt, 'conf_thres', 0.5) and prob >= opt.conf_thres):
                            # save detections meeting confidence threshold even if label is unknown
                            should_save = True

                    if should_save:
                        x_mm, y_mm, z_mm = pos
                        # include camera rotation vector for later world-frame conversion
                        try:
                            rot = tuple(cam_w_pose.get_rotation_vector())
                        except Exception:
                            rot = (0.0, 0.0, 0.0)

                        entry = {
                            'label': str(getattr(obj, 'label', 'unknown')),
                            'probability': float(prob) if prob is not None else None,
                            'cam_x_mm': float(x_mm),
                            'cam_y_mm': float(y_mm),
                            'cam_z_mm': float(z_mm),
                            'cam_world_x_mm': float(cam_translation[0]),
                            'cam_world_y_mm': float(cam_translation[1]),
                            'cam_world_z_mm': float(cam_translation[2]),
                            'cam_world_rot_vec': [float(rot[0]), float(rot[1]), float(rot[2])],
                            # compute world coordinates for the cone (meters)
                            'world_x_m': None,
                            'world_y_m': None,
                            'world_z_m': None,
                            'ts': time.time()
                        }

                        # compute world coords for this detection BEFORE saving
                        try:
                            cam_point = sl.Translation()
                            cam_point.init_vector(float(x_mm) / 1000.0, float(y_mm) / 1000.0, float(z_mm) / 1000.0)
                            try:
                                try:
                                    tr = get_pose_transform(cam_w_pose)
                                    world_pt = tr * cam_point
                                except Exception as e:
                                    print(f"[DEBUG] Failed to obtain pose Transform: {e}")
                                    raise
                                wp = world_pt.get()
                                entry['world_x_m'] = float(wp[0])
                                entry['world_y_m'] = float(wp[1])
                                entry['world_z_m'] = float(wp[2])
                                print(f"[DEBUG] Computed world coords for obj id={getattr(obj,'id','?')}: {entry['world_x_m']:.3f}, {entry['world_y_m']:.3f}, {entry['world_z_m']:.3f}")
                            except Exception as e:
                                try:
                                    tr_repr = repr(tr)
                                except Exception:
                                    tr_repr = '<no-tr>'
                                print(f"[DEBUG] Failed to compute transform->world point: {e} | transform={tr_repr}")
                                entry['world_x_m'] = float(cam_translation[0]) / 1000.0
                                entry['world_y_m'] = float(cam_translation[1]) / 1000.0
                                entry['world_z_m'] = float(cam_translation[2]) / 1000.0
                        except Exception as e:
                            print(f"[DEBUG] Unexpected error computing world coords: {e}")

                        # load existing map
                        existing = []
                        try:
                            if os.path.exists(cone_map_path):
                                with open(cone_map_path, 'r') as mf:
                                    existing = json.load(mf)
                        except Exception as e:
                            print(f"[DEBUG] Could not read existing cone map: {e}")
                            existing = []

                        # simple dedupe by camera-frame proximity (100 mm)
                        def close(a, b, tol=100.0):
                            try:
                                return abs(a - b) <= tol
                            except Exception:
                                return False

                        already = False
                        for e in existing:
                            try:
                                if close(e.get('cam_x_mm', 1e9), entry['cam_x_mm']) and close(e.get('cam_y_mm', 1e9), entry['cam_y_mm']) and close(e.get('cam_z_mm', 1e9), entry['cam_z_mm']):
                                    already = True
                                    break
                            except Exception:
                                continue

                        if not already:
                            existing.append(entry)
                            # write atomically to avoid partial files
                            try:
                                tmp_path = cone_map_path + '.tmp'
                                with open(tmp_path, 'w') as mf:
                                    json.dump(existing, mf)
                                os.replace(tmp_path, cone_map_path)
                                print(f"[DEBUG] Wrote cone_map entry: {entry}")
                                # push landmark immediately to positional viewer if possible
                                try:
                                    if viewer_pt is not None and entry.get('world_x_m') is not None:
                                        # try non-blocking acquire to avoid deadlock
                                        acquired = viewer_pt.mutex.acquire(False)
                                        if acquired:
                                            try:
                                                cl = [1.0, 0.4, 0.0]
                                                wx = entry.get('world_x_m')
                                                wy = entry.get('world_y_m')
                                                wz = entry.get('world_z_m')
                                                if wx is None or wy is None or wz is None:
                                                    print("[DEBUG] Immediate push skipped: missing world coords")
                                                else:
                                                    try:
                                                        fx, fy, fz = float(wx), float(wy), float(wz)
                                                    except Exception:
                                                        print("[DEBUG] Immediate push skipped: invalid numeric world coords")
                                                    else:
                                                        if not (np.isfinite(fx) and np.isfinite(fy) and np.isfinite(fz)):
                                                            print("[DEBUG] Immediate push skipped: NaN/Inf in world coords")
                                                        else:
                                                            viewer_pt.landmarks.add_point_clr([fx, fy, fz], cl)
                                                            viewer_pt.landmarks.push_to_GPU()
                                                            print(f"[DEBUG] Immediately pushed landmark to positional viewer: {[fx, fy, fz]}")
                                            finally:
                                                try:
                                                    viewer_pt.mutex.release()
                                                except Exception:
                                                    pass
                                        else:
                                            # couldn't acquire viewer mutex; skip immediate push (map file contains entries)
                                            print("[DEBUG] Positional viewer busy; will display landmark on next refresh")
                                except Exception as e:
                                    print(f"[DEBUG] Failed immediate push to positional viewer: {e}")
                            except Exception as e:
                                print(f"[DEBUG] Failed to write cone_map.json: {e}")
                        else:
                            print("[DEBUG] Skipped duplicate cone entry")
                        # compute world coords for sending to positional viewer
                        try:
                            # ZED SDK: create translation for camera-frame point (convert mm->m)
                            cam_point = sl.Translation()
                            cam_point.init_vector(float(x_mm) / 1000.0, float(y_mm) / 1000.0, float(z_mm) / 1000.0)
                            try:
                                try:
                                    tr = get_pose_transform(cam_w_pose)
                                    world_pt = tr * cam_point
                                except Exception as e:
                                    print(f"[DEBUG] Failed to obtain pose Transform (second attempt): {e}")
                                    raise
                                wp = world_pt.get()
                                entry['world_x_m'] = float(wp[0])
                                entry['world_y_m'] = float(wp[1])
                                entry['world_z_m'] = float(wp[2])
                            except Exception:
                                # best-effort fallback: use camera translation as base (converted to meters)
                                entry['world_x_m'] = float(cam_translation[0]) / 1000.0
                                entry['world_y_m'] = float(cam_translation[1]) / 1000.0
                                entry['world_z_m'] = float(cam_translation[2]) / 1000.0
                        except Exception:
                            pass

            except Exception as e:
                print(f"Error reading object positions: {e}")

            # update GL viewer if available
            if viewer:
                # update point cloud and objects as before
                if point_cloud_render is not None:
                    try:
                        if not skip_pointcloud_update:
                            viewer.updateData(point_cloud_render, objects)
                    except Exception as e:
                        print(f"Error updating GL viewer: {e}")
                        skip_pointcloud_update = True
                        print("[WARN] Disabling point-cloud updates to GL viewer due to previous error")
                else:
                    print("Point cloud not available this frame; GL viewer not updated")

                # Prepare landmarks from detected objects: convert camera-frame positions to world-frame
                landmarks_world = []
                try:
                    # get camera world pose (sl.Pose) as rotation matrix + translation
                    # cam_w_pose already obtained earlier via zed.get_position
                    # We can use sl.Transform to compute world point: world = cam_pose * cam_point
                    for obj in objects.object_list:
                        try:
                            p = obj.position
                            if p is None:
                                continue
                            if hasattr(p, 'get'):
                                pv = p.get()
                                cam_pt = sl.Translation()
                                cam_pt.init_vector(float(pv[0]) / 1000.0, float(pv[1]) / 1000.0, float(pv[2]) / 1000.0)
                            else:
                                cam_pt = sl.Translation()
                                cam_pt.init_vector(float(p[0]) / 1000.0, float(p[1]) / 1000.0, float(p[2]) / 1000.0)

                            # Use Pose.pose_data to build a Transform and apply it
                            try:
                                try:
                                    tr = get_pose_transform(cam_w_pose)
                                    world_pt = tr * cam_pt
                                except Exception as e:
                                    print(f"[DEBUG] Failed to obtain pose Transform when preparing landmarks: {e}")
                                    raise
                                wp = world_pt.get()
                                wx, wy, wz = float(wp[0]), float(wp[1]), float(wp[2])
                                # skip invalid numeric results
                                if not (np.isfinite(wx) and np.isfinite(wy) and np.isfinite(wz)):
                                    continue
                                landmarks_world.append([wx, wy, wz])
                            except Exception:
                                # fallback to camera translation in meters
                                landmarks_world.append([float(cam_translation[0]) / 1000.0, float(cam_translation[1]) / 1000.0, float(cam_translation[2]) / 1000.0])
                        except Exception:
                            continue
                except Exception:
                    landmarks_world = []

                # send landmarks to viewer (viewer expects meters)
                try:
                    # flatten and validate points
                    fl_points = []
                    for p in landmarks_world:
                        if p is None or len(p) < 3:
                            continue
                        try:
                            x, y, z = float(p[0]), float(p[1]), float(p[2])
                        except Exception:
                            continue
                        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                            continue
                        fl_points.append([x, y, z])

                    if fl_points:
                        viewer.update_landmarks(fl_points)
                    else:
                        viewer.update_landmarks(None)
                except Exception as e:
                    print(f"Error updating landmarks in GL viewer: {e}")

                # Also update positional-tracking viewer with world landmarks and write world map
                try:
                    # Read existing map entries (world coords if present)
                    world_entries = []
                    if os.path.exists(cone_map_path):
                        try:
                            with open(cone_map_path, 'r') as mf:
                                world_entries = json.load(mf)
                        except Exception:
                            world_entries = []

                    # collect points from world_entries (prefer explicit world coords)
                    pt_list = []
                    for e in world_entries:
                        wx = e.get('world_x_m')
                        wy = e.get('world_y_m')
                        wz = e.get('world_z_m')
                        if wx is not None and wy is not None and wz is not None:
                            pt_list.append([wx, wy, wz])

                    if viewer_pt is not None:
                        # clear and update positional tracker's landmarks
                        try:
                            viewer_pt.mutex.acquire()
                            viewer_pt.landmarks.clear()
                            cl = [1.0, 0.4, 0.0]
                            for p in pt_list:
                                viewer_pt.landmarks.add_point_clr([float(p[0]), float(p[1]), float(p[2])], cl)
                            viewer_pt.landmarks.push_to_GPU()
                            print(f"[DEBUG] Updated positional viewer with {len(pt_list)} landmarks")
                        finally:
                            try:
                                viewer_pt.mutex.release()
                            except Exception:
                                pass
                except Exception as e:
                    print(f"Error updating positional-tracking viewer landmarks: {e}")

            # display 2D
            np.copyto(image_left_ocv, image_left.get_data())

            # Draw thread detections (in blue) for immediate feedback
            try:
                lock.acquire()
                local_dets = list(detections) if detections else []
                lock.release()
                if local_dets:
                    for det_obj in local_dets:
                        try:
                            bb2d = getattr(det_obj, 'bounding_box_2d', None)
                            if bb2d is None:
                                continue
                            pts = (np.array(bb2d).reshape((-1, 2)) * np.array(image_scale)).astype(np.int32)
                            cv2.polylines(image_left_ocv, [pts], True, (255, 0, 0, 255), 2)
                        except Exception:
                            continue
            except Exception:
                try:
                    lock.release()
                except Exception:
                    pass

            # Draw objects (green) using renderer if available
            try:
                cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            except Exception:
                try:
                    for obj in objects.object_list:
                        bb = np.array(getattr(obj, 'bounding_box', []))
                        if bb.size:
                            pts = (bb.reshape((-1, 2)) * np.array(image_scale)).astype(np.int32)
                            cv2.polylines(image_left_ocv, [pts], True, (0, 200, 0, 255), 2)
                except Exception:
                    pass

            cv2.imshow("Combined | 2D view", image_left_ocv)
            key = cv2.waitKey(10)
            if key == 27 or key == ord('q') or key == ord('Q'):
                exit_signal = True
        else:
            if opt.svo is not None and zed.get_svo_position() >= zed.get_svo_number_of_frames():
                print("SVO playback completed")
                break
            else:
                # temporary small pause
                sleep(0.01)

    if viewer:
        viewer.exit()
    exit_signal = True
    zed.close()


if __name__ == "__main__":
    try:
        with torch.no_grad():
            main()
    except Exception as e:
        print(f"Fatal error: {e}")
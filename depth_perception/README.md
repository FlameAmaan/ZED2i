# Object Detection

## Setting up

- Install YOLOv8 using pip:

```bash
pip install ultralytics
```

## Run the program

```bash
python detector.py --weights yolov8m.pt  # [--img_size 512 --conf_thres 0.1 --svo path/to/file.svo]
```

### Features

- The camera point cloud is displayed in a 3D OpenGL view  
- 3D bounding boxes around detected objects are drawn  
- Object classes and confidence thresholds can be changed

## Training your own model

This can use any model trained with YOLOv8, including custom-trained ones.

---

## Troubleshooting (OpenGL)

If you encounter the following error:

```
freeglut (detector.py): ERROR: Internal error <FBConfig with necessary capabilities not found> in function fgOpenWindow
```

Follow these steps:

1. Check the OpenGL vendor string:  
   ```bash
   glxinfo | grep OpenGL
   ```

2. If the vendor is **not NVIDIA**, check the current PRIME profile:  
   ```bash
   sudo prime-select query
   ```

3. If it returns `on-demand`, switch to NVIDIA and reboot:  
   ```bash
   sudo prime-select nvidia
   sudo reboot
   ```

4. After reboot, verify again:  
   ```bash
   glxinfo | grep OpenGL
   ```
   The vendor string should read:  
   ```
   OpenGL vendor string: NVIDIA Corporation
   ```

5. If it still doesn’t work, reinstall drivers and reboot:  
   ```bash
   sudo ubuntu-drivers install
   sudo reboot
   ```

---

**Source:** https://support.stereolabs.com/hc/en-us/articles/8422008229143-How-can-I-solve-ZED-SDK-OpenGL-issues-under-Ubuntu

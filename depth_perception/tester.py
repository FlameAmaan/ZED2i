# test_yolo_load.py
import traceback
import torch
import ultralytics

# allowlist ultralytics DetectionModel if you trust the file
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

try:
    ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
    print("Loaded checkpoint type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Keys:", list(ckpt.keys())[:20])
except Exception:
    traceback.print_exc()
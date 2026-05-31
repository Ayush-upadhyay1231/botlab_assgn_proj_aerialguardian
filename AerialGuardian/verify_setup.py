# verify_setup.py
# Run this to confirm your full environment is working

import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import cv2
print(f"OpenCV: {cv2.__version__}")

import numpy as np
print(f"NumPy: {np.__version__}")

from ultralytics import YOLO
print(f"Ultralytics: loaded successfully")

import supervision as sv
print(f"Supervision: {sv.__version__}")

from sahi import AutoDetectionModel
print(f"SAHI: loaded successfully")

import filterpy
print(f"FilterPy: {filterpy.__version__}")

import onnx
print(f"ONNX: {onnx.__version__}")

# Quick GPU inference test
if torch.cuda.is_available():
    print("\n--- Running quick GPU inference test ---")
    model = YOLO("yolov8n.pt")   # Downloads ~6 MB
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(dummy, device=0, verbose=False)
    print("YOLO inference on GPU:  PASSED")
else:
    print("\n  No CUDA GPU detected. Pipeline will run on CPU (very slow).")

print("\nEnvironment setup: COMPLETE ")
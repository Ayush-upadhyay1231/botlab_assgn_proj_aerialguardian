# src/detection/detector.py
"""
YOLO + SAHI detector module.
Provides a unified interface regardless of whether SAHI is active.

Design decision: We wrap SAHI into the same API as raw YOLO so
the tracker never needs to know which backend is running.
This is the Adapter pattern — a key design pattern to mention in interviews.
"""

from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    """
    Normalized detection output.
    Using a dataclass ensures every downstream component
    gets exactly the same data structure regardless of backend.
    """
    x1: float      # pixel coords, absolute
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0   # always 0 (person) in our pipeline


class DroneDetector:
    """
    Unified detector: supports raw YOLO or SAHI-enhanced inference.

    Why separate this into its own class?
    - Single Responsibility: detection logic lives here, not in main.py
    - Easy to swap models (YOLOv8 → YOLOv11) without touching tracking code
    - Testable in isolation
    """

    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.25,
        iou_threshold:  float = 0.45,
        device:         str   = "0",
        use_sahi:       bool  = True,
        slice_size:     int   = 512,
        slice_overlap:  float = 0.2,
        img_size:       int   = 1280,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.use_sahi       = use_sahi
        self.slice_size     = slice_size
        self.slice_overlap  = slice_overlap
        self.img_size       = img_size

        # Parse device
        if device.isdigit():
            self.device = f"cuda:{device}"
        elif device == "cpu":
            self.device = "cpu"
        else:
            self.device = device

        # Load model
        if use_sahi:
            self._load_sahi(weights_path)
        else:
            self._load_yolo(weights_path)

        print(f"Detector initialized | SAHI={use_sahi} | device={self.device}")

    def _load_yolo(self, weights_path: str):
        """Load raw YOLO model."""
        self.model = YOLO(weights_path)
        self.sahi_model = None

    def _load_sahi(self, weights_path: str):
        """
        Load YOLO wrapped in SAHI's AutoDetectionModel.
        SAHI needs the model in its own format to manage slicing.
        """
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type      = "yolov8",
            model_path      = weights_path,
            confidence_threshold = self.conf_threshold,
            device          = self.device,
        )
        # Also keep raw model for non-SAHI fallback
        self.model = YOLO(weights_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.
        Returns list of Detection objects (pixel coordinates).
        """
        if self.use_sahi:
            return self._detect_sahi(frame)
        else:
            return self._detect_yolo(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Standard YOLO inference."""
        results = self.model(
            frame,
            conf    = self.conf_threshold,
            iou     = self.iou_threshold,
            imgsz   = self.img_size,
            device  = self.device,
            classes = [0],          # person only
            verbose = False,
        )[0]

        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()    # (N, 4)
            confs = results.boxes.conf.cpu().numpy()    # (N,)
            for (x1, y1, x2, y2), conf in zip(boxes, confs):
                detections.append(Detection(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=float(conf)
                ))
        return detections

    def _detect_sahi(self, frame: np.ndarray) -> List[Detection]:
        """
        SAHI sliced inference.

        How it works:
        1. Divide frame into overlapping tiles of slice_size × slice_size
        2. Run YOLO on each tile independently
        3. Map results back to full-frame coordinates
        4. Merge overlapping boxes with WBF (Weighted Box Fusion)

        WBF vs NMS: WBF averages the coordinates of overlapping boxes
        instead of just picking one — produces tighter boxes at boundaries.
        """
        import cv2
        # SAHI expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = get_sliced_prediction(
            frame_rgb,
            self.sahi_model,
            slice_height        = self.slice_size,
            slice_width         = self.slice_size,
            overlap_height_ratio = self.slice_overlap,
            overlap_width_ratio  = self.slice_overlap,
            postprocess_type    = "GREEDYNMM",   # GREEDYNMM > NMS for crowds
            postprocess_match_threshold = self.iou_threshold,
            verbose             = 0,
        )

        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            detections.append(Detection(
                x1=bbox.minx, y1=bbox.miny,
                x2=bbox.maxx, y2=bbox.maxy,
                confidence=pred.score.value
            ))
        return detections

    def detections_to_numpy(self, detections: List[Detection]) -> np.ndarray:
        """
        Convert Detection list to (N, 5) numpy array: [x1, y1, x2, y2, conf]
        This is the format ByteTrack expects.
        """
        if not detections:
            return np.empty((0, 5), dtype=np.float32)
        return np.array([
            [d.x1, d.y1, d.x2, d.y2, d.confidence]
            for d in detections
        ], dtype=np.float32)
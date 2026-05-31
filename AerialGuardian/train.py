# train.py
"""
Train YOLOv8 on VisDrone person detection.
Includes drone-specific settings and callbacks for monitoring.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.callbacks.base import on_train_epoch_end


def get_args():
    p = argparse.ArgumentParser(description="Train Aerial Guardian detector")
    p.add_argument("--model",   default="yolov8n.pt",
                   help="Base YOLO model or path to checkpoint")
    p.add_argument("--data",    default="data/processed/dataset.yaml")
    p.add_argument("--epochs",  type=int, default=150)
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--imgsz",   type=int, default=1280)
    p.add_argument("--resume",  action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--device",  default="0",
                   help="GPU index (0, 1, ...) or 'cpu'")
    return p.parse_args()


def main():
    args = get_args()

    model = YOLO(args.model)

    # Custom training parameters — overrides yaml where both exist
    results = model.train(
        data      = args.data,
        epochs    = args.epochs,
        batch     = args.batch,
        imgsz     = args.imgsz,
        device    = args.device,
        resume    = args.resume,

        # Small object specific
        conf      = 0.001,          # Low threshold during eval
        iou       = 0.6,            # Tight NMS for packed crowds

        # Augmentation
        mosaic    = 1.0,
        mixup     = 0.15,
        copy_paste= 0.1,
        scale     = 0.75,
        flipud    = 0.0,

        # Output
        project   = "models/runs",
        name      = "visdrone_person",
        save_period = 10,
        plots     = True,           # Save training curves

        # Performance
        workers   = 4,              # DataLoader workers (reduce if RAM is low)
        amp       = True,           # Mixed precision (FP16) — critical for speed
        cache     = False,          # Set True if you have lots of RAM (>32GB)
    )

    print(f"\nTraining complete. Best weights: {results.save_dir}/weights/best.pt")

    # ── Post-training evaluation ──────────────────────────────────────
    print("\nRunning validation on best weights...")
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    metrics = best_model.val(
        data   = args.data,
        imgsz  = args.imgsz,
        device = args.device,
        conf   = 0.25,              # Inference threshold
        iou    = 0.45,
    )
    print(f"mAP50:   {metrics.box.map50:.4f}")
    print(f"mAP50-95:{metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
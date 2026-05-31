# src/utils/visualize_annotations.py
"""
Draws YOLO-format bounding boxes on sample images.
Run this to verify your conversion is correct before training.
"""

import cv2
import numpy as np
from pathlib import Path
import random

PROCESSED_ROOT = Path("data/processed")
NUM_SAMPLES    = 8


def draw_yolo_boxes(img, label_path):
    h, w = img.shape[:2]
    result = img.copy()

    if not label_path.exists():
        return result

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)

            # Convert from normalized to pixel coords
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Overlay annotation count
    n_boxes = sum(1 for l in open(label_path) if l.strip())
    cv2.putText(result, f"{n_boxes} persons",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return result


def main():
    img_dir = PROCESSED_ROOT / "images" / "train"
    lbl_dir = PROCESSED_ROOT / "labels" / "train"

    all_imgs = list(img_dir.glob("*.jpg"))
    samples  = random.sample(all_imgs, min(NUM_SAMPLES, len(all_imgs)))

    grid_imgs = []
    for img_path in samples:
        img = cv2.imread(str(img_path))
        lbl = lbl_dir / (img_path.stem + ".txt")
        annotated = draw_yolo_boxes(img, lbl)
        annotated = cv2.resize(annotated, (640, 360))
        grid_imgs.append(annotated)

    # Build a 2×4 grid
    rows = [np.hstack(grid_imgs[i:i+4]) for i in range(0, len(grid_imgs), 4)]
    grid = np.vstack(rows)

    out_path = "outputs/annotation_check.jpg"
    cv2.imwrite(out_path, grid)
    print(f"Verification grid saved to: {out_path}")
    cv2.imshow("Annotation Check (press any key)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
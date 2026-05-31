# src/utils/prepare_dataset.py
"""
Converts VisDrone MOT annotations to YOLO detection format.
Filters PERSON class only (categories 1 and 2).
Handles ignored regions, truncation, and heavy occlusion.
"""

import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


# ── CONFIG ─────────────────────────────────────────────────────────────────
VISDRONE_ROOT   = Path("data/raw/VisDrone")
OUTPUT_ROOT     = Path("data/processed")
PERSON_CLASSES  = {1, 2}        # pedestrian + people
IGNORE_CLASS    = 0             # ignored region
MIN_BBOX_AREA   = 4 * 4         # skip boxes smaller than 4x4 px
MAX_OCCLUSION   = 2             # 0=visible, 1=partial, 2=heavy → skip 3=unknown
TRAIN_RATIO     = 0.85
SEED            = 42
# ───────────────────────────────────────────────────────────────────────────


def visdrone_to_yolo(bbox_left, bbox_top, bbox_w, bbox_h, img_w, img_h):
    """
    Convert VisDrone (x1, y1, w, h) absolute coords
    to YOLO (cx, cy, w, h) normalized coords.

    Why normalize? YOLO requires values in [0,1] so it's
    resolution-independent — the same label works for any resized copy.
    """
    cx = (bbox_left + bbox_w / 2.0) / img_w
    cy = (bbox_top  + bbox_h / 2.0) / img_h
    w  = bbox_w / img_w
    h  = bbox_h / img_h
    # Clamp to [0, 1] — handles edge-clipped annotations
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))
    return cx, cy, w, h


def parse_annotation_line(line):
    """
    Parse one line from a VisDrone annotation file.
    Returns a dict or None if the line is malformed.
    """
    parts = line.strip().split(",")
    if len(parts) < 6:
        return None
    try:
        return {
            "x1":         int(parts[0]),
            "y1":         int(parts[1]),
            "w":          int(parts[2]),
            "h":          int(parts[3]),
            "score":      int(parts[4]),
            "category":   int(parts[5]),
            "truncation": int(parts[6]) if len(parts) > 6 else 0,
            "occlusion":  int(parts[7]) if len(parts) > 7 else 0,
        }
    except ValueError:
        return None


def should_skip(ann):
    """
    Returns True if we should skip this annotation.
    Filters out:
      - ignored regions (category 0)
      - non-person categories
      - zero-score (flagged as ignore by annotators)
      - boxes that are too small
      - unknown occlusion (3)
    """
    if ann["category"] == IGNORE_CLASS:
        return True
    if ann["category"] not in PERSON_CLASSES:
        return True
    if ann["score"] == 0:
        return True
    if ann["w"] * ann["h"] < MIN_BBOX_AREA:
        return True
    if ann["occlusion"] == 3:   # unknown — unreliable label
        return True
    return False


def convert_sequence(seq_dir, ann_file, out_img_dir, out_lbl_dir):
    """
    Convert one VisDrone sequence (folder of frames + one annotation file)
    into YOLO-format image+label pairs.

    Returns count of (frames_processed, annotations_kept, annotations_skipped)
    """
    if not ann_file.exists():
        return 0, 0, 0

    # Read all annotation lines; VisDrone MOT files are frame-indexed
    # Each line is: frame_index, target_id, x1, y1, w, h, score, cat, trunc, occ
    # NOTE: MOT format has a leading frame_index and target_id
    annotations_by_frame = {}
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            try:
                frame_idx  = int(parts[0])
                # target_id = int(parts[1])  # not needed for detection
                ann = {
                    "x1":         int(parts[2]),
                    "y1":         int(parts[3]),
                    "w":          int(parts[4]),
                    "h":          int(parts[5]),
                    "score":      int(parts[6]),
                    "category":   int(parts[7]),
                    "truncation": int(parts[8]) if len(parts) > 8 else 0,
                    "occlusion":  int(parts[9]) if len(parts) > 9 else 0,
                }
                if frame_idx not in annotations_by_frame:
                    annotations_by_frame[frame_idx] = []
                annotations_by_frame[frame_idx].append(ann)
            except (ValueError, IndexError):
                continue

    frames = sorted(seq_dir.glob("*.jpg"))
    kept_total, skipped_total, frames_processed = 0, 0, 0

    for frame_path in frames:
        # Frame index is the stem: "0000001" → 1
        frame_idx = int(frame_path.stem)
        anns = annotations_by_frame.get(frame_idx, [])

        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        yolo_lines = []
        for ann in anns:
            if should_skip(ann):
                skipped_total += 1
                continue

            cx, cy, w, h = visdrone_to_yolo(
                ann["x1"], ann["y1"], ann["w"], ann["h"], img_w, img_h
            )

            if w < 1e-4 or h < 1e-4:   # degenerate box after normalization
                skipped_total += 1
                continue

            # class 0 = person (we only have one class)
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            kept_total += 1

        # Copy image
        dst_img = out_img_dir / frame_path.name
        shutil.copy2(frame_path, dst_img)

        # Write label (empty file if no persons — YOLO needs this)
        dst_lbl = out_lbl_dir / (frame_path.stem + ".txt")
        dst_lbl.write_text("\n".join(yolo_lines))

        frames_processed += 1

    return frames_processed, kept_total, skipped_total


def build_train_val_split(all_sequences, train_ratio, seed):
    """
    Split sequences (not frames) into train/val.
    Splitting by sequence prevents data leakage —
    consecutive frames of the same video must stay together.
    """
    random.seed(seed)
    seqs = sorted(all_sequences)
    random.shuffle(seqs)
    split = int(len(seqs) * train_ratio)
    return seqs[:split], seqs[split:]


def write_dataset_yaml(output_root, class_names):
    """Write the dataset.yaml that Ultralytics YOLO reads."""
    yaml_content = f"""# VisDrone Person Detection Dataset
# Generated by prepare_dataset.py

path: {output_root.resolve().as_posix()}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}

# VisDrone-specific notes:
# - Images are 1920x1080 drone footage
# - Person objects are often 10-50px tall
# - Training with img_sz=1280 is strongly recommended
"""
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"dataset.yaml written to: {yaml_path}")


def main():
    seq_root = VISDRONE_ROOT / "sequences"
    ann_root = VISDRONE_ROOT / "annotations"

    if not seq_root.exists():
        raise FileNotFoundError(
            f"Sequences directory not found: {seq_root}\n"
            f"Make sure you extracted the VisDrone dataset correctly."
        )

    all_sequences = [d for d in seq_root.iterdir() if d.is_dir()]
    print(f"Found {len(all_sequences)} sequences")

    train_seqs, val_seqs = build_train_val_split(all_sequences, TRAIN_RATIO, SEED)
    print(f"Train sequences: {len(train_seqs)}")
    print(f"Val   sequences: {len(val_seqs)}")

    for split, seqs in [("train", train_seqs), ("val", val_seqs)]:
        out_img = OUTPUT_ROOT / "images" / split
        out_lbl = OUTPUT_ROOT / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        total_frames, total_kept, total_skipped = 0, 0, 0
        for seq_dir in tqdm(seqs, desc=f"Converting {split}"):
            ann_file = ann_root / f"{seq_dir.name}.txt"
            f, k, s = convert_sequence(seq_dir, ann_file, out_img, out_lbl)
            total_frames  += f
            total_kept    += k
            total_skipped += s

        print(f"\n[{split}] Frames: {total_frames} | "
              f"Annotations kept: {total_kept} | Skipped: {total_skipped}")

    write_dataset_yaml(OUTPUT_ROOT, ["person"])
    print("\nDataset preparation complete ✅")


if __name__ == "__main__":
    main()
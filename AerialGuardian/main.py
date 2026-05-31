import argparse
import time
import sys
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.detection.detector import DroneDetector
from src.tracking.bytetracker import AerialTracker
from src.compensation.motion_compensator import MotionCompensator
from src.utils.visualizer import TrackVisualizer


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",   required=True)
    p.add_argument("--output",  default="outputs/videos/result.mp4")
    p.add_argument("--config",  default="configs/pipeline.yaml")
    p.add_argument("--weights", default=None)
    p.add_argument("--device",  default=None)
    p.add_argument("--no-sahi", action="store_true")
    p.add_argument("--no-mc",   action="store_true")
    p.add_argument("--show",    action="store_true")
    p.add_argument("--conf",    type=float, default=None)
    return p.parse_args()


class ImageSequenceCapture:
    def __init__(self, folder_path, fps=25.0):
        folder = Path(folder_path)
        self.frames = sorted(folder.glob("*.jpg"))
        if not self.frames:
            self.frames = sorted(folder.glob("*.png"))
        if not self.frames:
            raise FileNotFoundError(
                f"No JPG/PNG images found in: {folder_path}\n"
                f"Contents: {list(folder.iterdir())[:10]}"
            )
        self.idx = 0
        self.fps = fps
        first = cv2.imread(str(self.frames[0]))
        self.h, self.w = first.shape[:2]
        print(f"Sequence: {len(self.frames)} frames | {self.w}x{self.h}")

    def isOpened(self):
        return self.idx < len(self.frames)

    def read(self):
        if self.idx >= len(self.frames):
            return False, None
        frame = cv2.imread(str(self.frames[self.idx]))
        self.idx += 1
        if frame is None:
            return False, None
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:  return self.w
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return self.h
        if prop == cv2.CAP_PROP_FPS:          return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:  return len(self.frames)
        return 0

    def release(self):
        pass


def open_video(video_path):
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Path does not exist: {video_path}\n"
            f"Current directory: {Path.cwd()}\n"
            f"Available sequences: {list(Path('data/raw/VisDrone/sequences').iterdir())[:5] if Path('data/raw/VisDrone/sequences').exists() else 'NOT FOUND'}"
        )
    if p.is_dir():
        print(f"Mode: IMAGE SEQUENCE")
        return ImageSequenceCapture(str(p))
    else:
        print(f"Mode: VIDEO FILE")
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        return cap


def setup_writer(cap, output_path, codec="mp4v"):
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h)), fps


def run_pipeline(args, cfg):
    det_cfg = cfg["detector"]
    trk_cfg = cfg["tracker"]
    mc_cfg  = cfg["motion_compensation"]
    out_cfg = cfg["output"]
    opt_cfg = cfg["optimization"]

    weights  = args.weights or det_cfg["weights"]
    device   = args.device  or det_cfg["device"]
    use_sahi = (not args.no_sahi) and det_cfg["use_sahi"]
    use_mc   = (not args.no_mc)   and mc_cfg["enabled"]
    conf     = args.conf    or det_cfg["conf_threshold"]

    print("=" * 55)
    print("  The Aerial Guardian")
    print("=" * 55)
    print(f"  Input:   {args.video}")
    print(f"  Weights: {weights}")
    print(f"  SAHI:    {use_sahi} | MC: {use_mc} | Device: {device}")
    print("=" * 55)

    detector = DroneDetector(
        weights_path   = weights,
        conf_threshold = conf,
        iou_threshold  = det_cfg["iou_threshold"],
        device         = device,
        use_sahi       = use_sahi,
        slice_size     = det_cfg["slice_size"],
        slice_overlap  = det_cfg["slice_overlap"],
        img_size       = det_cfg["img_size"],
    )

    tracker = AerialTracker(
        track_thresh = trk_cfg["track_thresh"],
        track_buffer = trk_cfg["track_buffer"],
        match_thresh = trk_cfg["match_thresh"],
        frame_rate   = trk_cfg["frame_rate"],
        min_hits     = trk_cfg["min_hits"],
        tail_length  = trk_cfg["tail_length"],
    )

    compensator = MotionCompensator() if use_mc else None
    visualizer  = TrackVisualizer()

    cap = open_video(args.video)
    writer, _ = setup_writer(cap, args.output)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_q = deque(maxlen=30)
    frame_idx = 0

    print(f"\nProcessing {total} frames...\n")

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        H = None
        if compensator:
            if frame_idx % mc_cfg["refresh_every_n_frames"] == 0:
                compensator.refresh_corners(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                )
            _, H = compensator.process(frame)

        detections = detector.detect(frame)
        dets_np    = detector.detections_to_numpy(detections)

        if H is not None and len(dets_np) > 0:
            dets_np = compensator.compensate_detections(dets_np, H)

        tracks = tracker.update(dets_np, frame.shape[:2])

        elapsed = time.perf_counter() - t0
        fps = 1.0 / max(elapsed, 1e-6)
        fps_q.append(fps)
        smooth_fps = sum(fps_q) / len(fps_q)

        annotated = visualizer.draw(
            frame.copy(), tracks,
            tracker.get_trajectory,
            tracker.get_color,
            fps=smooth_fps,
            n_detections=len(dets_np),
        )

        writer.write(annotated)

        if args.show:
            cv2.imshow("Aerial Guardian", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  [{frame_idx}/{total}] "
                  f"FPS:{smooth_fps:.1f} Tracks:{len(tracks)}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    avg = sum(fps_q) / len(fps_q) if fps_q else 0
    print(f"\nDone! {frame_idx} frames | Avg FPS: {avg:.1f}")
    print(f"Output: {args.output}")


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    run_pipeline(args, cfg)


if __name__ == "__main__":
    main()
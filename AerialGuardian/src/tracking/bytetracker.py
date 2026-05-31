# src/tracking/bytetracker.py
"""
ByteTrack wrapper for the Aerial Guardian pipeline.

We use the supervision library's ByteTrack implementation,
which is Windows-compatible, well-maintained, and integrates
cleanly with our Detection dataclass.

Why supervision's ByteTrack over the original?
- No C++ extension needed (works on Windows)
- Same algorithm, production-tested
- Direct numpy interface
"""

import numpy as np
import supervision as sv
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class Track:
    """
    Represents one active tracked person.
    """
    track_id:   int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    age: int    = 0   # frames since first seen
    hits: int   = 0   # total detection hits


class AerialTracker:
    """
    ByteTrack-based multi-object tracker tuned for drone footage.

    Key tuning decisions for drones:
    - track_thresh: lower than default (0.25 vs 0.5) — aerial persons
      are small and detectors are less confident
    - track_buffer: higher (60 frames) — drones pan/tilt, causing
      temporary disappearances. We need to hold tracks longer.
    - match_thresh: tighter IoU (0.8) — aerial objects are small,
      small IoU errors matter more
    """

    def __init__(
        self,
        track_thresh:    float = 0.25,
        track_buffer:    int   = 60,
        match_thresh:    float = 0.80,
        frame_rate:      int   = 25,
        min_hits:        int   = 3,       # frames before track is "confirmed"
        tail_length:     int   = 30,      # frames of trajectory to draw
    ):
        self.min_hits   = min_hits
        self.tail_length = tail_length

        # ByteTracker from supervision
        self.tracker = sv.ByteTrack(
        track_activation_threshold = track_thresh,
        lost_track_buffer          = track_buffer,
        minimum_matching_threshold = match_thresh,
        frame_rate                 = frame_rate,
)

        # Trajectory storage: {track_id: deque of (cx, cy) tuples}
        self.trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=tail_length)
        )

        # Hit counter: only show confirmed tracks
        self.hit_counts: Dict[int, int] = defaultdict(int)

        # Color palette for track IDs (deterministic by ID)
        self._colors = self._generate_color_palette(256)

    def update(self, detections_np: np.ndarray, frame_shape: Tuple[int, int]) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections_np: (N, 5) array [x1, y1, x2, y2, conf]
            frame_shape:   (height, width) of the frame

        Returns:
            List of confirmed Track objects
        """
        h, w = frame_shape

        if detections_np.shape[0] == 0:
            # No detections — advance Kalman filter, no new tracks
            sv_detections = sv.Detections.empty()
        else:
            sv_detections = sv.Detections(
                xyxy       = detections_np[:, :4],
                confidence = detections_np[:, 4],
                class_id   = np.zeros(len(detections_np), dtype=int),
            )

        tracked = self.tracker.update_with_detections(sv_detections)

        tracks = []
        for i in range(len(tracked)):
            tid  = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = tracked.xyxy[i]
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5

            # Update hit count
            self.hit_counts[tid] += 1

            # Update trajectory
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self.trajectories[tid].append((int(cx), int(cy)))

            # Only emit confirmed tracks
            if self.hit_counts[tid] >= self.min_hits:
                tracks.append(Track(
                    track_id   = tid,
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence = conf,
                    hits       = self.hit_counts[tid],
                ))

        # Clean up stale trajectories
        active_ids = {t.track_id for t in tracks}
        stale = [tid for tid in self.trajectories if tid not in active_ids]
        for tid in stale:
            if self.hit_counts.get(tid, 0) > 0:
                self.hit_counts[tid] -= 1   # decay — don't immediately delete
                if self.hit_counts[tid] == 0:
                    del self.trajectories[tid]
                    del self.hit_counts[tid]

        return tracks

    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """Return the trajectory points for a given track ID."""
        return list(self.trajectories[track_id])

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Return a deterministic BGR color for a track ID."""
        return self._colors[track_id % len(self._colors)]

    @staticmethod
    def _generate_color_palette(n: int) -> List[Tuple[int, int, int]]:
        """
        Generate N visually distinct BGR colors using the HSV color space.
        Using HSV gives better perceptual separation than random RGB.
        """
        import colorsys
        colors = []
        for i in range(n):
            hue = i / n
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            colors.append((int(b * 255), int(g * 255), int(r * 255)))
        return colors
# src/utils/visualizer.py
"""
Rendering utilities: bounding boxes, IDs, trajectory tails, FPS counter.
All rendering is done with OpenCV for maximum speed.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Deque
from collections import deque


class TrackVisualizer:

    def __init__(
        self,
        tail_alpha:  float = 0.7,     # Opacity of oldest tail point
        box_thickness: int = 2,
        text_scale: float = 0.5,
        text_thickness: int = 1,
    ):
        self.tail_alpha    = tail_alpha
        self.box_thickness = box_thickness
        self.text_scale    = text_scale
        self.text_thickness = text_thickness

    def draw(
        self,
        frame:       np.ndarray,
        tracks:      list,              # List[Track] from AerialTracker
        get_trajectory,                 # Callable: track_id → List[(cx, cy)]
        get_color,                      # Callable: track_id → (B, G, R)
        fps:         float = 0.0,
        n_detections: int  = 0,
    ) -> np.ndarray:
        """
        Draw all tracks and UI elements onto the frame.
        Modifies frame IN PLACE for performance (avoids copy).
        """
        # Draw trajectory tails FIRST (behind boxes)
        for track in tracks:
            color = get_color(track.track_id)
            tail  = get_trajectory(track.track_id)
            self._draw_tail(frame, tail, color)

        # Draw bounding boxes and IDs ON TOP
        for track in tracks:
            color = get_color(track.track_id)
            self._draw_box(frame, track, color)
            self._draw_label(frame, track, color)

        # HUD overlay
        self._draw_hud(frame, fps=fps, n_tracks=len(tracks), n_dets=n_detections)

        return frame

    def _draw_tail(
        self, frame: np.ndarray,
        tail: List[Tuple[int, int]],
        color: Tuple[int, int, int]
    ):
        """
        Draw trajectory tail with fading opacity.
        Oldest points are more transparent (fade effect).
        """
        if len(tail) < 2:
            return

        for i in range(1, len(tail)):
            # Fade: newest points are brightest
            alpha = self.tail_alpha * (i / len(tail))
            pt1   = tail[i - 1]
            pt2   = tail[i]

            # Draw as overlay segment with alpha blending
            overlay = frame.copy()
            thickness = max(1, int(2 * (i / len(tail))))  # thicker at tip
            cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_box(self, frame: np.ndarray, track, color: Tuple[int, int, int]):
        """Draw bounding box."""
        x1, y1 = int(track.x1), int(track.y1)
        x2, y2 = int(track.x2), int(track.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

    def _draw_label(self, frame: np.ndarray, track, color: Tuple[int, int, int]):
        """Draw ID label with background fill for readability."""
        x1, y1 = int(track.x1), int(track.y1)
        label   = f"ID:{track.track_id}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        # Background rectangle
        lx1, ly1 = x1, max(0, y1 - th - 4)
        lx2, ly2 = x1 + tw + 4, y1
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)

        # Text
        cv2.putText(
            frame, label, (lx1 + 2, ly2 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
            (0, 0, 0), self.text_thickness, cv2.LINE_AA
        )

    def _draw_hud(
        self, frame: np.ndarray,
        fps: float, n_tracks: int, n_dets: int
    ):
        """Heads-up display: FPS, track count, detection count."""
        h, w = frame.shape[:2]
        lines = [
            f"FPS: {fps:.1f}",
            f"Tracks: {n_tracks}",
            f"Dets: {n_dets}",
        ]
        y = 24
        for line in lines:
            cv2.putText(
                frame, line, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0), 1, cv2.LINE_AA
            )
            y += 24
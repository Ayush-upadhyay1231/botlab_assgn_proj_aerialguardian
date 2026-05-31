# src/compensation/motion_compensator.py
"""
Drone ego-motion compensation using sparse optical flow.

Algorithm:
1. Detect good feature points in the background (Shi-Tomasi corners)
2. Track those features frame-to-frame with Lucas-Kanade optical flow
3. Estimate the rigid motion (homography) between frames
4. Use the homography to stabilize detection coordinates
   before passing them to the tracker

Why homography (not just translation)?
Drone motion includes pan, tilt, rotation, and small scale changes.
A pure translation model would leave residual drift.
A homography handles all 8 degrees of freedom of projective motion.

Why optical flow over IMU?
VisDrone doesn't include IMU data. Optical flow works on any video.
In a real deployment with IMU access, you'd fuse both signals.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class MotionCompensator:

    def __init__(
        self,
        max_corners:     int   = 300,   # Feature points to track
        quality_level:   float = 0.01,  # Shi-Tomasi quality threshold
        min_distance:    int   = 10,    # Min pixels between corners
        block_size:      int   = 3,     # Corner detection window
        ransac_thresh:   float = 3.0,   # RANSAC inlier threshold (pixels)
        min_inliers:     int   = 20,    # Minimum inliers to trust homography
    ):
        self.max_corners   = max_corners
        self.quality_level = quality_level
        self.min_distance  = min_distance
        self.block_size    = block_size
        self.ransac_thresh = ransac_thresh
        self.min_inliers   = min_inliers

        # LK optical flow parameters
        self.lk_params = dict(
            winSize   = (21, 21),       # Tracking window — larger = more robust but slower
            maxLevel  = 3,              # Pyramid levels (handles fast motion)
            criteria  = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20, 0.01
            )
        )

        self.prev_gray:    Optional[np.ndarray] = None
        self.prev_corners: Optional[np.ndarray] = None
        self._homography:  Optional[np.ndarray] = None

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process a new frame. Returns:
          - stabilized frame (for visualization)
          - homography matrix H (for correcting detection coordinates)

        If no motion is detected (first frame or tracking fails),
        returns the original frame and None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            # First frame — initialize
            self.prev_gray    = gray
            self.prev_corners = self._detect_corners(gray)
            self._homography  = None
            return frame, None

        if self.prev_corners is None or len(self.prev_corners) < self.min_inliers:
            # Not enough corners to estimate motion — re-detect
            self.prev_gray    = gray
            self.prev_corners = self._detect_corners(gray)
            self._homography  = None
            return frame, None

        # Track corners from previous frame
        curr_corners, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            self.prev_corners, None,
            **self.lk_params
        )

        if curr_corners is None:
            self.prev_gray    = gray
            self.prev_corners = self._detect_corners(gray)
            self._homography  = None
            return frame, None

        # Keep only successfully tracked points
        good_prev = self.prev_corners[status == 1]
        good_curr = curr_corners[status == 1]

        if len(good_prev) < self.min_inliers:
            self.prev_gray    = gray
            self.prev_corners = self._detect_corners(gray)
            self._homography  = None
            return frame, None

        # Estimate homography with RANSAC
        # RANSAC robustly ignores moving objects (persons) as outliers —
        # they DON'T fit the background motion model
        H, inlier_mask = cv2.findHomography(
            good_prev, good_curr,
            cv2.RANSAC,
            self.ransac_thresh
        )

        if H is None:
            n_inliers = 0
        else:
            n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

        if n_inliers < self.min_inliers:
            # Motion estimation unreliable — skip compensation
            H = None

        self._homography = H

        # Update state
        self.prev_gray = gray
        # Re-detect corners periodically (every 15 frames is handled in main loop)
        self.prev_corners = curr_corners[status == 1].reshape(-1, 1, 2)

        # Apply warp for visualization (shows what a stabilized frame looks like)
        if H is not None:
            h, w = frame.shape[:2]
            stabilized = cv2.warpPerspective(frame, H, (w, h))
        else:
            stabilized = frame

        return stabilized, H

    def compensate_detections(
        self, detections_np: np.ndarray, H: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Apply the inverse homography to detection coordinates.

        We apply H_inv to move detections into the coordinate space
        of the PREVIOUS frame — where the Kalman filter's prediction lives.
        This way, a stationary person's box position stays consistent
        even as the drone moves.

        detections_np: (N, 5) array [x1, y1, x2, y2, conf]
        """
        if H is None or detections_np.shape[0] == 0:
            return detections_np

        compensated = detections_np.copy()

        # Invert homography: current → previous frame space
        H_inv = np.linalg.inv(H)

        # Transform corner points
        for i in range(len(detections_np)):
            x1, y1, x2, y2 = detections_np[i, :4]

            # Transform all 4 corners of the box
            corners = np.array([
                [x1, y1, 1.0],
                [x2, y1, 1.0],
                [x1, y2, 1.0],
                [x2, y2, 1.0],
            ]).T  # (3, 4)

            transformed = H_inv @ corners  # (3, 4)
            # Homogeneous → Euclidean
            transformed = transformed[:2] / transformed[2]  # (2, 4)

            compensated[i, 0] = transformed[0].min()   # x1
            compensated[i, 1] = transformed[1].min()   # y1
            compensated[i, 2] = transformed[0].max()   # x2
            compensated[i, 3] = transformed[1].max()   # y2

        return compensated

    def _detect_corners(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect Shi-Tomasi corner features."""
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners   = self.max_corners,
            qualityLevel = self.quality_level,
            minDistance  = self.min_distance,
            blockSize    = self.block_size,
        )
        return corners

    def refresh_corners(self, gray: np.ndarray):
        """Force re-detection of corners. Call every N frames to prevent drift."""
        self.prev_corners = self._detect_corners(gray)
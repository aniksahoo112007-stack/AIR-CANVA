"""Persistent transparent drawing surface and stroke smoothing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import (
    BASE_MAX_POINT_JUMP, DRAW_FAST_SPEED_THRESHOLD, ERASER_MULTIPLIER,
    FAST_MOVE_MAX_POINT_JUMP, GLOW_DOWNSCALE, MAX_DRAW_FRAME_GAP_SECONDS,
    MIN_DRAW_DISTANCE,
)
from .history_manager import CanvasSnapshot
from .stroke_smoother import StrokeSmoother


class DrawingCanvas:
    """Manage drawing state independently of camera and gesture logic."""

    def __init__(self, width: int, height: int) -> None:
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        self.mask = np.zeros((height, width), dtype=np.uint8)
        self.previous_point: tuple[int, int] | None = None
        self.smoother = StrokeSmoother()
        self._glow_cache = np.zeros_like(self.image)
        self._glow_dirty = True
        self.stroke_active = False
        self.active_hand_id: int | None = None
        self.previous_raw_point: tuple[int, int] | None = None
        self.previous_timestamp: float | None = None
        self.last_point_accepted = False
        self.last_rejection_reason = ""
        self.last_distance = 0.0

    def resize_if_needed(self, width: int, height: int) -> None:
        """Resize drawing layers while preserving existing artwork."""
        if self.image.shape[:2] == (height, width):
            return
        self.image = cv2.resize(self.image, (width, height), interpolation=cv2.INTER_LINEAR)
        self.mask = cv2.resize(self.mask, (width, height), interpolation=cv2.INTER_NEAREST)
        self._glow_cache = np.zeros_like(self.image)
        self._glow_dirty = True
        self.reset_stroke()

    def start_stroke(self, point: tuple[int, int], hand_id: int, timestamp: float) -> bool:
        """Validate and store the first anchor without drawing a dot."""
        valid, reason = self._validate_basic_point(point, timestamp)
        if not valid:
            self._reject(reason)
            return False
        self.reset_drawing_tracking()
        self.active_hand_id = hand_id
        self.previous_raw_point = point
        self.previous_point = self.smoother.update(point, timestamp)
        self.previous_timestamp = timestamp
        self.last_point_accepted = True
        self.last_rejection_reason = ""
        return True

    def append_stroke_point(self, point: tuple[int, int], hand_id: int, timestamp: float, color: tuple[int, int, int], size: int, erasing: bool) -> bool:
        """Append one validated point from the authoritative drawing hand."""
        if self.active_hand_id is None:
            return self.start_stroke(point, hand_id, timestamp) and False
        if self.active_hand_id != hand_id:
            self._reject("hand id changed")
            self.reset_drawing_tracking()
            return False
        if self.previous_point is None or self.previous_raw_point is None or self.previous_timestamp is None:
            return self.start_stroke(point, hand_id, timestamp) and False
        valid, reason = self._validate_basic_point(point, timestamp)
        if not valid:
            self._reject(reason)
            self.reset_drawing_tracking()
            return False
        dt = timestamp - self.previous_timestamp
        if dt > MAX_DRAW_FRAME_GAP_SECONDS:
            self._reject("frame gap")
            self.reset_drawing_tracking()
            return False
        distance = float(np.linalg.norm(np.subtract(point, self.previous_raw_point)))
        velocity = distance / max(dt, 1e-4)
        self.last_distance = distance
        allowed_jump = FAST_MOVE_MAX_POINT_JUMP if velocity >= DRAW_FAST_SPEED_THRESHOLD else BASE_MAX_POINT_JUMP
        if distance > allowed_jump:
            self._reject(f"jump {distance:.1f}px > {allowed_jump}px")
            self.reset_drawing_tracking()
            return False
        if distance < MIN_DRAW_DISTANCE:
            self.last_point_accepted = False
            self.last_rejection_reason = "below minimum distance"
            return False
        current = self.smoother.update(point, timestamp)
        smoothed_distance = float(np.linalg.norm(np.subtract(current, self.previous_point)))
        if smoothed_distance < MIN_DRAW_DISTANCE:
            self.previous_raw_point = point
            self.previous_timestamp = timestamp
            self.last_point_accepted = False
            self.last_rejection_reason = "smoothed movement below minimum"
            return False
        thickness = size * ERASER_MULTIPLIER if erasing else size
        ink = (0, 0, 0) if erasing else color
        mask_value = 0 if erasing else 255
        last = self.previous_point
        for interpolated in self.smoother.interpolate_points(self.previous_point, current):
            cv2.line(self.image, last, interpolated, ink, thickness, cv2.LINE_AA)
            cv2.line(self.mask, last, interpolated, mask_value, thickness, cv2.LINE_AA)
            last = interpolated
        radius = max(1, thickness // 2)
        cv2.circle(self.image, current, radius, ink, -1, cv2.LINE_AA)
        cv2.circle(self.mask, current, radius, mask_value, -1, cv2.LINE_AA)
        self.previous_point = current
        self.previous_raw_point = point
        self.previous_timestamp = timestamp
        self._glow_dirty = True
        self.stroke_active = True
        self.last_point_accepted = True
        self.last_rejection_reason = ""
        return True

    def end_stroke(self, _reason: str = "ended") -> None:
        self.reset_drawing_tracking()

    def reset_drawing_tracking(self) -> None:
        self.previous_point = None
        self.previous_raw_point = None
        self.previous_timestamp = None
        self.active_hand_id = None
        self.smoother.reset()
        self.stroke_active = False

    def draw_to(self, point: tuple[int, int], color: tuple[int, int, int], size: int, erasing: bool, timestamp: float | None = None) -> None:
        """Compatibility wrapper for tests and non-controller callers."""
        import time
        now = time.monotonic() if timestamp is None else timestamp
        if self.active_hand_id is None:
            self.start_stroke(point, 0, now)
        else:
            self.append_stroke_point(point, 0, now, color, size, erasing)

    def reset_stroke(self) -> None:
        """End a stroke so the next stroke cannot connect to it."""
        self.reset_drawing_tracking()

    def _validate_basic_point(self, point: tuple[int, int], timestamp: float) -> tuple[bool, str]:
        coordinates = np.asarray(point, dtype=np.float64)
        if not np.isfinite(coordinates).all():
            return False, "non-finite coordinate"
        if timestamp <= 0 or not np.isfinite(timestamp):
            return False, "invalid timestamp"
        height, width = self.image.shape[:2]
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            return False, "outside drawable region"
        return True, ""

    def _reject(self, reason: str) -> None:
        self.last_point_accepted = False
        self.last_rejection_reason = reason
        self.smoother.rejected_outliers += 1

    def clear(self) -> None:
        """Erase all artwork."""
        self.image.fill(0)
        self.mask.fill(0)
        self._glow_dirty = True
        self.reset_stroke()

    def restore(self, snapshot: CanvasSnapshot) -> None:
        """Restore canvas layers from undo/redo history."""
        self.image = snapshot.image.copy()
        self.mask = snapshot.mask.copy()
        self._glow_dirty = True
        self.reset_stroke()

    def overlay(self, frame: np.ndarray, glow: bool = False) -> np.ndarray:
        """Composite persistent strokes and an optional cached glow."""
        result = frame.copy()
        if glow and np.any(self.mask):
            if self._glow_dirty:
                self._rebuild_glow()
            cv2.addWeighted(result, 1.0, self._glow_cache, 0.72, 0, result)
        visible = self.mask > 0
        result[visible] = self.image[visible]
        return result

    def _rebuild_glow(self) -> None:
        scale = max(0.25, min(1.0, GLOW_DOWNSCALE))
        small_size = (max(1, int(self.image.shape[1] * scale)), max(1, int(self.image.shape[0] * scale)))
        colored = cv2.bitwise_and(self.image, self.image, mask=self.mask)
        small = cv2.resize(colored, small_size, interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=5.0)
        self._glow_cache = cv2.resize(blurred, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_LINEAR)
        self._glow_dirty = False

    def save(self, output_dir: Path, background: tuple[int, int, int] | None = None) -> Path:
        """Save transparent artwork or flatten it onto a board background."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Could not create output directory '{output_dir}': {exc}") from exc
        filename = datetime.now().strftime("air_canvas_%Y%m%d_%H%M%S_%f.png")
        path = output_dir / filename
        if background is None:
            output = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
            output[:, :, 3] = self.mask
        else:
            output = self.overlay(np.full_like(self.image, background), glow=False)
        if not cv2.imwrite(str(path), output):
            raise OSError(f"OpenCV could not write drawing to '{path}'.")
        return path

    def save_composite(self, frame: np.ndarray, output_dir: Path) -> Path:
        """Save camera plus drawing as a timestamped JPEG-quality PNG."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Could not create output directory '{output_dir}': {exc}") from exc
        path = output_dir / datetime.now().strftime("air_canvas_camera_%Y%m%d_%H%M%S_%f.png")
        composed = self.overlay(frame, glow=False)
        if not cv2.imwrite(str(path), composed):
            raise OSError(f"OpenCV could not write drawing to '{path}'.")
        return path

"""Single-stage adaptive exponential smoothing for canvas drawing."""

from __future__ import annotations

import math

import numpy as np

from .config import DRAW_FAST_SPEED_THRESHOLD, DRAW_SMOOTHING_FAST, DRAW_SMOOTHING_SLOW, INTERPOLATION_STEP_PIXELS


class StrokeSmoother:
    """A low-lag adaptive EMA; validation belongs to the drawing pipeline."""

    def __init__(self) -> None:
        self.previous_raw: np.ndarray | None = None
        self.filtered: np.ndarray | None = None
        self.timestamp: float | None = None
        self.velocity = 0.0
        self.rejected_outliers = 0

    def update(self, raw_point: tuple[int, int], timestamp: float) -> tuple[int, int]:
        raw = np.asarray(raw_point, dtype=np.float32)
        if self.filtered is None or self.previous_raw is None or self.timestamp is None:
            self.previous_raw = self.filtered = raw.copy()
            self.timestamp = timestamp
            self.velocity = 0.0
            return tuple(np.rint(raw).astype(int))
        dt = max(1e-4, timestamp - self.timestamp)
        self.velocity = float(np.linalg.norm(raw - self.previous_raw) / dt)
        ratio = min(1.0, self.velocity / DRAW_FAST_SPEED_THRESHOLD)
        alpha = DRAW_SMOOTHING_SLOW + (DRAW_SMOOTHING_FAST - DRAW_SMOOTHING_SLOW) * ratio
        self.filtered += alpha * (raw - self.filtered)
        self.previous_raw = raw.copy()
        self.timestamp = timestamp
        return tuple(np.rint(self.filtered).astype(int))

    @staticmethod
    def interpolate_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        distance = math.dist(start, end)
        count = max(1, int(math.ceil(distance / INTERPOLATION_STEP_PIXELS)))
        return [
            (round(start[0] + (end[0] - start[0]) * index / count), round(start[1] + (end[1] - start[1]) * index / count))
            for index in range(1, count + 1)
        ]

    def reject_outlier(self, raw: np.ndarray, _dt: float) -> bool:
        """Compatibility helper; authoritative rejection is canvas-aware."""
        return not np.isfinite(raw).all()

    def reset(self) -> None:
        self.previous_raw = self.filtered = None
        self.timestamp = None
        self.velocity = 0.0

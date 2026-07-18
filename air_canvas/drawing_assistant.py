"""Bounded adaptive tremor reduction for native drawing points."""

from __future__ import annotations

from collections import deque
import math

import numpy as np

from .config import ASSIST_HIGH_WINDOW, ASSIST_LOW_WINDOW, ASSIST_MEDIUM_WINDOW


class DrawingAssistant:
    levels = ("off", "low", "medium", "high")
    windows = {"off": 1, "low": ASSIST_LOW_WINDOW, "medium": ASSIST_MEDIUM_WINDOW, "high": ASSIST_HIGH_WINDOW}

    def __init__(self, level: str = "medium") -> None:
        self.level = "medium"
        self._points: deque[tuple[np.ndarray, float]] = deque(maxlen=ASSIST_HIGH_WINDOW)
        self.raw_points: list[tuple[int, int]] = []
        self.stabilized_points: list[tuple[int, int]] = []
        self.set_assistance_level(level)

    def set_assistance_level(self, level: str) -> None:
        normalized = level.lower()
        if normalized not in self.levels:
            raise ValueError(f"Unknown drawing assistance level: {level}")
        self.level = normalized

    def cycle_level(self) -> str:
        self.set_assistance_level(self.levels[(self.levels.index(self.level) + 1) % len(self.levels)])
        self.reset()
        return self.level

    def add_point(self, point: tuple[int, int], timestamp: float) -> tuple[int, int]:
        value = np.asarray(point, dtype=np.float64)
        self.raw_points.append(point)
        self._points.append((value, timestamp))
        if self.level == "off" or len(self._points) < 2:
            result = point
        else:
            samples = list(self._points)[-self.windows[self.level]:]
            coordinates = np.stack([sample[0] for sample in samples])
            weights = np.linspace(0.45, 1.0, len(samples), dtype=np.float64)
            average = np.average(coordinates, axis=0, weights=weights)
            previous, previous_time = samples[-2]
            speed = float(np.linalg.norm(value - previous)) / max(timestamp - previous_time, 1e-4)
            deliberate = min(1.0, speed / 850.0)
            raw_weight = 0.18 + deliberate * 0.72
            if len(samples) >= 3:
                first = samples[-2][0] - samples[-3][0]
                second = value - samples[-2][0]
                denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-6)
                angle = math.degrees(math.acos(float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))))
                if angle > 55.0:
                    raw_weight = max(raw_weight, 0.82)
            stabilized = average * (1.0 - raw_weight) + value * raw_weight
            result = tuple(np.rint(stabilized).astype(int))
        self.stabilized_points.append(result)
        return result

    def get_stabilized_point(self) -> tuple[int, int] | None:
        return self.stabilized_points[-1] if self.stabilized_points else None

    def finish_stroke(self) -> list[tuple[int, int]]:
        points = list(self.stabilized_points)
        self.reset()
        return points

    def reset(self) -> None:
        self._points.clear()
        self.raw_points.clear()
        self.stabilized_points.clear()

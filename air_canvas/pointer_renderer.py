"""Bounded, fading laser pointer renderer."""

from __future__ import annotations

from collections import deque
import time

import cv2
import numpy as np

from .config import LASER_COLOR, LASER_GLOW_RADIUS, LASER_RADIUS, LASER_TRAIL_LENGTH, LASER_TRAIL_LIFETIME_SECONDS


class PointerRenderer:
    def __init__(self) -> None:
        self.points: deque[tuple[tuple[int, int], float]] = deque(maxlen=LASER_TRAIL_LENGTH)
        self.visible = False

    def update(self, point: tuple[int, int] | None, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if point is None: self.visible = False
        else: self.visible = True; self.points.append((point, now))
        while self.points and now-self.points[0][1] > LASER_TRAIL_LIFETIME_SECONDS: self.points.popleft()

    def clear(self) -> None: self.points.clear(); self.visible = False

    def render(self, layer: np.ndarray, origin: tuple[int, int], now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if not self.visible or not self.points: return
        ox, oy = origin
        for point, stamped in self.points:
            alpha = max(0.0, 1.0-(now-stamped)/LASER_TRAIL_LIFETIME_SECONDS)
            x, y = point[0]-ox, point[1]-oy
            cv2.circle(layer, (x,y), max(2, round(LASER_RADIUS*alpha)), (*LASER_COLOR, round(210*alpha)), -1, cv2.LINE_AA)
        x, y = self.points[-1][0][0]-ox, self.points[-1][0][1]-oy
        cv2.circle(layer, (x,y), LASER_GLOW_RADIUS, (*LASER_COLOR, 48), 2, cv2.LINE_AA)
        cv2.circle(layer, (x,y), LASER_RADIUS, (255,255,255,245), -1, cv2.LINE_AA)

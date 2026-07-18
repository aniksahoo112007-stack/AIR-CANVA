"""Time-based UI notifications and lightweight ripple animations."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import animation_speed


@dataclass
class AnimatedNotification:
    text: str
    started_at: float
    duration: float
    color: tuple[int, int, int]


@dataclass
class Ripple:
    point: tuple[int, int]
    started_at: float
    color: tuple[int, int, int]


class AnimationManager:
    """Render frame-rate-independent fades and click ripples."""

    def __init__(self) -> None:
        self.notification: AnimatedNotification | None = None
        self._queue: list[AnimatedNotification] = []
        self.ripples: list[Ripple] = []

    def notify(self, text: str, now: float, duration: float = 1.6, color: tuple[int, int, int] = (80, 255, 190), queue: bool = False) -> None:
        notice = AnimatedNotification(text, now, duration / max(animation_speed, 0.1), color)
        if queue and self.notification is not None:
            self._queue.append(notice)
        else:
            self.notification = notice

    def ripple(self, point: tuple[int, int], now: float, color: tuple[int, int, int]) -> None:
        self.ripples.append(Ripple(point, now, color))
        self.ripples = self.ripples[-5:]

    def draw(self, frame: np.ndarray, now: float) -> None:
        alive: list[Ripple] = []
        for ripple in self.ripples:
            age = (now - ripple.started_at) * animation_speed
            if age >= 0.65:
                continue
            alive.append(ripple)
            progress = age / 0.65
            radius = int(10 + progress * 48)
            alpha = 1.0 - progress
            layer = frame.copy()
            cv2.circle(layer, ripple.point, radius, ripple.color, max(1, int(4 * alpha)), cv2.LINE_AA)
            cv2.addWeighted(layer, alpha * 0.75, frame, 1.0 - alpha * 0.75, 0, frame)
        self.ripples = alive

        notice = self.notification
        if notice is None:
            return
        age = now - notice.started_at
        if age >= notice.duration:
            if self._queue:
                following = self._queue.pop(0)
                following.started_at = now
                self.notification = following
            else:
                self.notification = None
            return
        fade = min(1.0, age / 0.18, (notice.duration - age) / 0.32)
        scale = 0.78 + 0.06 * min(age / 0.22, 1.0)
        size = cv2.getTextSize(notice.text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0]
        x = (frame.shape[1] - size[0]) // 2
        y = int(frame.shape[0] * 0.24)
        layer = frame.copy()
        cv2.rectangle(layer, (x - 24, y - 37), (x + size[0] + 24, y + 17), (8, 17, 28), -1)
        cv2.rectangle(layer, (x - 24, y - 37), (x + size[0] + 24, y + 17), notice.color, 1, cv2.LINE_AA)
        cv2.putText(layer, notice.text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, notice.color, 2, cv2.LINE_AA)
        cv2.addWeighted(layer, fade * 0.92, frame, 1.0 - fade * 0.92, 0, frame)

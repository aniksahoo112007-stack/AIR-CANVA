"""Compact header controls for display-only camera zoom."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import CAMERA_ZOOM_MAX, CAMERA_ZOOM_MIN
from .toolbar import rounded_rectangle
from .ui_layout import UILayout


@dataclass(frozen=True)
class ControlRect:
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class CameraViewControls:
    """Draw and hit-test zoom controls from the same rectangles."""

    def __init__(self) -> None:
        empty = ControlRect(0, 0, 0, 0)
        self.minus_rect = self.plus_rect = self.reset_rect = empty
        self.hovered: str | None = None
        self.pressed: str | None = None

    def update_layout(self, layout: UILayout) -> None:
        s = layout.scaled
        height, button = s(34), s(34)
        total_width = s(286)
        camera_control_x = max(s(330), layout.window_width - s(590))
        x = max(s(270), camera_control_x - total_width - s(18))
        y = s(11)
        self.minus_rect = ControlRect(x, y, x + button, y + height)
        self.plus_rect = ControlRect(x + s(142), y, x + s(142) + button, y + height)
        self.reset_rect = ControlRect(x + s(184), y, x + total_width, y + height)

    def update_mouse(self, event: int, x: int, y: int) -> str | None:
        controls = (("out", self.minus_rect), ("in", self.plus_rect), ("reset", self.reset_rect))
        self.hovered = next((name for name, rect in controls if rect.contains(x, y)), None)
        if event == cv2.EVENT_LBUTTONDOWN and self.hovered is not None:
            self.pressed = self.hovered
            return self.hovered
        if event == cv2.EVENT_LBUTTONUP:
            self.pressed = None
        return None

    def render(self, frame: np.ndarray, layout: UILayout, zoom: float) -> None:
        self.update_layout(layout)
        s, sf = layout.scaled, layout.scaled_font
        disabled_out = zoom <= CAMERA_ZOOM_MIN + 0.001
        disabled_in = zoom >= CAMERA_ZOOM_MAX - 0.001
        for name, rect, disabled in (("out", self.minus_rect, disabled_out), ("in", self.plus_rect, disabled_in), ("reset", self.reset_rect, False)):
            fill = (18, 26, 38) if disabled else (34, 54, 72) if self.hovered == name else (22, 38, 54)
            if self.pressed == name and not disabled:
                fill = (42, 78, 94)
            rounded_rectangle(frame, rect.bounds, s(8), fill, -1)
            rounded_rectangle(frame, rect.bounds, s(8), (55, 90, 112) if disabled else (90, 205, 235), 1)
        cv2.putText(frame, "-", (self.minus_rect.x1 + s(12), self.minus_rect.y1 + s(23)), cv2.FONT_HERSHEY_DUPLEX, sf(0.48), (110, 130, 145) if disabled_out else (175, 235, 250), 1, cv2.LINE_AA)
        cv2.putText(frame, "+", (self.plus_rect.x1 + s(10), self.plus_rect.y1 + s(23)), cv2.FONT_HERSHEY_DUPLEX, sf(0.46), (110, 130, 145) if disabled_in else (175, 235, 250), 1, cv2.LINE_AA)
        label = f"ZOOM {zoom:.2f}x"
        cv2.putText(frame, label, (self.minus_rect.x2 + s(10), self.minus_rect.y1 + s(22)), cv2.FONT_HERSHEY_DUPLEX, sf(0.36), (205, 225, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, "RESET", (self.reset_rect.x1 + s(13), self.reset_rect.y1 + s(22)), cv2.FONT_HERSHEY_DUPLEX, sf(0.31), (170, 225, 240), 1, cv2.LINE_AA)

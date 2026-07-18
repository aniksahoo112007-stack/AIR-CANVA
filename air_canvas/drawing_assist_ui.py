"""Window-space controls and confirmation overlay for drawing assistance."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .toolbar import rounded_rectangle
from .ui_layout import UILayout


@dataclass(frozen=True)
class AssistRect:
    x1: int; y1: int; x2: int; y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class DrawingAssistUI:
    actions = ("assist", "shape", "clean", "desktop")

    def __init__(self) -> None:
        self.rects: dict[str, AssistRect] = {}
        self.accept_rect = AssistRect(0, 0, 0, 0)
        self.cancel_rect = AssistRect(0, 0, 0, 0)
        self.hovered: str | None = None

    def update_layout(self, layout: UILayout) -> None:
        s = layout.scaled
        x, y, width, height, gap = s(16), layout.toolbar_rect.y2 + s(12), s(92), s(32), s(7)
        self.rects = {action: AssistRect(x + i * (width + gap), y, x + i * (width + gap) + width, y + height)
                      for i, action in enumerate(self.actions)}

    def update_mouse(self, event: int, x: int, y: int) -> str | None:
        candidates = list(self.rects.items()) + [("accept", self.accept_rect), ("cancel", self.cancel_rect)]
        self.hovered = next((name for name, rect in candidates if rect.contains(x, y)), None)
        return self.hovered if event == cv2.EVENT_LBUTTONDOWN else None

    def render_controls(self, frame: np.ndarray, layout: UILayout, assist: str, auto_shape: bool, cleanup: str) -> None:
        self.update_layout(layout)
        self.accept_rect = AssistRect(0, 0, 0, 0)
        self.cancel_rect = AssistRect(0, 0, 0, 0)
        s, sf = layout.scaled, layout.scaled_font
        labels = {"assist": f"ASSIST {assist[:3].upper()}", "shape": f"SHAPE {'ON' if auto_shape else 'OFF'}", "clean": f"CLEAN {cleanup[:3].upper()}", "desktop": "DESKTOP"}
        for action, rect in self.rects.items():
            active = action == "assist" and assist != "off" or action == "shape" and auto_shape
            fill = (34, 70, 72) if active else (22, 38, 54)
            if self.hovered == action:
                fill = (42, 78, 94)
            rounded_rectangle(frame, rect.bounds, s(8), fill, -1)
            rounded_rectangle(frame, rect.bounds, s(8), (80, 210, 220) if active else (70, 115, 140), 1)
            text = labels[action]
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, sf(0.29), 1)[0]
            cv2.putText(frame, text, (rect.x1 + (rect.x2-rect.x1-size[0])//2, rect.y1+s(21)), cv2.FONT_HERSHEY_DUPLEX, sf(0.29), (205, 235, 242), 1, cv2.LINE_AA)

    def render_confirmation(self, frame: np.ndarray, layout: UILayout, title: str, subtitle: str = "ENTER / PINCH TO ACCEPT") -> None:
        s, sf = layout.scaled, layout.scaled_font
        width, height = s(460), s(94)
        x, y = (layout.window_width - width) // 2, layout.status_bar_rect.y - height - s(18)
        bounds = (x, y, x + width, y + height)
        overlay = frame.copy(); rounded_rectangle(overlay, bounds, s(14), (8, 17, 29), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        rounded_rectangle(frame, bounds, s(14), (80, 220, 255), 1)
        cv2.putText(frame, title, (x+s(16), y+s(28)), cv2.FONT_HERSHEY_DUPLEX, sf(0.43), (110, 235, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (x+s(16), y+s(51)), cv2.FONT_HERSHEY_SIMPLEX, sf(0.29), (145, 170, 195), 1, cv2.LINE_AA)
        self.accept_rect = AssistRect(x + width - s(174), y + s(58), x + width - s(92), y + s(84))
        self.cancel_rect = AssistRect(x + width - s(86), y + s(58), x + width - s(10), y + s(84))
        for name, rect, color in (("accept", self.accept_rect, (35, 92, 70)), ("cancel", self.cancel_rect, (60, 55, 70))):
            rounded_rectangle(frame, rect.bounds, s(7), color if self.hovered != name else tuple(min(255, c+25) for c in color), -1)
            cv2.putText(frame, name.upper(), (rect.x1+s(10), rect.y1+s(18)), cv2.FONT_HERSHEY_DUPLEX, sf(0.28), (215, 238, 245), 1, cv2.LINE_AA)

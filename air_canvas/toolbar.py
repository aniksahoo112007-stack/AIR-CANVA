"""Premium neon toolbar with shared render and hit-test button models."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .config import COLORS, DEBUG_TOOLBAR_INTERACTION, TOOLBAR_SAFE_INSET
from .ui_layout import UILayout


@dataclass
class ToolbarButton:
    """Single source of truth for a rendered and interactive toolbar action."""

    label: str
    action: str
    rectangle: tuple[int, int, int, int]
    enabled: bool = True
    hovered: bool = False

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rectangle
        return x1 <= x <= x2 and y1 <= y <= y2


def rounded_rectangle(image: np.ndarray, bounds: tuple[int, int, int, int], radius: int, color: tuple[int, int, int], thickness: int = -1) -> None:
    """Draw a rounded rectangle using inexpensive OpenCV primitives."""
    x1, y1, x2, y2 = bounds
    radius = min(radius, max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for center in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius), (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
    else:
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        for center, start, end in (((x1 + radius, y1 + radius), 180, 270), ((x2 - radius, y1 + radius), 270, 360), ((x2 - radius, y2 - radius), 0, 90), ((x1 + radius, y2 - radius), 90, 180)):
            cv2.ellipse(image, center, (radius, radius), 0, start, end, color, thickness, cv2.LINE_AA)


class Toolbar:
    """Responsive translucent toolbar using identical draw/hit rectangles."""

    actions = ("red", "blue", "green", "yellow", "white", "eraser", "undo", "redo", "board", "clear", "save")

    def __init__(self, frame_width: int) -> None:
        self._width = 0
        self.buttons: list[ToolbarButton] = []
        self._panel_cache: np.ndarray | None = None
        self._panel_mask: np.ndarray | None = None
        self._layout = UILayout.create(frame_width, 720, frame_width, 720)
        self.update_layout(self._layout)

    @property
    def tools(self) -> tuple[str, ...]:
        return self.actions

    @property
    def y_range(self) -> tuple[int, int]:
        if not self.buttons:
            return 0, 0
        return self.buttons[0].rectangle[1], self.buttons[0].rectangle[3]

    def update_width(self, frame_width: int) -> None:
        self.update_layout(UILayout.create(frame_width, 720, frame_width, 720))

    def update_layout(self, layout: UILayout) -> None:
        signature = (layout.window_width, layout.window_height, round(layout.ui_scale, 3))
        if getattr(self, "_layout_signature", None) == signature:
            return
        self._layout_signature = signature
        self._layout = layout
        self._width = layout.window_width
        gap = layout.scaled(7)
        group_extra = layout.scaled(10)
        padding = layout.scaled(10)
        available = layout.toolbar_rect.width - padding * 2 - gap * (len(self.actions) - 1) - group_extra * 2
        button_width = max(layout.scaled(52), min(layout.scaled(105), available // len(self.actions)))
        total_width = button_width * len(self.actions) + gap * (len(self.actions) - 1) + group_extra * 2
        left = layout.toolbar_rect.x + (layout.toolbar_rect.width - total_width) // 2
        button_height = min(layout.scaled(58), layout.toolbar_rect.height - layout.scaled(20))
        y1 = layout.toolbar_rect.y + (layout.toolbar_rect.height - button_height) // 2
        y2 = y1 + button_height
        self.buttons = []
        x = left
        for index, action in enumerate(self.actions):
            if index in (6, 8):
                x += group_extra
            right = x + button_width
            self.buttons.append(ToolbarButton(action.upper(), action, (x, y1, right, y2)))
            x = right + gap
        self._panel_cache = None
        self._panel_mask = None

    def update_states(self, can_undo: bool, can_redo: bool) -> None:
        for button in self.buttons:
            button.enabled = button.action not in {"undo", "redo"} or (can_undo if button.action == "undo" else can_redo)

    def get_hovered_button(self, x: int, y: int) -> ToolbarButton | None:
        """Return the actual rendered button under a full-frame pixel point."""
        hovered: ToolbarButton | None = None
        for button in self.buttons:
            button.hovered = button.contains(x, y)
            if button.hovered:
                hovered = button
        return hovered

    def is_inside_safe_area(self, action_id: str, x: int, y: int) -> bool:
        button = next((item for item in self.buttons if item.action == action_id), None)
        if button is None:
            return False
        x1, y1, x2, y2 = button.rectangle
        return x1 + TOOLBAR_SAFE_INSET <= x <= x2 - TOOLBAR_SAFE_INSET and y1 + TOOLBAR_SAFE_INSET <= y <= y2 - TOOLBAR_SAFE_INSET

    def hit_test(self, point: tuple[int, int]) -> str | None:
        button = self.get_hovered_button(*point)
        return None if button is None else button.action

    def clear_hover(self) -> None:
        for button in self.buttons:
            button.hovered = False

    def draw(self, frame: np.ndarray, active_tool_id: str, hovered_action_id: str | None, dwell_progress: float, now: float, *, can_undo: bool, can_redo: bool, whiteboard: bool) -> None:
        if not self.buttons:
            return
        self.update_states(can_undo, can_redo)
        x1, y1, x2, y2 = self._layout.toolbar_rect.bounds
        bounds = (x1, y1, x2, y2)
        panel_width, panel_height = x2 - x1, y2 - y1
        if self._panel_cache is None or self._panel_cache.shape[:2] != (panel_height, panel_width):
            self._panel_cache = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            self._panel_mask = np.zeros((panel_height, panel_width), dtype=np.uint8)
            rounded_rectangle(self._panel_cache, (0, 0, panel_width - 1, panel_height - 1), 20, (7, 15, 26), -1)
            rounded_rectangle(self._panel_mask, (0, 0, panel_width - 1, panel_height - 1), 20, (255, 255, 255), -1)
        assert self._panel_mask is not None
        roi = frame[y1:y2, x1:x2]
        blended = cv2.addWeighted(roi, 0.18, self._panel_cache, 0.82, 0)
        roi[self._panel_mask > 0] = blended[self._panel_mask > 0]
        rounded_rectangle(frame, bounds, 20, (105, 100, 180), 1)

        for button in self.buttons:
            active = button.action == active_tool_id
            color = COLORS.get(button.action.title(), (120, 130, 150))
            fill = tuple(max(18, int(channel * (0.38 if button.hovered else 0.24))) for channel in color)
            if not button.enabled:
                fill = (18, 23, 31)
            rounded_rectangle(frame, button.rectangle, 14, fill, -1)
            if active:
                pulse = 0.5 + 0.5 * math.sin(now * 5.5)
                glow_color = tuple(min(255, int(channel * (0.72 + pulse * 0.35))) for channel in color)
                bounds = button.rectangle
                rounded_rectangle(frame, (bounds[0] - 3, bounds[1] - 3, bounds[2] + 3, bounds[3] + 3), 17, tuple(int(c * 0.35) for c in glow_color), 4)
                rounded_rectangle(frame, bounds, 14, glow_color, 2)
                cv2.circle(frame, ((bounds[0] + bounds[2]) // 2, bounds[3] - 5), 2, (245, 250, 255), -1, cv2.LINE_AA)
            elif button.hovered:
                rounded_rectangle(frame, button.rectangle, 14, (100, 245, 255) if button.enabled else (110, 120, 135), 3)
            else:
                rounded_rectangle(frame, button.rectangle, 14, (75, 92, 115), 1)
            if button.action == hovered_action_id and dwell_progress > 0:
                x1b, y1b, x2b, y2b = button.rectangle
                cv2.line(frame, (x1b + 5, y2b - 4), (x1b + 5 + int((x2b - x1b - 10) * dwell_progress), y2b - 4), (100, 245, 255), 4, cv2.LINE_AA)
            label = ("CAM" if whiteboard else "BOARD") if button.action == "board" else button.label.upper()
            scale = self._layout.scaled_font(0.34 if len(label) > 5 else 0.39)
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, 1)[0]
            tx = button.rectangle[0] + (button.rectangle[2] - button.rectangle[0] - size[0]) // 2
            ty = button.rectangle[1] + (button.rectangle[3] - button.rectangle[1] + size[1]) // 2
            text_color = (235, 244, 255) if button.enabled else (74, 82, 96)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, scale, text_color, 1, cv2.LINE_AA)

            if DEBUG_TOOLBAR_INTERACTION:
                cv2.rectangle(frame, (button.rectangle[0], button.rectangle[1]), (button.rectangle[2], button.rectangle[3]), (255, 110, 50), 1, cv2.LINE_AA)

        hovered = next((button for button in self.buttons if button.hovered), None)
        if hovered is not None:
            cv2.putText(frame, f"HOVER: {hovered.label.upper()}", (x1 + self._layout.scaled(8), y2 + self._layout.scaled(20)), cv2.FONT_HERSHEY_DUPLEX, self._layout.scaled_font(0.42), (100, 245, 255), 1, cv2.LINE_AA)

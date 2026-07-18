"""Cached cyber-tech header, help panel and status bar rendering."""

from __future__ import annotations

import cv2
import numpy as np

from .toolbar import rounded_rectangle
from .ui_layout import UILayout
from .config import SHOW_GESTURE_HELP


class UIRenderer:
    """Build static chrome once per frame size and add dynamic status text."""

    def __init__(self) -> None:
        self._size: tuple[int, int] = (0, 0)
        self._static: np.ndarray | None = None
        self._static_mask: np.ndarray | None = None
        self._visible: np.ndarray | None = None
        self.help_visible = SHOW_GESTURE_HELP

    def _ensure_cache(self, layout: UILayout) -> None:
        width, height = layout.window_width, layout.window_height
        signature = (width, height, layout.header_rect.height, layout.status_bar_rect.height)
        if getattr(self, "_layout_signature", None) == signature:
            return
        self._layout_signature = signature
        self._size = (width, height)
        layer = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        header_h = layout.header_rect.height
        status_y = layout.status_bar_rect.y
        cv2.rectangle(layer, (0, 0), (width, header_h), (5, 12, 24), -1)
        cv2.rectangle(mask, (0, 0), (width, header_h), 225, -1)
        cv2.line(layer, (0, header_h), (width, header_h), (235, 70, 200), 1, cv2.LINE_AA)
        cv2.rectangle(mask, (0, status_y), (width, height), 230, -1)
        cv2.rectangle(layer, (0, status_y), (width, height), (6, 14, 25), -1)

        self._static, self._static_mask = layer, mask
        self._visible = mask > 0

    def draw(self, frame: np.ndarray, *, layout: UILayout, fps: float, tracking: bool, hand_count: int = 0, tool: str, brush_size: int, mode: str, glow: bool, skeleton: bool, view: str, history_position: int, history_total: int, quality: str = "high", assist: str = "off", auto_shape: bool = False, cleanup: str = "balanced") -> None:
        height, width = frame.shape[:2]
        self._ensure_cache(layout)
        assert self._static is not None and self._visible is not None
        blended = cv2.addWeighted(frame, 0.16, self._static, 0.84, 0)
        frame[self._visible] = blended[self._visible]

        s, sf = layout.scaled, layout.scaled_font
        cv2.putText(frame, "AIR CANVAS", (s(22), s(36)), cv2.FONT_HERSHEY_DUPLEX, sf(0.82), (255, 235, 255), max(1, s(2)), cv2.LINE_AA)
        cv2.putText(frame, "GESTURE CONTROLLED DRAWING SYSTEM", (s(23), s(62)), cv2.FONT_HERSHEY_SIMPLEX, sf(0.39), (160, 174, 198), 1, cv2.LINE_AA)
        live_color = (70, 255, 145) if tracking else (70, 100, 140)
        cv2.circle(frame, (width - s(246), s(29)), s(5), live_color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"{hand_count} HAND{'S' if hand_count != 1 else ''} TRACKED" if tracking else "SEARCHING FOR HAND", (width - s(232), s(34)), cv2.FONT_HERSHEY_DUPLEX, sf(0.42), live_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:05.1f} FPS", (width - s(112), s(62)), cv2.FONT_HERSHEY_DUPLEX, sf(0.44), (100, 220, 255), 1, cv2.LINE_AA)

        bottom = height - s(14)
        status = f"TOOL {tool.upper()}  |  BRUSH {brush_size:02d}px  |  MODE {mode.upper()}  |  VIEW {view.upper()}  |  FX {quality.upper()}  |  HISTORY {history_position}/{history_total}"
        cv2.putText(frame, status, (s(18), bottom), cv2.FONT_HERSHEY_DUPLEX, sf(0.40), (190, 215, 240), 1, cv2.LINE_AA)
        shortcuts = "S SAVE  Z/Y HISTORY  P QUALITY  T TRAILS  K PARTICLES  H HAND  X SWAP  W BOARD  Q QUIT"
        shortcut_size = cv2.getTextSize(shortcuts, cv2.FONT_HERSHEY_SIMPLEX, sf(0.29), 1)[0]
        cv2.putText(frame, shortcuts, (max(s(18), width - shortcut_size[0] - s(18)), height - s(31)), cv2.FONT_HERSHEY_SIMPLEX, sf(0.29), (95, 140, 180), 1, cv2.LINE_AA)
        assistance = f"ASSIST {assist.upper()}  |  AUTO SHAPE {'ON' if auto_shape else 'OFF'}  |  CLEANUP {cleanup.upper()}"
        cv2.putText(frame, assistance, (s(18), height - s(31)), cv2.FONT_HERSHEY_SIMPLEX, sf(0.29), (90, 205, 210), 1, cv2.LINE_AA)
        toggles = f"GLOW {'ON' if glow else 'OFF'} / HAND {'ON' if skeleton else 'OFF'}"
        cv2.putText(frame, toggles, (width - s(260), bottom), cv2.FONT_HERSHEY_SIMPLEX, sf(0.29), (80, 220, 190), 1, cv2.LINE_AA)

    def draw_help_panel(self, frame: np.ndarray, layout: UILayout) -> None:
        """Render help last in final window coordinates, never camera space."""
        if not self.help_visible:
            return
        panel = layout.gesture_panel_rect
        shadow = (panel.x + layout.scaled(5), panel.y + layout.scaled(6), panel.x2 + layout.scaled(5), panel.y2 + layout.scaled(6))
        overlay = frame.copy()
        rounded_rectangle(overlay, shadow, layout.scaled(16), (2, 5, 10), -1)
        rounded_rectangle(overlay, panel.bounds, layout.scaled(16), (8, 18, 32), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        rounded_rectangle(frame, panel.bounds, layout.scaled(16), (60, 220, 255), 1)
        self._draw_help_content(frame, layout)

    @staticmethod
    def _draw_help_content(frame: np.ndarray, layout: UILayout) -> None:
        panel, s, sf = layout.gesture_panel_rect, layout.scaled, layout.scaled_font
        panel_x = panel.x + s(14)
        cv2.putText(frame, "GESTURE MATRIX", (panel_x, panel.y + s(30)), cv2.FONT_HERSHEY_DUPLEX, sf(0.46), (100, 230, 255), 1, cv2.LINE_AA)
        help_rows = (("01", "INDEX", "DRAW"), ("02", "TWO FINGERS", "SELECT"), ("03", "PINCH", "ACTIVATE"), ("04", "FIST", "PAUSE"), ("05", "OPEN PALM", "CLEAR"))
        y = panel.y + s(64)
        for icon, gesture, action in help_rows:
            cv2.circle(frame, (panel_x + s(11), y - s(5)), s(11), (34, 56, 78), -1, cv2.LINE_AA)
            cv2.putText(frame, icon, (panel_x + s(4), y - s(1)), cv2.FONT_HERSHEY_SIMPLEX, sf(0.27), (90, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, gesture, (panel_x + s(32), y - s(4)), cv2.FONT_HERSHEY_DUPLEX, sf(0.34), (220, 230, 242), 1, cv2.LINE_AA)
            cv2.putText(frame, action, (panel_x + s(32), y + s(12)), cv2.FONT_HERSHEY_SIMPLEX, sf(0.29), (120, 150, 178), 1, cv2.LINE_AA)
            y += s(36)


    @staticmethod
    def draw_toolbar_debug(
        frame: np.ndarray,
        *,
        fingertip: tuple[int, int] | None,
        gesture: str,
        pinch_distance: float,
        pinch_active: bool,
        hovered: str | None,
        toolbar_y: tuple[int, int],
    ) -> None:
        """Show the exact hit-test inputs and coordinate space."""
        if fingertip is not None:
            x, y = fingertip
            cv2.line(frame, (x - 9, y), (x + 9, y), (30, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - 9), (x, y + 9), (30, 255, 255), 1, cv2.LINE_AA)
        lines = (
            f"TIP PX: {fingertip if fingertip else 'NONE'}",
            f"GESTURE: {gesture}",
            f"PINCH DIST: {pinch_distance:.4f}",
            f"PINCH ACTIVE: {'YES' if pinch_active else 'NO'}",
            f"HOVERED: {hovered or 'NONE'}",
            f"TOOLBAR Y: {toolbar_y[0]}..{toolbar_y[1]}",
            f"FRAME/WINDOW: {frame.shape[1]}x{frame.shape[0]}",
        )
        x, y = 16, 200
        panel = frame.copy()
        cv2.rectangle(panel, (8, y - 23), (285, y + len(lines) * 19 + 5), (4, 10, 18), -1)
        cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
        for line in lines:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (90, 240, 255), 1, cv2.LINE_AA)
            y += 19

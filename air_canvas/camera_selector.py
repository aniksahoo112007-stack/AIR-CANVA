"""Mouse-driven OpenCV camera dropdown with test preview and use actions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera_manager import CameraInfo
from .toolbar import rounded_rectangle
from .ui_layout import UILayout


@dataclass(frozen=True)
class Rect:
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    @property
    def tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class CameraSelector:
    """Keep camera UI geometry and mouse hit testing in one object."""

    def __init__(self) -> None:
        self.cameras: list[CameraInfo] = []
        self.active_index: int | None = None
        self.open = False
        self.hovered_index: int | None = None
        self.control_rect = Rect(0, 0, 0, 0)
        self.refresh_rect = Rect(0, 0, 0, 0)
        self.panel_rect = Rect(0, 0, 0, 0)
        self.row_rects: dict[int, Rect] = {}
        self.test_rects: dict[int, Rect] = {}
        self.use_rects: dict[int, Rect] = {}
        self._selected_camera: int | None = None
        self._test_camera: int | None = None
        self._refresh_requested = False
        self.preview: np.ndarray | None = None
        self.preview_index: int | None = None
        self.preview_status = ""

    def set_cameras(self, cameras: list[CameraInfo], active_index: int | None) -> None:
        self.cameras = list(cameras)
        self.active_index = active_index

    def show_no_feed(self) -> None:
        self.open = True

    def update_mouse(self, event: int, x: int, y: int, _flags: int = 0) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.hovered_index = next((index for index, rect in self.row_rects.items() if rect.contains(x, y)), None)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.control_rect.contains(x, y):
            self.open = not self.open
            return
        if self.refresh_rect.contains(x, y):
            self._refresh_requested = True
            self.open = True
            return
        if self.open:
            for index, rect in self.test_rects.items():
                if rect.contains(x, y):
                    self._test_camera = index
                    return
            for index, rect in self.use_rects.items():
                if rect.contains(x, y):
                    self._selected_camera = index
                    self.open = False
                    return
            if self.panel_rect.contains(x, y):
                return
        self.open = False

    def get_selected_camera(self) -> int | None:
        selected, self._selected_camera = self._selected_camera, None
        return selected

    def get_test_camera(self) -> int | None:
        requested, self._test_camera = self._test_camera, None
        return requested

    def get_refresh_requested(self) -> bool:
        requested, self._refresh_requested = self._refresh_requested, False
        return requested

    def set_test_result(self, index: int, info: CameraInfo | None, preview: np.ndarray | None) -> None:
        self.preview_index = index
        self.preview = preview
        if info is not None and preview is not None:
            self.preview_status = "USABLE" if info.usable else "QUESTIONABLE - FRAMES RECEIVED"
        else:
            self.preview_status = "NO SIGNAL"

    def render(self, frame: np.ndarray, live: bool, layout: UILayout | None = None) -> None:
        height, width = frame.shape[:2]
        layout = layout or UILayout.create(width, height, width, height)
        s, sf = layout.scaled, layout.scaled_font
        control_w, control_h = s(176), s(34)
        x = max(s(330), width - s(590))
        top = s(11)
        self.control_rect = Rect(x, top, x + control_w, top + control_h)
        self.refresh_rect = Rect(x + control_w + s(8), top, x + control_w + s(42), top + control_h)
        active = "NONE" if self.active_index is None else str(self.active_index)
        rounded_rectangle(frame, self.control_rect.tuple, 9, (15, 27, 42), -1)
        rounded_rectangle(frame, self.control_rect.tuple, 9, (80, 220, 255) if live else (40, 80, 255), 1)
        cv2.putText(frame, f"CAMERA {active}  v", (x + s(10), top + s(23)), cv2.FONT_HERSHEY_DUPLEX, sf(0.40), (225, 238, 250), 1, cv2.LINE_AA)
        rounded_rectangle(frame, self.refresh_rect.tuple, 9, (15, 27, 42), -1)
        cv2.putText(frame, "R", (self.refresh_rect.x1 + s(11), top + s(23)), cv2.FONT_HERSHEY_DUPLEX, sf(0.44), (100, 230, 255), 1, cv2.LINE_AA)
        info = next((item for item in self.cameras if item.index == self.active_index), None)
        resolution = "NO SIGNAL" if info is None else f"{info.width}x{info.height}  {'LIVE' if live else 'NO SIGNAL'}"
        status_color = (70, 255, 145) if live else (50, 90, 255)
        cv2.putText(frame, resolution, (x, top + control_h + s(17)), cv2.FONT_HERSHEY_SIMPLEX, sf(0.32), status_color, 1, cv2.LINE_AA)
        if not self.open:
            return
        row_height = s(56)
        rows_height = max(1, len(self.cameras)) * row_height
        preview_height = s(125) if self.preview is not None or self.preview_status else s(42)
        panel_width = min(s(680), width - s(24))
        panel_x2 = min(width - s(12), self.refresh_rect.x2)
        panel_x1 = max(s(12), panel_x2 - panel_width)
        panel_top = layout.header_rect.height + s(4)
        self.panel_rect = Rect(panel_x1, panel_top, panel_x2, min(height - s(8), panel_top + s(24) + rows_height + preview_height))
        panel = frame.copy()
        rounded_rectangle(panel, self.panel_rect.tuple, 14, (7, 15, 26), -1)
        cv2.addWeighted(panel, 0.93, frame, 0.07, 0, frame)
        rounded_rectangle(frame, self.panel_rect.tuple, 14, (75, 190, 230), 1)
        cv2.putText(frame, "SELECT CAMERA", (self.panel_rect.x1 + s(12), panel_top + s(20)), cv2.FONT_HERSHEY_DUPLEX, sf(0.40), (110, 225, 255), 1, cv2.LINE_AA)
        if self.active_index is None:
            cv2.putText(frame, "NO CAMERA FEED - TEST OR USE A CAMERA BELOW", (self.panel_rect.x1 + 155, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 100, 255), 1, cv2.LINE_AA)
        self.row_rects.clear(); self.test_rects.clear(); self.use_rects.clear()
        y = panel_top + s(28)
        for camera in self.cameras:
            row = Rect(self.panel_rect.x1 + s(8), y, self.panel_rect.x2 - s(8), y + row_height - s(6))
            self.row_rects[camera.index] = row
            if camera.index == self.hovered_index:
                rounded_rectangle(frame, row.tuple, 8, (28, 48, 67), -1)
            marker = "ACTIVE" if camera.index == self.active_index else "READY" if camera.usable else "QUESTIONABLE"
            name = "Built-in Camera" if camera.index == 0 else "USB Camera Candidate" if camera.index == 1 else "Camera Candidate"
            cv2.putText(frame, f"{name} {camera.index}  {camera.width}x{camera.height}  {camera.backend_name}", (row.x1 + s(8), y + s(20)), cv2.FONT_HERSHEY_SIMPLEX, sf(0.34), (225, 232, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, marker, (row.x1 + s(8), y + s(39)), cv2.FONT_HERSHEY_SIMPLEX, sf(0.28), (70, 255, 145) if camera.index == self.active_index else (110, 170, 205), 1, cv2.LINE_AA)
            test_rect = Rect(row.x2 - s(104), y + s(10), row.x2 - s(55), y + s(39))
            use_rect = Rect(row.x2 - s(49), y + s(10), row.x2 - s(4), y + s(39))
            self.test_rects[camera.index] = test_rect; self.use_rects[camera.index] = use_rect
            rounded_rectangle(frame, test_rect.tuple, 6, (30, 55, 76), -1)
            rounded_rectangle(frame, use_rect.tuple, 6, (34, 80, 65), -1)
            cv2.putText(frame, "TEST", (test_rect.x1 + 5, test_rect.y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (130, 225, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "USE", (use_rect.x1 + 7, use_rect.y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (110, 255, 175), 1, cv2.LINE_AA)
            y += row_height
        if not self.cameras:
            cv2.putText(frame, "NO CAMERA FEED - click R to refresh", (self.panel_rect.x1 + 12, y + 24), cv2.FONT_HERSHEY_DUPLEX, 0.38, (50, 100, 255), 1, cv2.LINE_AA)
        elif self.preview is not None:
            preview = cv2.resize(self.preview, (160, 90), interpolation=cv2.INTER_AREA)
            py = min(y + 6, self.panel_rect.y2 - 98)
            frame[py:py + 90, self.panel_rect.x1 + 12:self.panel_rect.x1 + 172] = preview
            preview_color = (100, 235, 170) if self.preview_status == "USABLE" else (80, 190, 255)
            cv2.putText(frame, f"CAMERA {self.preview_index}: {self.preview_status}", (self.panel_rect.x1 + 184, py + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.32, preview_color, 1, cv2.LINE_AA)
        elif self.preview_status:
            cv2.putText(frame, f"CAMERA {self.preview_index}: {self.preview_status}", (self.panel_rect.x1 + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (50, 100, 255), 1, cv2.LINE_AA)

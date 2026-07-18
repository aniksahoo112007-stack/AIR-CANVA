"""Canonical window-space layout and native-camera display mapping."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import (
    CAMERA_DISPLAY_MODE, UI_REFERENCE_HEIGHT, UI_REFERENCE_WIDTH,
    UI_SCALE_MAX, UI_SCALE_MIN, UI_SCALE_OVERRIDE,
    CAMERA_ZOOM_DEFAULT, CAMERA_ZOOM_MIN, CAMERA_ZOOM_MAX, CAMERA_ZOOM_STEP,
    CAMERA_PAN_X_DEFAULT, CAMERA_PAN_Y_DEFAULT,
)


@dataclass
class CameraViewState:
    """Session-only zoom target and normalized framing for one camera."""

    current_zoom: float = CAMERA_ZOOM_DEFAULT
    target_zoom: float = CAMERA_ZOOM_DEFAULT
    pan_x: float = CAMERA_PAN_X_DEFAULT
    pan_y: float = CAMERA_PAN_Y_DEFAULT

    def set_zoom(self, value: float) -> None:
        self.target_zoom = max(CAMERA_ZOOM_MIN, min(CAMERA_ZOOM_MAX, value))

    def zoom_in(self) -> None:
        self.set_zoom(round(self.target_zoom + CAMERA_ZOOM_STEP, 2))

    def zoom_out(self) -> None:
        self.set_zoom(round(self.target_zoom - CAMERA_ZOOM_STEP, 2))

    def reset_view(self) -> None:
        self.current_zoom = CAMERA_ZOOM_DEFAULT
        self.target_zoom = CAMERA_ZOOM_DEFAULT
        self.pan_x = CAMERA_PAN_X_DEFAULT
        self.pan_y = CAMERA_PAN_Y_DEFAULT

    def pan(self, dx: float, dy: float) -> None:
        self.pan_x = max(-1.0, min(1.0, self.pan_x + dx))
        self.pan_y = max(-1.0, min(1.0, self.pan_y + dy))

    def update(self, elapsed: float) -> bool:
        before = self.current_zoom
        alpha = min(1.0, max(0.0, elapsed) / 0.20)
        self.current_zoom += (self.target_zoom - self.current_zoom) * alpha
        if abs(self.target_zoom - self.current_zoom) < 0.001:
            self.current_zoom = self.target_zoom
        return self.current_zoom != before


@dataclass(frozen=True)
class LayoutRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x2, self.y2


@dataclass(frozen=True)
class UILayout:
    window_width: int
    window_height: int
    ui_scale: float
    header_rect: LayoutRect
    toolbar_rect: LayoutRect
    content_rect: LayoutRect
    camera_rect: LayoutRect
    gesture_panel_rect: LayoutRect
    debug_panel_rect: LayoutRect
    status_bar_rect: LayoutRect
    display_mode: str
    display_scale: float
    crop_x: float
    crop_y: float
    letterbox_x: float
    letterbox_y: float
    pan_x: float
    pan_y: float

    @classmethod
    def create(cls, window_width: int, window_height: int, camera_width: int, camera_height: int, display_mode: str | None = None,
               zoom: float = CAMERA_ZOOM_DEFAULT, pan_x: float = CAMERA_PAN_X_DEFAULT, pan_y: float = CAMERA_PAN_Y_DEFAULT) -> "UILayout":
        width, height = max(640, window_width), max(480, window_height)
        base = min(width / UI_REFERENCE_WIDTH, height / UI_REFERENCE_HEIGHT)
        scale = max(UI_SCALE_MIN, min(UI_SCALE_MAX, base)) * UI_SCALE_OVERRIDE
        s = lambda value: max(1, int(round(value * scale)))
        header_h, status_h = s(86), s(50)
        content = LayoutRect(0, header_h, width, max(1, height - header_h - status_h))
        mode = (display_mode or CAMERA_DISPLAY_MODE).lower()
        if mode not in {"cover", "center_crop", "fit"}:
            mode = "cover"
        camera, display_scale, crop_x, crop_y, letterbox_x, letterbox_y = cls._camera_transform(
            content, camera_width, camera_height, mode, zoom, pan_x, pan_y)
        toolbar_w = min(width - s(24), s(1260))
        toolbar = LayoutRect((width - toolbar_w) // 2, header_h + s(8), toolbar_w, s(78))
        panel_w = min(s(270), max(s(220), width // 4))
        gesture = LayoutRect(width - panel_w - s(16), toolbar.y2 + s(12), panel_w, s(250))
        debug = LayoutRect(s(16), toolbar.y2 + s(12), min(s(320), width - s(32)), s(170))
        return cls(width, height, scale, LayoutRect(0, 0, width, header_h), toolbar, content, camera,
                   gesture, debug, LayoutRect(0, height - status_h, width, status_h),
                   "cover" if mode == "center_crop" else mode, display_scale, crop_x, crop_y,
                   letterbox_x, letterbox_y, pan_x, pan_y)

    @staticmethod
    def _camera_transform(content: LayoutRect, camera_width: int, camera_height: int, mode: str,
                          zoom: float, pan_x: float, pan_y: float) -> tuple[LayoutRect, float, float, float, float, float]:
        if camera_width <= 0 or camera_height <= 0:
            return content, 1.0, 0.0, 0.0, 0.0, 0.0
        if mode == "fit":
            base_scale = min(content.width / camera_width, content.height / camera_height)
        else:
            base_scale = max(content.width / camera_width, content.height / camera_height)
        display_scale = base_scale * max(CAMERA_ZOOM_MIN, min(CAMERA_ZOOM_MAX, zoom))
        scaled_width = int(round(camera_width * display_scale))
        scaled_height = int(round(camera_height * display_scale))
        overflow_x = max(0, scaled_width - content.width)
        overflow_y = max(0, scaled_height - content.height)
        crop_x = overflow_x * (1.0 - max(-1.0, min(1.0, pan_x))) / 2.0
        crop_y = overflow_y * (1.0 - max(-1.0, min(1.0, pan_y))) / 2.0
        letterbox_x = max(0.0, (content.width - scaled_width) / 2.0)
        letterbox_y = max(0.0, (content.height - scaled_height) / 2.0)
        return content, display_scale, crop_x, crop_y, letterbox_x, letterbox_y

    def scaled(self, value: float) -> int:
        return max(1, int(round(value * self.ui_scale)))

    def scaled_font(self, value: float) -> float:
        return max(0.1, value * self.ui_scale)

    def camera_to_window(self, point: tuple[int, int], camera_size: tuple[int, int]) -> tuple[int, int]:
        width, height = camera_size
        if width <= 0 or height <= 0:
            return point
        return (
            self.content_rect.x + int(round(point[0] * self.display_scale - self.crop_x + self.letterbox_x)),
            self.content_rect.y + int(round(point[1] * self.display_scale - self.crop_y + self.letterbox_y)),
        )

    def window_to_camera(self, point: tuple[int, int], camera_size: tuple[int, int]) -> tuple[int, int]:
        width, height = camera_size
        return (
            int(round((point[0] - self.content_rect.x + self.crop_x - self.letterbox_x) / max(1e-9, self.display_scale))),
            int(round((point[1] - self.content_rect.y + self.crop_y - self.letterbox_y) / max(1e-9, self.display_scale))),
        )

    def present(self, camera_frame: np.ndarray, background: tuple[int, int, int] = (2, 7, 14)) -> np.ndarray:
        output = np.full((self.window_height, self.window_width, 3), background, dtype=np.uint8)
        content = self.content_rect
        source_height, source_width = camera_frame.shape[:2]
        scaled_width = max(1, int(round(source_width * self.display_scale)))
        scaled_height = max(1, int(round(source_height * self.display_scale)))
        resized = cv2.resize(camera_frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        destination_x = content.x + int(round(self.letterbox_x - self.crop_x))
        destination_y = content.y + int(round(self.letterbox_y - self.crop_y))
        x1, y1 = max(content.x, destination_x), max(content.y, destination_y)
        x2, y2 = min(content.x2, destination_x + scaled_width), min(content.y2, destination_y + scaled_height)
        if x2 > x1 and y2 > y1:
            source_x, source_y = x1 - destination_x, y1 - destination_y
            output[y1:y2, x1:x2] = resized[source_y:source_y + (y2 - y1), source_x:source_x + (x2 - x1)]
        return output

"""Generate deterministic UI previews for visual regression inspection."""

from __future__ import annotations

import cv2
import numpy as np

from air_canvas.camera_manager import CameraInfo
from air_canvas.camera_selector import CameraSelector
from air_canvas.camera_view_controls import CameraViewControls
from air_canvas.toolbar import Toolbar
from air_canvas.ui_layout import UILayout
from air_canvas.ui_renderer import UIRenderer


def render(width: int, height: int, filename: str, zoom: float = 1.0) -> None:
    source = np.zeros((height, width, 3), dtype=np.uint8)
    source[:, :, 0] = 80
    source[:, :, 1] = np.linspace(30, 210, height, dtype=np.uint8)[:, None]
    source[:, :, 2] = np.linspace(20, 180, width, dtype=np.uint8)[None, :]
    layout = UILayout.create(1600, 900, width, height, zoom=zoom)
    frame = layout.present(source)
    ui = UIRenderer()
    ui.draw(
        frame, layout=layout, fps=30.0, tracking=True, hand_count=1, tool="Red",
        brush_size=7, mode="Draw", glow=True, skeleton=True, view="Camera",
        history_position=2, history_total=4,
    )
    toolbar = Toolbar(1600)
    toolbar.update_layout(layout)
    toolbar.draw(frame, "red", None, 0.0, 0.0, can_undo=True, can_redo=True, whiteboard=False)
    ui.draw_help_panel(frame, layout)
    selector = CameraSelector()
    selector.set_cameras([CameraInfo(0, cv2.CAP_DSHOW, "DSHOW", width, height, 30.0)], 0)
    selector.render(frame, True, layout)
    CameraViewControls().render(frame, layout, zoom)
    if not cv2.imwrite(filename, frame):
        raise OSError(f"Could not write {filename}")


if __name__ == "__main__":
    render(640, 480, "outputs/ui_640x480.png")
    render(640, 480, "outputs/ui_640x480_zoom_065.png", 0.65)
    render(1280, 720, "outputs/ui_1280x720.png")

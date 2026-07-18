"""Deterministic Phase 2 canvas/history/whiteboard integration checks."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from air_canvas.gesture_detector import Gesture
from air_canvas.drawing_canvas import DrawingCanvas
from air_canvas.toolbar import Toolbar
from main import AirCanvasApp


def stroke(app: AirCanvasApp, start: tuple[int, int], end: tuple[int, int], now: float) -> None:
    midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    app._handle_gesture(Gesture.DRAW, start, None, False, 0.2, now, (640, 480), 1, 0.9)
    app._handle_gesture(Gesture.DRAW, midpoint, None, False, 0.2, now + 0.02, (640, 480), 1, 0.9)
    app._handle_gesture(Gesture.DRAW, end, None, False, 0.2, now + 0.04, (640, 480), 1, 0.9)
    app._handle_gesture(Gesture.SELECT, end, None, False, 0.2, now + 0.05, (640, 480), 1, 0.9)


def run() -> None:
    app = AirCanvasApp()
    app.canvas = DrawingCanvas(640, 480)
    app.toolbar = Toolbar(640)
    app.current_camera_frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    app._apply_fullscreen = lambda: None  # type: ignore[method-assign]
    now = time.monotonic()

    states: list[np.ndarray] = []
    for index in range(3):
        y = 180 + index * 45
        stroke(app, (100, y), (250, y), now + index)
        states.append(app.canvas.mask.copy())
    assert app.history.current_position == 3
    assert app.history.total_states == 3

    for expected in (states[1], states[0], np.zeros_like(states[0])):
        app._undo(now)
        assert np.array_equal(app.canvas.mask, expected)
    assert not app.history.can_undo()
    assert app.history.current_position == 0 and app.history.total_states == 3

    for expected in states:
        app._redo(now)
        assert np.array_equal(app.canvas.mask, expected)
    assert not app.history.can_redo()

    before_erase = app.canvas.mask.copy()
    app.active_tool_id = "eraser"
    stroke(app, (100, 225), (250, 225), now + 4)
    assert int(app.canvas.mask.sum()) < int(before_erase.sum())
    app._undo(now)
    assert np.array_equal(app.canvas.mask, before_erase)

    before_clear = app.canvas.mask.copy()
    app._clear(now, None, (640, 480))
    assert not app.canvas.mask.any()
    app._undo(now)
    assert np.array_equal(app.canvas.mask, before_clear)

    app._undo(now)
    app.active_tool_id = "blue"
    stroke(app, (320, 300), (440, 330), now + 5)
    assert not app.history.can_redo()

    drawing_before_view = app.canvas.image.copy()
    mask_before_view = app.canvas.mask.copy()
    app._toggle_whiteboard(now)
    assert app.whiteboard_mode and app.whiteboard_theme == "light"
    app._toggle_board_theme(now)
    assert app.whiteboard_mode and app.whiteboard_theme == "dark"
    assert np.array_equal(app.canvas.image, drawing_before_view)
    assert np.array_equal(app.canvas.mask, mask_before_view)

    stroke(app, (300, 380), (480, 380), now + 6)
    whiteboard_stroke = app.canvas.mask.copy()
    app._undo(now)
    app._redo(now)
    assert np.array_equal(app.canvas.mask, whiteboard_stroke)
    app._toggle_whiteboard(now)
    assert not app.whiteboard_mode
    assert np.array_equal(app.canvas.mask, whiteboard_stroke)

    app._handle_key(26, now, (640, 480), control=True, shift=False)
    app._handle_key(26, now, (640, 480), control=True, shift=True)
    app._handle_key(25, now, (640, 480), control=True, shift=False)

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        transparent = app.canvas.save(output)
        assert cv2.imread(str(transparent), cv2.IMREAD_UNCHANGED).shape[2] == 4
        board = app.canvas.save(output, background=(16, 20, 30))
        assert cv2.imread(str(board), cv2.IMREAD_UNCHANGED).shape[2] == 3
        composite = app.canvas.save_composite(app.current_camera_frame, output)
        assert cv2.imread(str(composite), cv2.IMREAD_UNCHANGED).shape[2] == 3

    print("Phase 2 history, edge cases, shortcuts, views, and exports: OK")


if __name__ == "__main__":
    run()

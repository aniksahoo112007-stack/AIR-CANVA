"""Toolbar coordinate, hover, edge activation, and fallback-key checks."""

from __future__ import annotations

import time

import numpy as np

from air_canvas.drawing_canvas import DrawingCanvas
from air_canvas.gesture_detector import Gesture, GestureDetector
from air_canvas.hand_tracker import TrackedHand
from air_canvas.toolbar import Toolbar
from main import AirCanvasApp


def run() -> None:
    toolbar = Toolbar(1280)
    expected = ["red", "blue", "green", "yellow", "white", "eraser", "undo", "redo", "board", "clear", "save"]
    assert [button.action for button in toolbar.buttons] == expected
    for button in toolbar.buttons:
        x1, y1, x2, y2 = button.rectangle
        hovered = toolbar.get_hovered_button((x1 + x2) // 2, (y1 + y2) // 2)
        assert hovered is button and hovered.hovered
        assert toolbar.y_range[0] <= (y1 + y2) // 2 <= toolbar.y_range[1]

    app = AirCanvasApp()
    app.canvas = DrawingCanvas(1280, 720)
    app.toolbar = toolbar
    activated: list[str] = []
    def record(action: str) -> None:
        activated.append(action)
        app.last_activated_action_id = action
    app.execute_toolbar_action = record  # type: ignore[method-assign]
    now = time.monotonic()
    blue = toolbar.buttons[1]
    bx = (blue.rectangle[0] + blue.rectangle[2]) // 2
    by = (blue.rectangle[1] + blue.rectangle[3]) // 2

    app._handle_gesture(Gesture.SELECT, (bx, by), "blue", False, 0.08, now, (1280, 720), 1, 0.9)
    app._handle_gesture(Gesture.PINCH, (bx, by), "blue", True, 0.04, now + 0.01, (1280, 720), 1, 0.9)
    app._handle_gesture(Gesture.PINCH, (bx, by), "blue", True, 0.04, now + 2.0, (1280, 720), 1, 0.9)
    assert activated == ["blue"], "A held pinch must activate exactly once"
    assert not app.canvas.mask.any(), "Selection and pinch must never draw"

    app._handle_gesture(Gesture.SELECT, (500, 400), None, False, 0.08, now + 2.1, (1280, 720), 1, 0.9)
    app._handle_gesture(Gesture.SELECT, (bx, by), "blue", False, 0.08, now + 2.5, (1280, 720), 1, 0.9)
    app._handle_gesture(Gesture.PINCH, (bx, by), "blue", True, 0.04, now + 2.6, (1280, 720), 1, 0.9)
    assert activated == ["blue", "blue"], "Leaving a button must unlock optional pinch activation"

    app._handle_gesture(Gesture.SELECT, (500, 400), None, False, 0.09, now + 2.3, (1280, 720), 1, 0.9)
    assert app.pinch_previous is False and app.debug_hovered is None

    activated.clear()
    for index, button in enumerate(toolbar.buttons):
        x1, y1, x2, y2 = button.rectangle
        point = ((x1 + x2) // 2, (y1 + y2) // 2)
        moment = now + 3.0 + index
        app._handle_gesture(Gesture.SELECT, point, button.action, False, 0.09, moment, (1280, 720), 1, 0.9)
        app._handle_gesture(Gesture.PINCH, point, button.action, True, 0.04, moment + 0.02, (1280, 720), 1, 0.9)
    assert activated == expected, "Every visible button must pinch-activate its shared action"

    activated.clear()
    for key in "1234567890":
        app._handle_key(ord(key), now, (1280, 720))
    assert activated == expected[:9], "0 is reserved for camera-view reset; Clear remains on C and the toolbar"

    points = np.zeros((21, 3), dtype=np.float32)
    points[:, :2] = (0.5, 0.8)
    for mcp, pip, tip, x in ((5, 6, 8, 0.43), (9, 10, 12, 0.48), (13, 14, 16, 0.53), (17, 18, 20, 0.58)):
        points[mcp, :2] = (x, 0.62)
        points[pip, :2] = (x, 0.44)
        points[tip, :2] = (x, 0.20)
    points[3, :2] = (0.50, 0.75)
    points[4, :2] = (0.50, 0.79)
    hand = TrackedHand(points, np.zeros((21, 2), dtype=np.int32), "Right")
    assert GestureDetector().detect(hand).gesture is Gesture.SELECT

    print("Toolbar hitboxes, tolerant selection, pinch edges, and number fallbacks: OK")


if __name__ == "__main__":
    run()

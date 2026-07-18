"""Non-camera tests for authoritative toolbar state and action dispatch."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from air_canvas.config import COLORS
from air_canvas.drawing_canvas import DrawingCanvas
from air_canvas.toolbar import Toolbar
from main import AirCanvasApp


class ToolbarDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = AirCanvasApp()
        self.app.canvas = DrawingCanvas(640, 480)
        self.app.toolbar = Toolbar(640)
        self.app._action_size = (640, 480)

    def assert_selected(self, action_id: str) -> None:
        self.assertEqual(self.app.active_tool_id, action_id)
        self.assertIn(action_id, [button.action for button in self.app.toolbar.buttons])
        # This is the exact value passed to the status renderer in the event loop.
        self.assertEqual(self.app.active_tool_id.title(), action_id.title())

    def test_color_and_eraser_state(self) -> None:
        self.app.execute_toolbar_action("blue")
        self.assert_selected("blue")
        self.assertEqual(self.app.active_color, COLORS["Blue"])

        self.app.execute_toolbar_action("green")
        self.assert_selected("green")
        self.assertEqual(self.app.active_color, COLORS["Green"])

        self.app.execute_toolbar_action("eraser")
        self.assert_selected("eraser")
        self.assertEqual(self.app.active_color, (255, 255, 255))

    def test_selected_tool_drives_actual_drawing_and_erasing(self) -> None:
        self.app.execute_toolbar_action("blue")
        self.app.canvas.draw_to((100, 100), self.app.active_color, 7, self.app.active_tool_id == "eraser")
        self.app.canvas.draw_to((120, 100), self.app.active_color, 7, self.app.active_tool_id == "eraser")
        self.assertTrue(self.app.canvas.mask.any())
        self.assertTrue(((self.app.canvas.image == COLORS["Blue"]).all(axis=2) & (self.app.canvas.mask > 0)).any())

        self.app.canvas.reset_stroke()
        self.app.execute_toolbar_action("eraser")
        self.app.canvas.draw_to((100, 100), self.app.active_color, 7, self.app.active_tool_id == "eraser")
        self.app.canvas.draw_to((120, 100), self.app.active_color, 7, self.app.active_tool_id == "eraser")
        self.assertLess(self.app.canvas.mask.sum(), 255 * 200)

    def test_command_actions_are_connected(self) -> None:
        for action_id, method_name in (
            ("undo", "_undo"), ("redo", "_redo"), ("board", "_toggle_whiteboard"),
            ("clear", "_clear"), ("save", "_save"),
        ):
            method = Mock()
            setattr(self.app, method_name, method)
            self.app.execute_toolbar_action(action_id)
            method.assert_called_once()
            self.assertEqual(self.app.last_activated_action_id, action_id)

    def test_dwell_activates_once_and_leave_cancels(self) -> None:
        start = 100.0
        point = (100, 100)
        blue = next(button for button in self.app.toolbar.buttons if button.action == "blue")
        point = ((blue.rectangle[0] + blue.rectangle[2]) // 2, (blue.rectangle[1] + blue.rectangle[3]) // 2)
        self.app._update_toolbar_dwell("blue", point, start)
        self.app._update_toolbar_dwell("blue", point, start + 0.17)
        self.assertEqual(self.app.active_tool_id, "red")
        self.app._update_toolbar_dwell(None, (0, 0), start + 0.18)
        self.assertEqual(self.app.active_tool_id, "red")

        self.app._update_toolbar_dwell("blue", point, start + 1.0)
        self.app._update_toolbar_dwell("blue", point, start + 1.19)
        self.assertEqual(self.app.active_tool_id, "blue")
        self.app._update_toolbar_dwell("blue", point, start + 3.0)
        self.assertEqual(self.app.last_activated_action_id, "blue")
        self.assertTrue(self.app.toolbar_selection_locked)


if __name__ == "__main__":
    unittest.main()

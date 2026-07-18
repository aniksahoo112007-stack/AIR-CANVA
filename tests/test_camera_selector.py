"""Mouse hit testing and queued selector actions."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from air_canvas.camera_manager import CameraInfo
from air_canvas.camera_selector import CameraSelector


class CameraSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = CameraSelector()
        self.selector.set_cameras([
            CameraInfo(0, cv2.CAP_DSHOW, "DSHOW", 1280, 720, 30.0),
            CameraInfo(1, cv2.CAP_DSHOW, "DSHOW", 1280, 720, 30.0),
        ], 1)
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.selector.render(self.frame, True)

    def test_click_control_test_use_refresh_and_outside(self) -> None:
        control = self.selector.control_rect
        self.selector.update_mouse(cv2.EVENT_LBUTTONDOWN, control.x1 + 3, control.y1 + 3)
        self.assertTrue(self.selector.open)
        self.selector.render(self.frame, True)
        test = self.selector.test_rects[0]
        self.selector.update_mouse(cv2.EVENT_LBUTTONDOWN, test.x1 + 2, test.y1 + 2)
        self.assertEqual(self.selector.get_test_camera(), 0)
        use = self.selector.use_rects[0]
        self.selector.update_mouse(cv2.EVENT_LBUTTONDOWN, use.x1 + 2, use.y1 + 2)
        self.assertEqual(self.selector.get_selected_camera(), 0)
        refresh = self.selector.refresh_rect
        self.selector.update_mouse(cv2.EVENT_LBUTTONDOWN, refresh.x1 + 2, refresh.y1 + 2)
        self.assertTrue(self.selector.get_refresh_requested())
        self.selector.update_mouse(cv2.EVENT_LBUTTONDOWN, 2, 300)
        self.assertFalse(self.selector.open)


if __name__ == "__main__":
    unittest.main()

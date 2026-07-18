"""Camera discovery and preference tests without physical hardware."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from air_canvas.camera_manager import CameraManager


class FakeCapture:
    cameras = {0: (640, 480, 30.0), 2: (1920, 1080, 30.0), 3: (640, 480, 30.0)}
    instances: list["FakeCapture"] = []

    def __init__(self, index: int, backend: int) -> None:
        self.index = index
        self.backend = backend
        self.opened = index in self.cameras and backend == cv2.CAP_DSHOW
        self.released = False
        self.frame_number = 0
        self.instances.append(self)

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def set(self, _property: int, _value: float) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return self.cameras[self.index][2] if property_id == cv2.CAP_PROP_FPS and self.index in self.cameras else 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.isOpened():
            return False, None
        width, height, _ = self.cameras[self.index]
        self.frame_number += 1
        if self.index == 3:
            return True, np.zeros((height, width, 3), dtype=np.uint8)
        frame = np.full((height, width, 3), 45 + self.frame_number, dtype=np.uint8)
        frame[:, self.frame_number % width] = 180
        return True, frame

    def release(self) -> None:
        self.released = True


class CameraManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeCapture.instances.clear()

    @patch("air_canvas.camera_manager.cv2.VideoCapture", side_effect=FakeCapture)
    def test_discovers_and_prefers_best_nonzero_camera(self, _factory: object) -> None:
        manager = CameraManager()
        self.assertTrue(manager.open_preferred_camera())
        self.assertEqual([info.index for info in manager.available_cameras], [0, 2])
        self.assertEqual([info.index for info in manager.detected_cameras], [0, 2, 3])
        self.assertFalse(manager.detected_cameras[-1].usable)
        self.assertIsNotNone(manager.active_info)
        self.assertEqual(manager.active_info.index, 2)  # type: ignore[union-attr]
        self.assertEqual(manager.active_info.backend_name, "DSHOW")  # type: ignore[union-attr]
        manager.release()
        self.assertIsNone(manager.capture)

    @patch("air_canvas.camera_manager.cv2.VideoCapture", side_effect=FakeCapture)
    def test_override_and_switch_preserve_a_working_capture(self, _factory: object) -> None:
        manager = CameraManager()
        self.assertTrue(manager.open_preferred_camera(0))
        self.assertEqual(manager.active_info.index, 0)  # type: ignore[union-attr]
        self.assertTrue(manager.switch_camera(1))
        self.assertEqual(manager.active_info.index, 2)  # type: ignore[union-attr]

    @patch("air_canvas.camera_manager.cv2.VideoCapture", side_effect=FakeCapture)
    def test_questionable_frames_can_be_force_selected(self, _factory: object) -> None:
        manager = CameraManager()
        self.assertTrue(manager.open_camera(3))
        self.assertFalse(manager.active_info.usable)  # type: ignore[union-attr]
        self.assertEqual(manager.active_info.index, 3)  # type: ignore[union-attr]

    @patch("air_canvas.camera_manager.cv2.VideoCapture", side_effect=FakeCapture)
    def test_temporary_test_capture_is_always_released(self, _factory: object) -> None:
        manager = CameraManager()
        info, preview = manager.test_camera(2)
        self.assertIsNotNone(info)
        self.assertIsNotNone(preview)
        self.assertTrue(all(capture.released for capture in FakeCapture.instances))


if __name__ == "__main__":
    unittest.main()

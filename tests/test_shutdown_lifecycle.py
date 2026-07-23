"""Regression tests for authoritative application shutdown."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock, patch

from air_canvas.annotation_controller import AnnotationController
from air_canvas.camera_manager import CameraManager
from main import AirCanvasApp


class ShutdownLifecycleTests(unittest.TestCase):
    def test_window_visibility_check_starts_after_first_presented_frame(self) -> None:
        app = AirCanvasApp()
        app._window_created = True

        with patch("main.cv2.getWindowProperty", return_value=0) as visibility:
            self.assertTrue(app._main_window_is_visible())

        visibility.assert_not_called()

    def test_main_window_close_stops_before_camera_or_window_recreation(self) -> None:
        app = AirCanvasApp()
        app.tracker = Mock()
        app.dual_tracker = Mock()
        app.canvas = Mock()
        app.toolbar = Mock()
        app._window_created = True
        app._window_presented = True
        app.camera_manager = Mock()

        with patch("main.cv2.getWindowProperty", return_value=0):
            self.assertEqual(app._event_loop(), 0)

        self.assertTrue(app.shutdown_requested.is_set())
        self.assertEqual(app._shutdown_reason, "window closed")
        app.camera_manager.read_frame.assert_not_called()

    def test_shutdown_is_idempotent_and_does_not_restore_a_hidden_window(self) -> None:
        app = AirCanvasApp()
        app.canvas = Mock()
        app.tracker = Mock()
        app.dual_tracker = Mock()
        app.camera_manager = Mock()
        app.desktop_annotation = Mock()
        app.desktop_hotkeys = Mock()
        app._desktop_window_hidden = True
        tracker = app.tracker
        camera_manager = app.camera_manager
        desktop_annotation = app.desktop_annotation
        desktop_hotkeys = app.desktop_hotkeys

        with (
            patch("main.cv2.destroyAllWindows") as destroy_windows,
            patch("main.cv2.moveWindow") as move_window,
            patch("main.cv2.resizeWindow") as resize_window,
        ):
            app.request_shutdown("test")
            app.shutdown()
            app.shutdown()

        camera_manager.release.assert_called_once()
        tracker.close.assert_called_once()
        desktop_annotation.close.assert_called_once()
        desktop_hotkeys.unregister_active.assert_called_once()
        desktop_hotkeys.close.assert_called_once()
        destroy_windows.assert_called_once()
        move_window.assert_not_called()
        resize_window.assert_not_called()
        self.assertTrue(app._shutdown_complete)

    def test_escape_q_and_desktop_exit_use_authoritative_shutdown_event(self) -> None:
        for key, expected_reason in ((27, "escape"), (ord("q"), "keyboard quit")):
            with self.subTest(key=key):
                app = AirCanvasApp()
                app.canvas = Mock()
                self.assertFalse(app._handle_key(key, 1.0, (640, 480)))
                self.assertTrue(app.shutdown_requested.is_set())
                self.assertEqual(app._shutdown_reason, expected_reason)

        app = AirCanvasApp()
        app.desktop_annotation = Mock(active=True)
        app._handle_desktop_action("exit", 1.0)
        self.assertTrue(app.shutdown_requested.is_set())
        self.assertEqual(app._shutdown_reason, "desktop overlay exit")

    def test_annotation_toolbar_exit_marks_full_app_exit_request(self) -> None:
        controller = AnnotationController.__new__(AnnotationController)
        controller.active = True
        controller.exit_requested = False
        controller.notify = Mock()

        controller._activate("exit")

        self.assertFalse(controller.active)
        self.assertTrue(controller.exit_requested)

    def test_camera_recovery_stops_when_shared_shutdown_event_is_set(self) -> None:
        shutdown_requested = threading.Event()
        manager = CameraManager(shutdown_requested)
        shutdown_requested.set()

        with (
            patch.object(manager, "open_camera") as open_camera,
            patch.object(manager, "discover_cameras") as discover_cameras,
        ):
            self.assertFalse(manager.recover_camera())

        open_camera.assert_not_called()
        discover_cameras.assert_not_called()
        self.assertIs(manager.shutdown_requested, shutdown_requested)


if __name__ == "__main__":
    unittest.main()

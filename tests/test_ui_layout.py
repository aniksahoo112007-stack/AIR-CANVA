"""Responsive window layout invariants across camera resolutions."""

from __future__ import annotations

import unittest

import numpy as np
import cv2

from air_canvas.camera_view_controls import CameraViewControls
from air_canvas.camera_manager import CameraInfo
from air_canvas.drawing_canvas import DrawingCanvas
from air_canvas.toolbar import Toolbar
from air_canvas.ui_layout import CameraViewState, UILayout
from air_canvas.ui_renderer import UIRenderer
from main import AirCanvasApp


class UILayoutTests(unittest.TestCase):
    def test_chrome_is_independent_of_camera_resolution(self) -> None:
        usb = UILayout.create(1600, 900, 640, 480)
        builtin = UILayout.create(1600, 900, 1280, 720)
        self.assertEqual(usb.ui_scale, 1.0)
        self.assertEqual(usb.header_rect, builtin.header_rect)
        self.assertEqual(usb.toolbar_rect, builtin.toolbar_rect)
        self.assertEqual(usb.gesture_panel_rect, builtin.gesture_panel_rect)
        self.assertEqual(usb.status_bar_rect, builtin.status_bar_rect)

    def test_fit_preserves_aspect_ratio_and_coordinate_mapping(self) -> None:
        layout = UILayout.create(1600, 900, 640, 480, "fit")
        scaled_width = 640 * layout.display_scale
        scaled_height = 480 * layout.display_scale
        self.assertAlmostEqual(scaled_width / scaled_height, 4 / 3, places=2)
        native = (319, 211)
        displayed = layout.camera_to_window(native, (640, 480))
        restored = layout.window_to_camera(displayed, (640, 480))
        self.assertLessEqual(abs(restored[0] - native[0]), 1)
        self.assertLessEqual(abs(restored[1] - native[1]), 1)
        frame = layout.present(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(frame.shape, (900, 1600, 3))

    def test_cover_fills_content_and_round_trips_cropped_coordinates(self) -> None:
        for camera_size in ((640, 480), (1280, 720)):
            layout = UILayout.create(1600, 900, *camera_size, "cover")
            self.assertEqual(layout.camera_rect, layout.content_rect)
            self.assertEqual(layout.display_mode, "cover")
            self.assertGreaterEqual(layout.display_scale, layout.content_rect.width / camera_size[0])
            self.assertGreaterEqual(layout.display_scale, layout.content_rect.height / camera_size[1])
            self.assertTrue(layout.crop_x > 0 or layout.crop_y > 0)
            center = (camera_size[0] // 2, camera_size[1] // 2)
            displayed = layout.camera_to_window(center, camera_size)
            self.assertAlmostEqual(displayed[0], layout.content_rect.x + layout.content_rect.width // 2, delta=1)
            self.assertAlmostEqual(displayed[1], layout.content_rect.y + layout.content_rect.height // 2, delta=1)
            restored = layout.window_to_camera(displayed, camera_size)
            self.assertAlmostEqual(restored[0], center[0], delta=1)
            self.assertAlmostEqual(restored[1], center[1], delta=1)
            frame = layout.present(np.zeros((camera_size[1], camera_size[0], 3), dtype=np.uint8))
            content = layout.content_rect
            self.assertEqual(frame[content.y:content.y2, content.x:content.x2].shape[:2], (content.height, content.width))

    def test_toolbar_is_compact_and_does_not_overlap_gesture_panel(self) -> None:
        for window in ((1600, 900), (1200, 675), (1920, 1080)):
            layout = UILayout.create(*window, 640, 480)
            toolbar = Toolbar(window[0])
            toolbar.update_layout(layout)
            self.assertLessEqual(layout.toolbar_rect.height, int(window[1] * 0.12))
            self.assertLessEqual(layout.toolbar_rect.y2, layout.gesture_panel_rect.y)
            self.assertEqual(len(toolbar.buttons), 11)
            self.assertLessEqual(toolbar.buttons[-1].rectangle[2], layout.toolbar_rect.x2)

    def test_zoom_out_reveals_more_and_pan_uses_same_point_transform(self) -> None:
        normal = UILayout.create(1600, 900, 640, 480, "cover", 1.0)
        zoomed_out = UILayout.create(1600, 900, 640, 480, "cover", 0.65)
        self.assertLess(zoomed_out.display_scale, normal.display_scale)
        self.assertGreater(zoomed_out.letterbox_x, 0)
        self.assertLess(zoomed_out.crop_y, normal.crop_y)
        panned = UILayout.create(1600, 900, 640, 480, "cover", 1.4, 0.5, -0.5)
        point = (420, 260)
        displayed = panned.camera_to_window(point, (640, 480))
        restored = panned.window_to_camera(displayed, (640, 480))
        self.assertAlmostEqual(restored[0], point[0], delta=1)
        self.assertAlmostEqual(restored[1], point[1], delta=1)

    def test_zoom_state_clamps_animates_and_resets(self) -> None:
        state = CameraViewState()
        for _ in range(30):
            state.zoom_out()
        self.assertEqual(state.target_zoom, 0.65)
        state.update(0.1)
        self.assertGreater(state.current_zoom, state.target_zoom)
        state.pan(5.0, -5.0)
        self.assertEqual((state.pan_x, state.pan_y), (1.0, -1.0))
        state.reset_view()
        self.assertEqual((state.current_zoom, state.target_zoom, state.pan_x, state.pan_y), (1.0, 1.0, 0.0, 0.0))

    def test_visible_zoom_controls_share_rendered_hitboxes(self) -> None:
        layout = UILayout.create(1600, 900, 640, 480, "cover")
        controls = CameraViewControls()
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)
        controls.render(frame, layout, 1.0)
        for expected, rect in (("out", controls.minus_rect), ("in", controls.plus_rect), ("reset", controls.reset_rect)):
            action = controls.update_mouse(cv2.EVENT_LBUTTONDOWN, (rect.x1 + rect.x2) // 2, (rect.y1 + rect.y2) // 2)
            self.assertEqual(action, expected)

    def test_zoom_and_pan_memory_is_separate_per_camera(self) -> None:
        app = AirCanvasApp()
        app.camera_manager.active_info = CameraInfo(0, cv2.CAP_DSHOW, "DSHOW", 1280, 720, 30.0)
        camera_zero = app._active_view_state()
        camera_zero.set_zoom(1.4)
        camera_zero.pan(0.2, -0.1)
        app.camera_manager.active_info = CameraInfo(1, cv2.CAP_DSHOW, "DSHOW", 640, 480, 30.0)
        camera_one = app._active_view_state()
        camera_one.set_zoom(0.7)
        self.assertIsNot(camera_zero, camera_one)
        self.assertEqual((camera_one.target_zoom, camera_one.pan_x, camera_one.pan_y), (0.7, 0.0, 0.0))
        app.camera_manager.active_info = CameraInfo(0, cv2.CAP_DSHOW, "DSHOW", 1280, 720, 30.0)
        self.assertEqual((app._active_view_state().target_zoom, app._active_view_state().pan_x, app._active_view_state().pan_y), (1.4, 0.2, -0.1))

    def test_windows_mouse_wheel_flags_decode_in_both_directions(self) -> None:
        self.assertEqual(AirCanvasApp._mouse_wheel_delta(120 << 16), 120)
        self.assertEqual(AirCanvasApp._mouse_wheel_delta(((-120) & 0xFFFF) << 16), -120)

    def test_app_mouse_zoom_and_pan_routing(self) -> None:
        app = AirCanvasApp()
        app.camera_manager.active_info = CameraInfo(1, cv2.CAP_DSHOW, "DSHOW", 640, 480, 30.0)
        app.layout = UILayout.create(1600, 900, 640, 480, "cover", 1.0)
        app.camera_view_controls.update_layout(app.layout)
        minus = app.camera_view_controls.minus_rect
        app._on_mouse(cv2.EVENT_LBUTTONDOWN, (minus.x1 + minus.x2) // 2, (minus.y1 + minus.y2) // 2, 0)
        self.assertEqual(app._active_view_state().target_zoom, 0.9)
        content = app.layout.content_rect
        center = (content.x + content.width // 2, content.y + content.height // 2)
        app._on_mouse(cv2.EVENT_MOUSEWHEEL, *center, 120 << 16)
        self.assertEqual(app._active_view_state().target_zoom, 1.0)
        app._on_mouse(cv2.EVENT_RBUTTONDOWN, *center, 0)
        app._on_mouse(cv2.EVENT_MOUSEMOVE, center[0] + 80, center[1] - 40, 0)
        app._on_mouse(cv2.EVENT_RBUTTONUP, center[0] + 80, center[1] - 40, 0)
        self.assertGreater(app._active_view_state().pan_x, 0)
        self.assertLess(app._active_view_state().pan_y, 0)

    def test_help_panel_is_window_space_and_visible_by_default(self) -> None:
        ui = UIRenderer()
        self.assertTrue(ui.help_visible)
        layouts = (
            UILayout.create(1600, 900, 640, 480, "cover", 0.65, 0.5, -0.5),
            UILayout.create(1600, 900, 1280, 720, "fit", 2.0, -0.5, 0.5),
        )
        self.assertEqual(layouts[0].gesture_panel_rect, layouts[1].gesture_panel_rect)
        for layout in layouts:
            panel = layout.gesture_panel_rect
            self.assertGreaterEqual(panel.x, 0)
            self.assertGreaterEqual(panel.y, layout.header_rect.y2)
            self.assertLessEqual(panel.x2, layout.window_width)
            self.assertLessEqual(panel.y2, layout.status_bar_rect.y)
            frame = np.full((900, 1600, 3), 120, dtype=np.uint8)
            before = frame[panel.y:panel.y2, panel.x:panel.x2].copy()
            ui.draw_help_panel(frame, layout)
            self.assertFalse(np.array_equal(before, frame[panel.y:panel.y2, panel.x:panel.x2]))

    def test_j_toggles_help_without_touching_layout(self) -> None:
        app = AirCanvasApp()
        app.canvas = DrawingCanvas(640, 480)
        original_layout = app.layout
        app._handle_key(ord("j"), 1.0, (640, 480))
        self.assertFalse(app.ui.help_visible)
        app._handle_key(ord("J"), 2.0, (640, 480))
        self.assertTrue(app.ui.help_visible)
        self.assertIs(app.layout, original_layout)


if __name__ == "__main__":
    unittest.main()

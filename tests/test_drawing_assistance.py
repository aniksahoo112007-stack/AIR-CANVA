"""Algorithm tests for stabilization, recognition, correction, and cleanup."""

from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from air_canvas.drawing_assistant import DrawingAssistant
from air_canvas.shape_recognizer import ShapeRecognizer, render_corrected_shape
from air_canvas.sketch_cleaner import SketchCleaner
from air_canvas.drawing_canvas import DrawingCanvas
from main import AirCanvasApp
from air_canvas.ui_layout import UILayout


class DrawingAssistanceTests(unittest.TestCase):
    def test_assistant_reduces_slow_tremor_and_preserves_fast_endpoint(self) -> None:
        assistant = DrawingAssistant("high")
        raw = [(20 + i * 4, 100 + (3 if i % 2 else -3)) for i in range(30)]
        stabilized = [assistant.add_point(point, 1.0 + i / 30) for i, point in enumerate(raw)]
        self.assertLess(np.std(np.asarray(stabilized[5:])[:, 1]), np.std(np.asarray(raw[5:])[:, 1]))
        fast = assistant.add_point((220, 100), 2.01)
        self.assertLess(abs(fast[0] - 220), 12)

    def test_recognizes_line_circle_rectangle_and_triangle(self) -> None:
        recognizer = ShapeRecognizer()
        line = [(20 + i * 6, 80 + (i % 3 - 1)) for i in range(30)]
        self.assertEqual(recognizer.recognize(line).shape_type, "line")
        circle = [(int(180 + 60 * math.cos(t)), int(160 + 60 * math.sin(t))) for t in np.linspace(0, 2 * math.pi, 70)]
        self.assertEqual(recognizer.recognize(circle).shape_type, "circle")
        rectangle = [(x, y) for x in range(80, 241, 8) for y in (80,)] + [(240, y) for y in range(88, 181, 8)] + [(x, 180) for x in range(240, 79, -8)] + [(80, y) for y in range(172, 79, -8)]
        self.assertIn(recognizer.recognize(rectangle).shape_type, {"rectangle", "square"})
        triangle = []
        for a, b in (((100, 220), (180, 70)), ((180, 70), (270, 220)), ((270, 220), (100, 220))):
            triangle.extend([(int(a[0] + (b[0]-a[0])*t), int(a[1] + (b[1]-a[1])*t)) for t in np.linspace(0, 1, 25)])
        self.assertEqual(recognizer.recognize(triangle).shape_type, "triangle")

    def test_random_freehand_is_not_forced_to_geometry(self) -> None:
        points = [(20, 100), (45, 60), (70, 110), (95, 55), (120, 120), (145, 65), (170, 130), (190, 80), (220, 145)]
        self.assertEqual(ShapeRecognizer().recognize(points).shape_type, "freehand")

    def test_corrected_shape_renders_and_cleanup_removes_only_tiny_dot(self) -> None:
        image = np.zeros((240, 320, 3), np.uint8); mask = np.zeros((240, 320), np.uint8)
        result = ShapeRecognizer().recognize([(20 + i * 5, 100 + i % 2) for i in range(45)])
        render_corrected_shape(image, mask, result, (0, 255, 0), 4)
        self.assertGreater(np.count_nonzero(mask), 100)
        cv2.line(image, (40, 180), (260, 180), (255, 0, 0), 4); cv2.line(mask, (40, 180), (260, 180), 255, 4)
        image[20, 20] = (255, 255, 255); mask[20, 20] = 255
        cleaned = SketchCleaner().clean(image, mask, "balanced")
        self.assertEqual(cleaned.mask[20, 20], 0)
        self.assertGreater(np.count_nonzero(cleaned.mask[178:183, 40:261]), 500)

    def test_shape_accept_is_undoable_back_to_rough_stroke(self) -> None:
        app = AirCanvasApp(); app.canvas = DrawingCanvas(320, 240)
        app.drawing_assistant.set_assistance_level("off")
        points = [(int(160 + 60 * math.cos(t)), int(120 + 60 * math.sin(t))) for t in np.linspace(0, 2 * math.pi, 70)]
        before = app.history.snapshot(app.canvas.image, app.canvas.mask)
        app.history.begin_stroke(app.canvas.image, app.canvas.mask)
        rough = np.asarray([(x + (i % 3 - 1) * 2, y) for i, (x, y) in enumerate(points)], np.int32)
        cv2.polylines(app.canvas.image, [rough.reshape(-1, 1, 2)], True, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.polylines(app.canvas.mask, [rough.reshape(-1, 1, 2)], True, 255, 4, cv2.LINE_AA)
        rough_image, rough_mask = app.canvas.image.copy(), app.canvas.mask.copy()
        app._stroke_before = before; app._stroke_color = (0, 0, 255); app._stroke_size = 4
        app.stroke_in_progress = True
        for i, point in enumerate(points):
            app.drawing_assistant.add_point(point, 1.0 + i / 30)
        app._end_stroke("test")
        self.assertIsNotNone(app.pending_shape)
        app._accept_pending_preview(4.0)
        self.assertFalse(np.array_equal(app.canvas.mask, rough_mask))
        app._undo(5.0)
        self.assertTrue(np.array_equal(app.canvas.image, rough_image))
        self.assertTrue(np.array_equal(app.canvas.mask, rough_mask))

    def test_cleanup_cancel_is_non_mutating_and_apply_is_undoable(self) -> None:
        app = AirCanvasApp(); app.canvas = DrawingCanvas(320, 240)
        cv2.line(app.canvas.image, (30, 120), (280, 120), (255, 0, 0), 4, cv2.LINE_AA)
        cv2.line(app.canvas.mask, (30, 120), (280, 120), 255, 4, cv2.LINE_AA)
        app.canvas.image[20, 20] = 255; app.canvas.mask[20, 20] = 255
        original_image, original_mask = app.canvas.image.copy(), app.canvas.mask.copy()
        app._request_cleanup(1.0); self.assertIsNotNone(app.pending_cleanup)
        app._reject_pending_preview()
        self.assertTrue(np.array_equal(app.canvas.mask, original_mask))
        app._request_cleanup(2.0); app._accept_pending_preview(2.1)
        self.assertEqual(app.canvas.mask[20, 20], 0)
        app._undo(3.0)
        self.assertTrue(np.array_equal(app.canvas.image, original_image))
        self.assertTrue(np.array_equal(app.canvas.mask, original_mask))

    def test_secondary_panel_mouse_actions_cycle_and_toggle(self) -> None:
        app = AirCanvasApp(); app.canvas = DrawingCanvas(640, 480)
        app.layout = UILayout.create(1600, 900, 640, 480)
        frame = np.zeros((900, 1600, 3), np.uint8)
        app.drawing_assist_ui.render_controls(frame, app.layout, app.drawing_assistant.level, app.auto_shape, app.cleanup_intensity)
        assist = app.drawing_assist_ui.rects["assist"]
        app._on_mouse(cv2.EVENT_LBUTTONDOWN, (assist.x1+assist.x2)//2, (assist.y1+assist.y2)//2, 0)
        self.assertEqual(app.drawing_assistant.level, "high")
        shape = app.drawing_assist_ui.rects["shape"]
        app._on_mouse(cv2.EVENT_LBUTTONDOWN, (shape.x1+shape.x2)//2, (shape.y1+shape.y2)//2, 0)
        self.assertFalse(app.auto_shape)


if __name__ == "__main__":
    unittest.main()

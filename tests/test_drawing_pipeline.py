"""Deterministic lifecycle and safety tests for canvas drawing."""

from __future__ import annotations

import unittest
import math

import cv2
import numpy as np

from air_canvas.drawing_canvas import DrawingCanvas


class DrawingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = DrawingCanvas(640, 480)
        self.color = (255, 80, 0)

    def test_first_point_and_stationary_jitter_make_no_dots(self) -> None:
        self.assertTrue(self.canvas.start_stroke((100, 100), 7, 1.0))
        self.assertFalse(self.canvas.mask.any())
        self.assertFalse(self.canvas.append_stroke_point((101, 100), 7, 1.03, self.color, 7, False))
        self.assertFalse(self.canvas.mask.any())

    def test_slow_line_is_continuous(self) -> None:
        self.canvas.start_stroke((100, 100), 7, 1.0)
        for index in range(1, 21):
            self.canvas.append_stroke_point((100 + index * 4, 100), 7, 1.0 + index / 30, self.color, 5, False)
        occupied = self.canvas.mask[100, 100:175] > 0
        self.assertGreater(occupied.mean(), 0.9)

    def test_slow_circle_is_one_connected_stroke(self) -> None:
        points = [
            (round(300 + 70 * math.cos(index * math.tau / 90)), round(220 + 70 * math.sin(index * math.tau / 90)))
            for index in range(91)
        ]
        self.canvas.start_stroke(points[0], 4, 1.0)
        for index, point in enumerate(points[1:], 1):
            self.canvas.append_stroke_point(point, 4, 1.0 + index / 30, self.color, 5, False)
        components, _ = cv2.connectedComponents((self.canvas.mask > 0).astype(np.uint8))
        self.assertEqual(components, 2)  # background plus one continuous stroke
        self.assertFalse(self.canvas.mask[220, 300])

    def test_new_stroke_does_not_connect_to_old_stroke(self) -> None:
        self.canvas.start_stroke((50, 50), 1, 1.0)
        self.canvas.append_stroke_point((80, 50), 1, 1.03, self.color, 5, False)
        self.canvas.end_stroke("gesture ended")
        self.canvas.start_stroke((500, 350), 1, 2.0)
        self.canvas.append_stroke_point((530, 350), 1, 2.03, self.color, 5, False)
        self.assertFalse(self.canvas.mask[200, 300])

    def test_jump_gap_and_hand_change_reset_without_connection(self) -> None:
        self.canvas.start_stroke((100, 100), 1, 1.0)
        self.assertFalse(self.canvas.append_stroke_point((400, 300), 1, 1.03, self.color, 5, False))
        self.assertEqual(self.canvas.last_rejection_reason[:4], "jump")
        self.assertIsNone(self.canvas.previous_point)
        self.assertFalse(self.canvas.append_stroke_point((410, 300), 2, 2.0, self.color, 5, False))
        self.assertEqual(self.canvas.active_hand_id, 2)
        self.assertFalse(self.canvas.append_stroke_point((420, 300), 3, 2.02, self.color, 5, False))
        self.assertEqual(self.canvas.last_rejection_reason, "hand id changed")
        self.assertIsNone(self.canvas.active_hand_id)
        self.assertFalse(self.canvas.append_stroke_point((410, 300), 2, 2.0, self.color, 5, False))
        self.assertFalse(self.canvas.append_stroke_point((420, 300), 2, 2.3, self.color, 5, False))
        self.assertEqual(self.canvas.last_rejection_reason, "frame gap")
        self.assertFalse(self.canvas.mask.any())

    def test_nonfinite_outside_and_invalid_time_are_rejected(self) -> None:
        self.assertFalse(self.canvas.start_stroke((700, 10), 1, 1.0))
        self.assertEqual(self.canvas.last_rejection_reason, "outside drawable region")
        self.assertFalse(self.canvas.start_stroke((10, 10), 1, 0.0))
        self.assertEqual(self.canvas.last_rejection_reason, "invalid timestamp")


if __name__ == "__main__":
    unittest.main()

"""Deterministic dual-hand association and drawing filter tests."""

from __future__ import annotations

import unittest

import numpy as np

from air_canvas.dual_hand_tracker import DualHandTracker
from air_canvas.hand_tracker import TrackedHand
from air_canvas.stroke_smoother import StrokeSmoother


def hand_at(x: float, handedness: str, confidence: float = 0.9) -> TrackedHand:
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, :2] = (x, 0.72)
    for mcp, pip, tip, offset in ((5, 6, 8, -0.04), (9, 10, 12, 0.0), (13, 14, 16, 0.04), (17, 18, 20, 0.08)):
        points[mcp, :2] = (x + offset, 0.62)
        points[pip, :2] = (x + offset, 0.46)
        points[tip, :2] = (x + offset, 0.25)
    pixels = np.column_stack((points[:, 0] * 640, points[:, 1] * 480)).astype(np.int32)
    return TrackedHand(points, pixels, handedness, confidence)


class FakeTracker:
    def __init__(self, frames: list[list[TrackedHand]]) -> None:
        self.frames = frames

    def detect_all(self, _frame: np.ndarray) -> list[TrackedHand]:
        return self.frames.pop(0)


class DualHandAndSmoothingTests(unittest.TestCase):
    def test_identity_survives_mediapipe_order_swap(self) -> None:
        right, left = hand_at(0.3, "Right"), hand_at(0.7, "Left")
        tracker = DualHandTracker(FakeTracker([[right, left], [hand_at(0.69, "Left"), hand_at(0.31, "Right")]]))  # type: ignore[arg-type]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        first = tracker.detect(frame, 1.0)
        second = tracker.detect(frame, 1.03)
        first_ids = {hand.handedness: hand.tracking_id for hand in first}
        second_ids = {hand.handedness: hand.tracking_id for hand in second}
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(sum(hand.is_primary for hand in second), 1)
        self.assertTrue(next(hand for hand in second if hand.handedness == "Right").is_primary)

    def test_filter_interpolates_with_adaptive_ema(self) -> None:
        smoother = StrokeSmoother()
        self.assertEqual(smoother.update((10, 10), 1.0), (10, 10))
        accepted = smoother.update((20, 10), 1.02)
        self.assertIsNotNone(accepted)
        self.assertGreater(len(smoother.interpolate_points((10, 10), (20, 10))), 1)
        fast = smoother.update((100, 10), 1.04)
        self.assertGreater(fast[0], accepted[0])
        self.assertTrue(smoother.reject_outlier(np.array((np.nan, 1.0)), 0.02))


if __name__ == "__main__":
    unittest.main()

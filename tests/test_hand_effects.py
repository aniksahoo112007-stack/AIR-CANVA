"""Headless validation for independent cinematic hand-effect state."""

from __future__ import annotations

import unittest

import numpy as np

from air_canvas.dual_hand_tracker import RoleTrackedHand
from air_canvas.gesture_detector import Gesture
from air_canvas.hand_effects import HandEffectsRenderer


def effect_hand(tracking_id: int, x: int, gesture: Gesture, primary: bool) -> RoleTrackedHand:
    pixels = np.zeros((21, 2), dtype=np.int32)
    pixels[:] = (x, 280)
    pixels[0] = (x, 360)
    for mcp, pip, tip, offset in ((5, 6, 8, -45), (9, 10, 12, -15), (13, 14, 16, 18), (17, 18, 20, 48)):
        pixels[mcp] = (x + offset, 300)
        pixels[pip] = (x + offset, 245)
        pixels[tip] = (x + offset, 185)
    pixels[4] = (x - 75, 275)
    normalized = np.column_stack((pixels[:, 0] / 800, pixels[:, 1] / 500, np.zeros(21))).astype(np.float32)
    return RoleTrackedHand(tracking_id, "Right" if primary else "Left", 0.95, normalized, normalized, pixels, tuple(pixels[8]), tuple(pixels[4]), (True,) * 5, gesture, 0.04, gesture is Gesture.PINCH, 1.0, primary, not primary)


class HandEffectsTests(unittest.TestCase):
    def test_two_hands_render_independent_holograms(self) -> None:
        renderer = HandEffectsRenderer()
        hands = [effect_hand(1, 250, Gesture.OPEN_PALM, True), effect_hand(2, 570, Gesture.PINCH, False)]
        renderer.update(hands, 1.0, 30.0)
        renderer.update(hands, 1.1, 30.0)
        frame = np.zeros((500, 800, 3), dtype=np.uint8)
        renderer.render(frame, hands, 1.1, (400, 90))
        self.assertTrue(frame.any())
        self.assertIsNot(renderer.states[1], renderer.states[2])
        self.assertIsNot(renderer.energy[1], renderer.energy[2])
        self.assertEqual(renderer.states[1].current_gesture, Gesture.OPEN_PALM)
        self.assertEqual(renderer.states[2].current_gesture, Gesture.PINCH)
        self.assertEqual(len(renderer.energy[2].particles), 0)

    def test_orientation_is_smoothed_and_effect_state_resets_after_gap(self) -> None:
        renderer = HandEffectsRenderer()
        hand = effect_hand(1, 320, Gesture.DRAW, True)
        renderer.update([hand], 1.0, 30.0); renderer.update([hand], 1.1, 30.0)
        frame = np.zeros((500, 800, 3), dtype=np.uint8); renderer.render(frame, [hand], 1.1)
        initial = renderer.states[1].palm_angle
        hand.pixels[[5, 17]] = hand.pixels[[17, 5]]
        renderer.update([hand], 1.2, 30.0); renderer.render(frame, [hand], 1.2)
        self.assertLess(abs(renderer.states[1].palm_angle-initial), 90.0)
        stale_particle = object(); renderer.energy[1].particles.append(stale_particle)
        renderer.update([hand], 2.0, 30.0)
        self.assertNotIn(stale_particle, renderer.energy[1].particles)


if __name__ == "__main__":
    unittest.main()

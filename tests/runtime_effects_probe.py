"""Headless throughput probe for the complete two-hand effects renderer."""

from __future__ import annotations

import time

import numpy as np

from air_canvas.gesture_detector import Gesture
from air_canvas.hand_effects import HandEffectsRenderer
from test_hand_effects import effect_hand


def run(frames: int = 240) -> None:
    renderer = HandEffectsRenderer()
    hands = [effect_hand(1, 420, Gesture.DRAW, True), effect_hand(2, 980, Gesture.OPEN_PALM, False)]
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    started = time.perf_counter()
    for index in range(frames):
        now = index / 30.0 + 1.0
        renderer.update(hands, now, 30.0)
        output = frame.copy()
        renderer.render(output, hands, now)
    elapsed = time.perf_counter() - started
    print(f"EFFECTS_PROBE frames={frames} fps={frames / elapsed:.1f} quality={renderer.quality} particles={sum(len(system.particles) for system in renderer.energy.values())}")


if __name__ == "__main__":
    run()

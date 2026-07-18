"""Time-based, independently evolving visual state for one tracked hand."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .gesture_detector import Gesture


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


@dataclass
class HandEffectState:
    tracking_id: int
    current_gesture: Gesture = Gesture.IDLE
    previous_gesture: Gesture = Gesture.IDLE
    gesture_transition_time: float = 0.0
    ring_expansion: float = 0.65
    rotation_angles: list[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0, 45.0])
    pulse_strength: float = 0.0
    particle_intensity: float = 0.25
    trail_intensity: float = 0.4
    fade: float = 0.0
    pinch_flash_timer: float = 0.0
    clear_progress: float = 0.0
    clear_burst_emitted: bool = False
    last_position: np.ndarray | None = None
    last_update: float = 0.0
    last_seen: float = 0.0
    entered_at: float = 0.0

    def update(self, gesture: Gesture, palm: tuple[int, int], now: float, visible: bool = True) -> None:
        if self.last_update == 0.0:
            self.last_update = self.entered_at = now
        dt = min(0.08, max(0.0, now - self.last_update))
        self.last_update = now
        if visible:
            self.last_seen = now
            self.fade += (1.0 - self.fade) * min(1.0, dt * 7.0)
        else:
            self.fade += (0.0 - self.fade) * min(1.0, dt * 5.0)
        if gesture is not self.current_gesture:
            self.previous_gesture = self.current_gesture
            self.current_gesture = gesture
            self.gesture_transition_time = now
            if gesture is Gesture.PINCH:
                self.pinch_flash_timer = 1.0
            if gesture is not Gesture.OPEN_PALM:
                self.clear_progress = 0.0
                self.clear_burst_emitted = False
        targets = {
            Gesture.OPEN_PALM: (1.12, 0.75, 0.65), Gesture.DRAW: (0.72, 1.0, 1.0),
            Gesture.SELECT: (0.92, 0.72, 0.75), Gesture.PINCH: (0.58, 1.0, 0.9),
            Gesture.FIST: (0.45, 0.35, 0.35), Gesture.IDLE: (0.72, 0.25, 0.35),
        }
        expansion, particles, trail = targets.get(gesture, targets[Gesture.IDLE])
        blend = smoothstep(min(1.0, dt * 8.0))
        self.ring_expansion += (expansion - self.ring_expansion) * blend
        self.particle_intensity += (particles - self.particle_intensity) * blend
        self.trail_intensity += (trail - self.trail_intensity) * blend
        self.pulse_strength = 0.5 + 0.5 * math.sin(now * (11.0 if gesture in {Gesture.PINCH, Gesture.SELECT} else 6.0))
        speeds = (31.0, -48.0, 78.0, -22.0, 110.0)
        gesture_speed = 1.6 if gesture is Gesture.SELECT else 0.65 if gesture is Gesture.OPEN_PALM else 1.0
        for index in range(len(self.rotation_angles)):
            self.rotation_angles[index] = (self.rotation_angles[index] + speeds[index] * gesture_speed * dt) % 360.0
        self.pinch_flash_timer = max(0.0, self.pinch_flash_timer - dt * 3.2)
        if gesture is Gesture.OPEN_PALM:
            self.clear_progress = min(1.0, self.clear_progress + dt)
        self.last_position = np.asarray(palm, dtype=np.float32)

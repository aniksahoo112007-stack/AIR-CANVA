"""Gesture classification from MediaPipe hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import PINCH_RELEASE_THRESHOLD, PINCH_START_THRESHOLD
from .hand_tracker import TrackedHand


class Gesture(str, Enum):
    """Gestures understood by Air Canvas."""

    DRAW = "Drawing"
    SELECT = "Selection"
    FIST = "Paused"
    OPEN_PALM = "Clear (hold)"
    PINCH = "Pinch"
    IDLE = "Idle"


@dataclass(frozen=True)
class GestureState:
    """A classified gesture and individual raised-finger states."""

    gesture: Gesture
    fingers_up: tuple[bool, bool, bool, bool, bool]
    pinch_distance: float
    pinch_active: bool


class GestureDetector:
    """Classify robust, orientation-tolerant gestures from landmark geometry."""

    _tips = (4, 8, 12, 16, 20)
    _pips = (3, 6, 10, 14, 18)

    def __init__(self) -> None:
        self._pinch_active = False

    def detect(self, hand: TrackedHand) -> GestureState:
        points = hand.normalized
        index_up = self._extended(points, 8, 6, 5)
        middle_up = self._extended(points, 12, 10, 9)
        ring_up = self._extended(points, 16, 14, 13)
        pinky_up = self._extended(points, 20, 18, 17)

        thumb_tip_to_wrist = np.linalg.norm(points[4, :2] - points[0, :2])
        thumb_ip_to_wrist = np.linalg.norm(points[3, :2] - points[0, :2])
        thumb_up = bool(thumb_tip_to_wrist > thumb_ip_to_wrist * 1.08)
        fingers = (thumb_up, index_up, middle_up, ring_up, pinky_up)
        pinch_distance = float(np.linalg.norm(points[4, :2] - points[8, :2]))

        if self._pinch_active:
            self._pinch_active = pinch_distance < PINCH_RELEASE_THRESHOLD
        else:
            self._pinch_active = pinch_distance < PINCH_START_THRESHOLD

        if self._pinch_active:
            gesture = Gesture.PINCH
        elif all(fingers):
            gesture = Gesture.OPEN_PALM
        elif not any(fingers):
            gesture = Gesture.FIST
        elif index_up and middle_up:
            gesture = Gesture.SELECT
        elif index_up and not middle_up and not ring_up and not pinky_up:
            gesture = Gesture.DRAW
        else:
            gesture = Gesture.IDLE
        return GestureState(
            gesture=gesture,
            fingers_up=fingers,
            pinch_distance=pinch_distance,
            pinch_active=self._pinch_active,
        )

    @property
    def pinch_active(self) -> bool:
        return self._pinch_active

    def reset_pinch(self) -> None:
        """Reset pinch hysteresis after the interaction context ends."""
        self._pinch_active = False

    @staticmethod
    def _extended(points: np.ndarray, tip: int, pip: int, mcp: int) -> bool:
        wrist = points[0, :2]
        tip_distance = np.linalg.norm(points[tip, :2] - wrist)
        pip_distance = np.linalg.norm(points[pip, :2] - wrist)
        straightness = np.linalg.norm(points[tip, :2] - points[mcp, :2])
        bent_length = (
            np.linalg.norm(points[tip, :2] - points[pip, :2])
            + np.linalg.norm(points[pip, :2] - points[mcp, :2])
        )
        return bool(tip_distance > pip_distance * 1.12 and straightness > bent_length * 0.72)

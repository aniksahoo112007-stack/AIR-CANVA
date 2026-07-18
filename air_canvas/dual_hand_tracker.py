"""Stable hand identities and primary/secondary role assignment."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .config import ALLOW_HAND_ROLE_SWITCHING, PREFERRED_DRAWING_HAND
from .gesture_detector import Gesture, GestureDetector
from .hand_tracker import HandTracker, TrackedHand


@dataclass
class RoleTrackedHand:
    tracking_id: int
    handedness: str
    confidence: float
    raw_landmarks: np.ndarray
    smoothed_landmarks: np.ndarray
    pixels: np.ndarray
    index_tip: tuple[int, int]
    thumb_tip: tuple[int, int]
    fingers_up: tuple[bool, bool, bool, bool, bool]
    gesture: Gesture
    pinch_distance: float
    pinch_active: bool
    last_seen: float
    is_primary: bool = False
    is_secondary: bool = False


@dataclass
class _Track:
    tracking_id: int
    handedness: str
    wrist: np.ndarray
    smoothed: np.ndarray
    last_seen: float
    detector: GestureDetector


class DualHandTracker:
    """Associate unordered MediaPipe detections using handedness and wrist proximity."""

    def __init__(self, tracker: HandTracker) -> None:
        self.tracker = tracker
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self.primary_id: int | None = None
        self._roles_swapped = False

    def detect(self, frame: np.ndarray, now: float | None = None) -> list[RoleTrackedHand]:
        now = time.monotonic() if now is None else now
        detections = self.tracker.detect_all(frame)
        available = set(self._tracks)
        assigned: list[tuple[TrackedHand, _Track]] = []
        for hand in sorted(detections, key=lambda item: item.confidence, reverse=True):
            wrist = hand.normalized[0, :2]
            candidates = [self._tracks[key] for key in available if now - self._tracks[key].last_seen < 0.7]
            same = [track for track in candidates if track.handedness == hand.handedness]
            pool = same or candidates
            track = min(pool, key=lambda item: float(np.linalg.norm(item.wrist - wrist)), default=None)
            if track is None or float(np.linalg.norm(track.wrist - wrist)) > 0.28:
                track = _Track(self._next_id, hand.handedness, wrist.copy(), hand.normalized.copy(), now, GestureDetector())
                self._tracks[track.tracking_id] = track
                self._next_id += 1
            available.discard(track.tracking_id)
            alpha = 0.58
            track.smoothed += alpha * (hand.normalized - track.smoothed)
            track.wrist = wrist.copy()
            track.handedness = hand.handedness
            track.last_seen = now
            assigned.append((hand, track))
        self._tracks = {key: value for key, value in self._tracks.items() if now - value.last_seen < 0.8}
        hands: list[RoleTrackedHand] = []
        height, width = frame.shape[:2]
        for hand, track in assigned:
            smooth_pixels = np.column_stack((track.smoothed[:, 0] * width, track.smoothed[:, 1] * height)).astype(np.int32)
            gesture_state = track.detector.detect(hand)
            hands.append(RoleTrackedHand(
                track.tracking_id, hand.handedness, hand.confidence, hand.normalized, track.smoothed.copy(), smooth_pixels,
                tuple(smooth_pixels[8]), tuple(smooth_pixels[4]), gesture_state.fingers_up, gesture_state.gesture,
                gesture_state.pinch_distance, gesture_state.pinch_active, now,
            ))
        self._assign_roles(hands)
        return hands

    def _assign_roles(self, hands: list[RoleTrackedHand]) -> None:
        ids = {hand.tracking_id for hand in hands}
        preferred = [hand for hand in hands if hand.handedness.lower() == PREFERRED_DRAWING_HAND.lower()]
        drawing = [hand for hand in hands if hand.gesture is Gesture.DRAW]
        previous = [hand for hand in hands if hand.tracking_id == self.primary_id]
        if not self._roles_swapped or self.primary_id not in ids:
            chosen = preferred or drawing or previous or sorted(hands, key=lambda hand: hand.confidence, reverse=True)
            self.primary_id = chosen[0].tracking_id if chosen else None
        for hand in hands:
            hand.is_primary = hand.tracking_id == self.primary_id
            hand.is_secondary = not hand.is_primary

    def swap_roles(self, hands: list[RoleTrackedHand]) -> bool:
        if not ALLOW_HAND_ROLE_SWITCHING or len(hands) < 2:
            return False
        secondary = next((hand for hand in hands if hand.is_secondary), None)
        if secondary is None:
            return False
        self.primary_id = secondary.tracking_id
        self._roles_swapped = True
        self._assign_roles(hands)
        return True

    def close(self) -> None:
        self.tracker.close()

    def reset_tracking(self) -> None:
        self._tracks.clear()
        self.primary_id = None
        self._roles_swapped = False

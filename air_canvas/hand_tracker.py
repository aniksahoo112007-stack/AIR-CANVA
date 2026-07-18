"""MediaPipe Tasks based hand landmark tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .config import (
    MAX_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


class HandTrackerError(RuntimeError):
    """Raised when the hand tracking model cannot be initialized or used."""


@dataclass(frozen=True)
class TrackedHand:
    """Normalized and pixel-space landmarks for one detected hand."""

    normalized: np.ndarray
    pixels: np.ndarray
    handedness: str
    confidence: float = 0.0


class HandTracker:
    """Track one hand using MediaPipe's supported Hand Landmarker Tasks API."""

    def __init__(self, model_path: Path, max_hands: int = MAX_HANDS) -> None:
        if not model_path.is_file():
            raise HandTrackerError(
                f"MediaPipe model not found: {model_path}\n"
                "Download hand_landmarker.task as described in README.md or set "
                "AIR_CANVAS_MODEL to its path."
            )

        try:
            base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )
            self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise HandTrackerError(f"Could not load MediaPipe model: {exc}") from exc

        self._last_timestamp_ms = 0

    def detect_all(self, bgr_frame: np.ndarray) -> list[TrackedHand]:
        """Return every detected hand from the persistent MediaPipe model."""
        if bgr_frame.size == 0:
            return []

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        try:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            raise HandTrackerError(f"Hand tracking failed: {exc}") from exc

        if not result.hand_landmarks:
            return []

        height, width = bgr_frame.shape[:2]
        hands: list[TrackedHand] = []
        for index, landmarks in enumerate(result.hand_landmarks):
            normalized = np.array([(point.x, point.y, point.z) for point in landmarks], dtype=np.float32)
            pixels = np.column_stack((normalized[:, 0] * width, normalized[:, 1] * height)).astype(np.int32)
            category = result.handedness[index][0] if index < len(result.handedness) and result.handedness[index] else None
            hands.append(TrackedHand(normalized, pixels, category.category_name if category else "Unknown", float(category.score) if category else 0.0))
        return hands

    def detect(self, bgr_frame: np.ndarray) -> TrackedHand | None:
        """Compatibility helper returning the strongest detected hand."""
        hands = self.detect_all(bgr_frame)
        return max(hands, key=lambda hand: hand.confidence) if hands else None

    @staticmethod
    def draw_landmarks(frame: np.ndarray, hand: TrackedHand) -> None:
        """Draw a neon 21-point skeleton without legacy Solutions APIs."""
        connections = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        )
        for start, end in connections:
            cv2.line(frame, tuple(hand.pixels[start]), tuple(hand.pixels[end]), (30, 70, 82), 5, cv2.LINE_AA)
            cv2.line(frame, tuple(hand.pixels[start]), tuple(hand.pixels[end]), (70, 235, 255), 1, cv2.LINE_AA)
        for index, point in enumerate(hand.pixels):
            center = tuple(point)
            color = (80, 90, 255) if index == 8 else (100, 245, 255)
            cv2.circle(frame, center, 6 if index == 8 else 4, tuple(int(c * 0.28) for c in color), -1, cv2.LINE_AA)
            cv2.circle(frame, center, 3 if index == 8 else 2, color, -1, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe native resources."""
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

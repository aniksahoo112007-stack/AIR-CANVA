"""Lightweight gesture-specific holographic primitives."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .gesture_detector import Gesture


def draw_gesture_animation(frame: np.ndarray, pixels: np.ndarray, gesture: Gesture, color: tuple[int, int, int], now: float, progress: float = 0.0) -> None:
    index, wrist = tuple(pixels[8]), tuple(pixels[0])
    pulse = 0.5 + 0.5 * math.sin(now * 8)
    if gesture is Gesture.SELECT:
        cv2.circle(frame, index, int(13 + pulse * 5), color, 1, cv2.LINE_AA)
        cv2.line(frame, (index[0] - 7, index[1]), (index[0] + 7, index[1]), color, 1, cv2.LINE_AA)
        cv2.line(frame, (index[0], index[1] - 7), (index[0], index[1] + 7), color, 1, cv2.LINE_AA)
    elif gesture is Gesture.PINCH:
        cv2.circle(frame, index, int(8 + pulse * 14), color, 2, cv2.LINE_AA)
    elif gesture is Gesture.OPEN_PALM:
        radius = int(24 + 18 * ((now * 1.8) % 1.0))
        cv2.circle(frame, tuple(np.mean(pixels[[0, 5, 9, 13, 17]], axis=0).astype(int)), radius, color, 2, cv2.LINE_AA)
    elif gesture is Gesture.FIST:
        cv2.circle(frame, wrist, int(22 + pulse * 5), (30, 100, 255), 3, cv2.LINE_AA)
    elif gesture is Gesture.DRAW:
        cv2.circle(frame, index, int(8 - pulse * 3), color, 2, cv2.LINE_AA)

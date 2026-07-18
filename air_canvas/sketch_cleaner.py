"""Conservative local cleanup of drawing-only canvas layers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CleanupResult:
    image: np.ndarray
    mask: np.ndarray
    removed_pixels: int


class SketchCleaner:
    levels = ("light", "balanced", "strong")

    def clean(self, image: np.ndarray, mask: np.ndarray, intensity: str = "balanced") -> CleanupResult:
        level = intensity.lower()
        if level not in self.levels:
            raise ValueError(f"Unknown cleanup intensity: {intensity}")
        minimum_area = {"light": 3, "balanced": 8, "strong": 16}[level]
        binary = (mask > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        kept = np.zeros_like(binary)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
                kept[labels == label] = 255
        if level != "light":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kernel, iterations=1 if level == "balanced" else 2)
        cleaned_image = image.copy()
        cleaned_image[kept == 0] = 0
        if level == "strong":
            softened = cv2.GaussianBlur(cleaned_image, (3, 3), 0.6)
            cleaned_image[kept > 0] = softened[kept > 0]
        removed = int(np.count_nonzero(mask) - np.count_nonzero(kept))
        return CleanupResult(cleaned_image, kept.astype(np.uint8), max(0, removed))

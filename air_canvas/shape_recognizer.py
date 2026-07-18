"""Conservative OpenCV geometry recognition and corrected-shape rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ShapeRecognitionResult:
    shape_type: str
    confidence: float
    original_points: np.ndarray
    corrected_points: np.ndarray
    bounding_box: tuple[int, int, int, int]
    center: tuple[float, float]
    rotation: float = 0.0


class ShapeRecognizer:
    def recognize(self, points: list[tuple[int, int]]) -> ShapeRecognitionResult:
        original = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(original) < 6:
            return self._freehand(original)
        x, y, w, h = cv2.boundingRect(original.astype(np.int32))
        diagonal = max(math.hypot(w, h), 1.0)
        line = cv2.fitLine(original, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        direction, origin = line[:2], line[2:]
        distances = np.abs((original[:, 0] - origin[0]) * direction[1] - (original[:, 1] - origin[1]) * direction[0])
        line_confidence = max(0.0, 1.0 - float(np.mean(distances)) / max(3.0, diagonal * 0.035))
        endpoint_span = float(np.linalg.norm(original[-1] - original[0])) / diagonal
        if line_confidence >= 0.80 and endpoint_span >= 0.65:
            projections = (original - origin) @ direction
            corrected = np.stack((origin + direction * projections.min(), origin + direction * projections.max()))
            return self._result("line", min(0.99, line_confidence * 0.9 + endpoint_span * 0.1), original, corrected, (x, y, w, h))

        perimeter = float(cv2.arcLength(original.reshape(-1, 1, 2), False))
        closed = float(np.linalg.norm(original[-1] - original[0])) <= max(16.0, perimeter * 0.12)
        if not closed or perimeter < 30:
            arrow = self._recognize_arrow(original, (x, y, w, h), diagonal)
            return arrow if arrow is not None else self._freehand(original)
        contour = original.astype(np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(contour)
        hull_perimeter = float(cv2.arcLength(hull, True))
        approximation = cv2.approxPolyDP(hull, 0.035 * hull_perimeter, True).reshape(-1, 2)
        area = abs(float(cv2.contourArea(hull)))
        if area < diagonal * 2.0:
            return self._freehand(original)
        if len(approximation) == 3:
            confidence = self._coverage_confidence(original, approximation, diagonal)
            return self._result("triangle", confidence, original, approximation, (x, y, w, h))
        if len(approximation) == 4 and cv2.isContourConvex(approximation.astype(np.int32)):
            rect = cv2.minAreaRect(original)
            box = cv2.boxPoints(rect)
            side_a, side_b = rect[1]
            rectangularity = min(1.0, area / max(side_a * side_b, 1.0))
            confidence = min(0.98, 0.72 + rectangularity * 0.25)
            shape = "square" if min(side_a, side_b) / max(side_a, side_b, 1.0) >= 0.86 else "rectangle"
            if shape == "square":
                center = np.asarray(rect[0]); half = (side_a + side_b) / 4.0
                unit_x = (box[1] - box[0]) / max(np.linalg.norm(box[1] - box[0]), 1e-6)
                unit_y = np.array([-unit_x[1], unit_x[0]])
                box = np.stack((center-half*unit_x-half*unit_y, center+half*unit_x-half*unit_y,
                                center+half*unit_x+half*unit_y, center-half*unit_x+half*unit_y))
            return self._result(shape, confidence, original, box, (x, y, w, h), float(rect[2]))
        if len(original) >= 12:
            circularity = 4.0 * math.pi * area / max(hull_perimeter * hull_perimeter, 1.0)
            if len(original) >= 5:
                ellipse = cv2.fitEllipse(original.reshape(-1, 1, 2))
                center, axes, angle = ellipse
                aspect = min(axes) / max(max(axes), 1.0)
                confidence = min(0.97, 0.55 + circularity * 0.42)
                samples = cv2.ellipse2Poly(tuple(map(round, center)), tuple(max(1, round(v / 2)) for v in axes), round(angle), 0, 360, 4)
                shape = "circle" if aspect >= 0.88 else "ellipse"
                if confidence >= 0.72:
                    return self._result(shape, confidence, original, samples, (x, y, w, h), float(angle))
        return self._freehand(original)

    @staticmethod
    def _coverage_confidence(points: np.ndarray, polygon: np.ndarray, diagonal: float) -> float:
        distances = [abs(cv2.pointPolygonTest(polygon.astype(np.float32).reshape(-1, 1, 2), tuple(map(float, point)), True)) for point in points]
        return max(0.0, min(0.96, 1.0 - float(np.mean(distances)) / max(4.0, diagonal * 0.08)))

    def _recognize_arrow(self, points: np.ndarray, box: tuple[int, int, int, int], diagonal: float) -> ShapeRecognitionResult | None:
        approximation = cv2.approxPolyDP(points.reshape(-1, 1, 2), 0.045 * cv2.arcLength(points.reshape(-1, 1, 2), False), False).reshape(-1, 2)
        if 4 <= len(approximation) <= 7:
            shaft = approximation[-3] - approximation[0]
            if np.linalg.norm(shaft) >= diagonal * 0.55:
                return self._result("arrow", 0.80, points, approximation, box)
        return None

    def _freehand(self, points: np.ndarray) -> ShapeRecognitionResult:
        if len(points):
            x, y, w, h = cv2.boundingRect(points.astype(np.int32)); center = tuple(np.mean(points, axis=0))
        else:
            x = y = w = h = 0; center = (0.0, 0.0)
        return ShapeRecognitionResult("freehand", 0.0, points, points, (x, y, w, h), center)

    @staticmethod
    def _result(shape: str, confidence: float, original: np.ndarray, corrected: np.ndarray,
                box: tuple[int, int, int, int], rotation: float = 0.0) -> ShapeRecognitionResult:
        return ShapeRecognitionResult(shape, float(confidence), original, np.asarray(corrected, np.float32), box,
                                      tuple(np.mean(corrected, axis=0).astype(float)), rotation)


def render_corrected_shape(image: np.ndarray, mask: np.ndarray, result: ShapeRecognitionResult,
                           color: tuple[int, int, int], thickness: int) -> None:
    points = np.rint(result.corrected_points).astype(np.int32)
    if result.shape_type == "line":
        cv2.line(image, tuple(points[0]), tuple(points[-1]), color, thickness, cv2.LINE_AA)
        cv2.line(mask, tuple(points[0]), tuple(points[-1]), 255, thickness, cv2.LINE_AA)
    else:
        closed = result.shape_type != "arrow"
        cv2.polylines(image, [points.reshape(-1, 1, 2)], closed, color, thickness, cv2.LINE_AA)
        cv2.polylines(mask, [points.reshape(-1, 1, 2)], closed, 255, thickness, cv2.LINE_AA)

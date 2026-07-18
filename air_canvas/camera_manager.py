"""Validated OpenCV camera discovery, selection, switching, and recovery."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import time

import cv2
import numpy as np

from .config import (
    CAMERA_FPS, CAMERA_HEIGHT, CAMERA_RECONNECT_ATTEMPTS, CAMERA_RECONNECT_DELAY_SECONDS,
    CAMERA_SCAN_END, CAMERA_SCAN_START, CAMERA_WIDTH, DEFAULT_CAMERA_INDEX,
    PREFER_EXTERNAL_CAMERA,
    CAMERA_MIN_FRAME_DIFFERENCE, CAMERA_MIN_FRAME_STDDEV, CAMERA_MIN_MEAN_BRIGHTNESS,
    CAMERA_VALIDATION_FRAMES,
    CAMERA_WARMUP_SECONDS, CAMERA_WARMUP_FRAME_LIMIT, CAMERA_VALID_FRAME_REQUIREMENT,
)


@dataclass(frozen=True)
class CameraInfo:
    index: int
    backend: int
    backend_name: str
    width: int
    height: int
    fps: float
    frames_received: bool = True
    detected: bool = True
    opened: bool = True
    receiving_frames: bool = True
    usable: bool = True
    mean_brightness: float = 0.0
    frame_stddev: float = 0.0
    mean_frame_difference: float = 0.0

    @property
    def pixels(self) -> int:
        return self.width * self.height


class CameraManager:
    """Own exactly one active capture and never trust `isOpened()` alone."""

    def __init__(self) -> None:
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)
        self.capture: cv2.VideoCapture | None = None
        self.active_info: CameraInfo | None = None
        self.available_cameras: list[CameraInfo] = []
        self.detected_cameras: list[CameraInfo] = []
        self.last_frame: np.ndarray | None = None
        self.last_read_success = False
        self.last_successful_frame_time = 0.0
        self.consecutive_failures = 0

    @staticmethod
    def _backends() -> tuple[tuple[int, str], ...]:
        if sys.platform == "win32":
            return ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY"))
        return ((cv2.CAP_ANY, "ANY"),)

    def discover_cameras(self) -> list[CameraInfo]:
        discovered: list[CameraInfo] = []
        detected: list[CameraInfo] = []
        for index in range(CAMERA_SCAN_START, CAMERA_SCAN_END + 1):
            if self.active_info is not None and self.capture is not None and index == self.active_info.index:
                info = self.active_info
            else:
                info = self._probe(index)
            if info is None:
                print(f"[CAMERA] Index {index} unavailable")
            else:
                detected.append(info)
                if info.usable:
                    discovered.append(info)
                    print(f"[CAMERA] Index {index} available: {info.width}x{info.height}, backend {info.backend_name}, {info.fps:.1f} FPS")
                else:
                    print(f"[CAMERA] Index {index} detected but unusable: {info.width}x{info.height}, backend {info.backend_name}")
        self.available_cameras = discovered
        self.detected_cameras = detected
        return list(discovered)

    def _probe(self, index: int) -> CameraInfo | None:
        for backend, name in self._backends():
            capture, info, _ = self._open_validated(index, backend, name)
            if capture is not None:
                capture.release()
                return info
        return None

    def open_camera(self, index: int) -> bool:
        known = next((info for info in self.available_cameras if info.index == index), None)
        candidates = self._backends()
        if known:
            candidates = ((known.backend, known.backend_name),) + tuple(item for item in candidates if item[0] != known.backend)
        old_capture, old_info = self.capture, self.active_info
        print(f"[CAMERA] Switching active capture from {None if old_info is None else old_info.index} to {index}")
        for backend, name in candidates:
            capture, info, frame = self._open_validated(index, backend, name)
            if capture is None or info is None or frame is None:
                continue
            if old_capture is not None:
                old_capture.release()
            self.capture = capture
            self.active_info = info
            self.last_frame = frame.copy()
            self.last_read_success = True
            self.last_successful_frame_time = time.monotonic()
            self.consecutive_failures = 0
            print(f"[CAMERA] Selected index {index} using {name}")
            print(f"[CAMERA] Active capture is now index {index}")
            return True
        self.capture, self.active_info = old_capture, old_info
        return False

    def open_preferred_camera(self, override_index: int | None = None) -> bool:
        cameras = self.discover_cameras()
        if override_index is not None:
            return self.open_camera(override_index)
        if not cameras:
            return False
        ordered = self._preference_order(cameras)
        return any(self.open_camera(info.index) for info in ordered)

    @staticmethod
    def _preference_order(cameras: list[CameraInfo]) -> list[CameraInfo]:
        external = [info for info in cameras if info.index != DEFAULT_CAMERA_INDEX]
        if PREFER_EXTERNAL_CAMERA and external:
            ordered = sorted(external, key=lambda info: (info.pixels, info.fps, -info.index), reverse=True)
            ordered.extend(info for info in cameras if info not in ordered)
        else:
            ordered = sorted(cameras, key=lambda info: (info.index != DEFAULT_CAMERA_INDEX, -info.pixels))
        return ordered

    def preferred_camera_info(self) -> CameraInfo | None:
        ordered = self._preference_order(self.available_cameras)
        return ordered[0] if ordered else None

    def switch_camera(self, direction: int = 1) -> bool:
        cameras = self.discover_cameras()
        if not cameras:
            return False
        indexes = [info.index for info in cameras]
        current = self.active_info.index if self.active_info else None
        start = indexes.index(current) if current in indexes else (-1 if direction > 0 else 0)
        for offset in range(1, len(indexes) + 1):
            candidate = indexes[(start + direction * offset) % len(indexes)]
            if candidate != current and self.open_camera(candidate):
                return True
        return False

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        if self.capture is None:
            self.last_read_success = False
            self.consecutive_failures += 1
            return False, None
        ok, frame = self.capture.read()
        valid = bool(ok and frame is not None and frame.size > 0 and frame.shape[0] >= 120 and frame.shape[1] >= 160)
        self.last_read_success = valid
        if valid:
            self.last_frame = frame.copy()
            self.last_successful_frame_time = time.monotonic()
            self.consecutive_failures = 0
            return True, frame
        self.consecutive_failures += 1
        return False, None

    def test_camera(self, index: int) -> tuple[CameraInfo | None, np.ndarray | None]:
        if self.active_info is not None and index == self.active_info.index and self.last_frame is not None:
            return self.active_info, self.last_frame.copy()
        print(f"[CAMERA] Opening temporary test capture for index {index}")
        try:
            for backend, name in self._backends():
                capture, info, frame = self._open_validated(index, backend, name)
                if capture is None:
                    continue
                capture.release()
                return info, None if frame is None else frame.copy()
            return None, None
        finally:
            print(f"[CAMERA] Releasing temporary test capture for index {index}")

    def _open_validated(self, index: int, backend: int, name: str) -> tuple[cv2.VideoCapture | None, CameraInfo | None, np.ndarray | None]:
        """Open one candidate at a time; caller owns a successful capture."""
        formats: tuple[tuple[int, int] | None, ...] = (None, (CAMERA_WIDTH, CAMERA_HEIGHT), (640, 480))
        for requested in formats:
            capture = cv2.VideoCapture(index, backend)
            keep_capture = False
            try:
                if not capture.isOpened():
                    continue
                if requested is not None:
                    self._apply_requested_format(capture, *requested)
                frames: list[np.ndarray] = []
                deadline = time.monotonic() + CAMERA_WARMUP_SECONDS
                for _ in range(CAMERA_WARMUP_FRAME_LIMIT):
                    ok, frame = capture.read()
                    if ok and frame is not None and frame.size > 0 and frame.shape[0] >= 120 and frame.shape[1] >= 160:
                        frames.append(frame)
                        if len(frames) >= CAMERA_VALID_FRAME_REQUIREMENT:
                            break
                    if time.monotonic() >= deadline:
                        break
                if len(frames) >= CAMERA_VALID_FRAME_REQUIREMENT:
                    metrics = self._frame_metrics(frames)
                    height, width = frames[-1].shape[:2]
                    fps = float(capture.get(cv2.CAP_PROP_FPS))
                    info = CameraInfo(index, backend, name, width, height, fps if np.isfinite(fps) and fps > 0 else 0.0, True, True, True, True, metrics[3], *metrics[:3])
                    print(f"[CAMERA] Index {index} {name} delivering frames at {width}x{height}" + (" (quality warning)" if not info.usable else ""))
                    keep_capture = True
                    return capture, info, frames[-1]
                print(f"[CAMERA] Index {index} {name} opened but no frames")
            finally:
                if not keep_capture:
                    capture.release()
        return None, None, None

    def switch_to(self, index: int) -> bool:
        return self.open_camera(index)

    def get_camera_status(self) -> dict[str, object]:
        return {
            "index": None if self.active_info is None else self.active_info.index,
            "backend": "NONE" if self.active_info is None else self.active_info.backend_name,
            "read_success": self.last_read_success,
            "frame_shape": None if self.last_frame is None else self.last_frame.shape,
            "mean_brightness": 0.0 if self.last_frame is None else float(np.mean(self.last_frame)),
            "frame_stddev": 0.0 if self.last_frame is None else float(np.std(self.last_frame)),
            "consecutive_failures": self.consecutive_failures,
            "last_successful_frame_time": self.last_successful_frame_time,
        }

    def recover_camera(self) -> bool:
        failed_index = self.active_info.index if self.active_info else None
        if failed_index is not None:
            self.release()
            for attempt in range(CAMERA_RECONNECT_ATTEMPTS):
                if attempt:
                    time.sleep(CAMERA_RECONNECT_DELAY_SECONDS)
                if self.open_camera(failed_index):
                    return True
        cameras = self.discover_cameras()
        fallbacks = sorted(cameras, key=lambda info: (info.index != DEFAULT_CAMERA_INDEX, info.pixels), reverse=False)
        return any(info.index != failed_index and self.open_camera(info.index) for info in fallbacks)

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.active_info = None

    @staticmethod
    def _frame_metrics(frames: list[np.ndarray]) -> tuple[float, float, float, bool]:
        if not frames:
            return 0.0, 0.0, 0.0, False
        samples = [cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) for frame in frames]
        mean = float(np.mean(samples[-1]))
        stddev = float(np.std(samples[-1]))
        differences = [float(np.mean(cv2.absdiff(samples[index - 1], samples[index]))) for index in range(1, len(samples))]
        difference = float(np.mean(differences)) if differences else 0.0
        usable = not (mean < CAMERA_MIN_MEAN_BRIGHTNESS and stddev < CAMERA_MIN_FRAME_STDDEV) and difference >= CAMERA_MIN_FRAME_DIFFERENCE
        return mean, stddev, difference, usable

    @staticmethod
    def _apply_requested_format(capture: cv2.VideoCapture, width: int, height: int) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

"""Short opt-in camera/model performance probe; not part of unit discovery."""

from __future__ import annotations

import statistics
import time

import cv2

from air_canvas.camera_manager import CameraManager
from air_canvas.config import MODEL_PATH, MIRROR_CAMERA
from air_canvas.hand_tracker import HandTracker


def run(frame_limit: int = 90) -> None:
    camera = CameraManager()
    if not camera.open_preferred_camera():
        raise RuntimeError("Camera unavailable")
    selected = camera.active_info
    tracker = HandTracker(MODEL_PATH)
    durations: list[float] = []
    maximum_hands = 0
    try:
        for _ in range(frame_limit):
            ok, frame = camera.read_frame()
            if not ok:
                continue
            started = time.perf_counter()
            hands = tracker.detect_all(cv2.flip(frame, 1) if MIRROR_CAMERA else frame)
            durations.append(time.perf_counter() - started)
            maximum_hands = max(maximum_hands, len(hands))
    finally:
        tracker.close()
        camera.release()
    if not durations:
        raise RuntimeError("Camera returned no frames")
    assert selected is not None
    print(f"CAMERA_PROBE index={selected.index} backend={selected.backend_name} resolution={selected.width}x{selected.height} frames={len(durations)} tracking_fps={1 / statistics.mean(durations):.1f} max_hands={maximum_hands}")


if __name__ == "__main__":
    run()

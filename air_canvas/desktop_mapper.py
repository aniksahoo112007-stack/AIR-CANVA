"""Calibrated normalized camera-to-monitor coordinate mapping."""

from __future__ import annotations

import json
from pathlib import Path

from .config import (DESKTOP_CURSOR_MARGIN_X, DESKTOP_CURSOR_MARGIN_Y, DESKTOP_CURSOR_MIRROR_X,
                     DESKTOP_CURSOR_SENSITIVITY_X, DESKTOP_CURSOR_SENSITIVITY_Y, ROOT_DIR)
from .monitor_manager import MonitorBounds


class DesktopMapper:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or ROOT_DIR / "desktop_calibration.json"
        self.data: dict[str, dict[str, float]] = {}
        self.camera_index = 0; self.monitor_index = 0
        self.load()

    def set_context(self, camera_index: int, monitor_index: int) -> None:
        self.camera_index, self.monitor_index = camera_index, monitor_index

    @property
    def key(self) -> str: return f"camera:{self.camera_index}/monitor:{self.monitor_index}"

    def camera_to_desktop(self, normalized_point: tuple[float, float], monitor: MonitorBounds) -> tuple[int, int]:
        calibration = self.data.get(self.key, {})
        min_x = calibration.get("min_x", DESKTOP_CURSOR_MARGIN_X); max_x = calibration.get("max_x", 1.0-DESKTOP_CURSOR_MARGIN_X)
        min_y = calibration.get("min_y", DESKTOP_CURSOR_MARGIN_Y); max_y = calibration.get("max_y", 1.0-DESKTOP_CURSOR_MARGIN_Y)
        x, y = normalized_point
        if DESKTOP_CURSOR_MIRROR_X: x = 1.0 - x
        x = 0.5 + (x-0.5)*DESKTOP_CURSOR_SENSITIVITY_X; y = 0.5 + (y-0.5)*DESKTOP_CURSOR_SENSITIVITY_Y
        nx = max(0.0, min(1.0, (x-min_x)/max(max_x-min_x, 1e-6)))
        ny = max(0.0, min(1.0, (y-min_y)/max(max_y-min_y, 1e-6)))
        return monitor.left + round(nx*(monitor.width-1)), monitor.top + round(ny*(monitor.height-1))

    def calibrate(self, points: list[tuple[float, float]]) -> None:
        if len(points) != 4: raise ValueError("Calibration requires four corner samples")
        xs = [1.0-p[0] if DESKTOP_CURSOR_MIRROR_X else p[0] for p in points]
        ys = [p[1] for p in points]
        self.data[self.key] = {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}
        self.save()

    def reset_calibration(self) -> None:
        self.data.pop(self.key, None); self.save()

    def load(self) -> None:
        try:
            if self.path.is_file(): self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError): self.data = {}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")

"""Transactional, bounded undo and redo history for drawing layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_HISTORY_STATES


@dataclass(frozen=True)
class CanvasSnapshot:
    """Only the persistent drawing layers; webcam frames are never retained."""

    image: np.ndarray
    mask: np.ndarray


class HistoryManager:
    """Record exactly one history entry for each meaningful completed action."""

    def __init__(self, limit: int = MAX_HISTORY_STATES) -> None:
        self.limit = max(1, limit)
        self._undo: list[CanvasSnapshot] = []
        self._redo: list[CanvasSnapshot] = []
        self._pending: CanvasSnapshot | None = None

    @staticmethod
    def snapshot(image: np.ndarray, mask: np.ndarray) -> CanvasSnapshot:
        return CanvasSnapshot(image.copy(), mask.copy())

    @staticmethod
    def _matches(snapshot: CanvasSnapshot, image: np.ndarray, mask: np.ndarray) -> bool:
        return np.array_equal(snapshot.mask, mask) and np.array_equal(snapshot.image, image)

    def begin_stroke(self, image: np.ndarray, mask: np.ndarray) -> None:
        """Capture the state before a stroke/action, once only."""
        if self._pending is None:
            self._pending = self.snapshot(image, mask)

    def commit_stroke(self, image: np.ndarray, mask: np.ndarray) -> bool:
        """Commit a pending state if the canvas meaningfully changed."""
        pending, self._pending = self._pending, None
        if pending is None or self._matches(pending, image, mask):
            return False
        if not self._undo or not self._matches(self._undo[-1], pending.image, pending.mask):
            self._undo.append(pending)
            if len(self._undo) > self.limit:
                self._undo.pop(0)
        self.clear_redo()
        return True

    def cancel_stroke(self) -> None:
        self._pending = None

    def undo(self, image: np.ndarray, mask: np.ndarray) -> CanvasSnapshot | None:
        self.cancel_stroke()
        if not self._undo:
            return None
        current = self.snapshot(image, mask)
        target = self._undo.pop()
        if not self._redo or not self._matches(self._redo[-1], current.image, current.mask):
            self._redo.append(current)
        return target

    def redo(self, image: np.ndarray, mask: np.ndarray) -> CanvasSnapshot | None:
        self.cancel_stroke()
        if not self._redo:
            return None
        current = self.snapshot(image, mask)
        target = self._redo.pop()
        if not self._undo or not self._matches(self._undo[-1], current.image, current.mask):
            self._undo.append(current)
            if len(self._undo) > self.limit:
                self._undo.pop(0)
        return target

    def clear_redo(self) -> None:
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def current_position(self) -> int:
        return len(self._undo)

    @property
    def total_states(self) -> int:
        return len(self._undo) + len(self._redo)

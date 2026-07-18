"""Future AI transform provider contract; no implementation or credentials."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AITransformProvider(ABC):
    @abstractmethod
    def transform(self, image: np.ndarray, prompt: str, style: str) -> np.ndarray:
        raise NotImplementedError

"""Bounded per-hand particles with adaptive density."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .config import MAX_PARTICLES_PER_HAND


@dataclass
class HandParticle:
    hand_id: int
    origin: np.ndarray
    velocity: np.ndarray
    born: float
    lifetime: float
    color: tuple[int, int, int]


class HandParticleSystem:
    def __init__(self) -> None:
        self.particles: list[HandParticle] = []
        self.rng = np.random.default_rng()

    def emit(self, hand_id: int, point: tuple[int, int], now: float, color: tuple[int, int, int], count: int, density: float = 1.0) -> None:
        current = sum(item.hand_id == hand_id for item in self.particles)
        count = min(int(count * density), MAX_PARTICLES_PER_HAND - current)
        for _ in range(max(0, count)):
            angle, speed = self.rng.uniform(0, math.tau), self.rng.uniform(18, 85)
            velocity = np.array((math.cos(angle) * speed, math.sin(angle) * speed), dtype=np.float32)
            self.particles.append(HandParticle(hand_id, np.array(point, dtype=np.float32), velocity, now, self.rng.uniform(0.25, 0.65), color))

    def draw(self, frame: np.ndarray, now: float) -> None:
        alive: list[HandParticle] = []
        for item in self.particles:
            age = now - item.born
            if age >= item.lifetime:
                continue
            alive.append(item)
            alpha = 1.0 - age / item.lifetime
            point = tuple(np.rint(item.origin + item.velocity * age).astype(int))
            cv2.circle(frame, point, max(1, int(3 * alpha)), tuple(int(c * alpha) for c in item.color), -1, cv2.LINE_AA)
        self.particles = alive

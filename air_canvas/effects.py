"""Lightweight cursor, trail and particle effects."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .config import MAX_FINGER_TRAIL, MAX_PARTICLES


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    born_at: float
    lifetime: float
    color: tuple[int, int, int]


class EffectsRenderer:
    """Maintain small bounded effect collections without full-frame blur."""

    def __init__(self) -> None:
        self.trail: list[tuple[tuple[int, int], float, tuple[int, int, int]]] = []
        self.particles: list[Particle] = []
        self._rng = np.random.default_rng()

    def add_cursor_point(self, point: tuple[int, int], now: float, color: tuple[int, int, int]) -> None:
        self.trail.append((point, now, color))
        self.trail = self.trail[-MAX_FINGER_TRAIL:]

    def burst(self, point: tuple[int, int], now: float, color: tuple[int, int, int], count: int = 24) -> None:
        for _ in range(min(count, MAX_PARTICLES - len(self.particles))):
            angle = self._rng.uniform(0, math.tau)
            speed = self._rng.uniform(45.0, 150.0)
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            self.particles.append(Particle(np.array(point, dtype=np.float32), velocity, now, self._rng.uniform(0.45, 0.9), color))

    def draw_trail(self, frame: np.ndarray, now: float) -> None:
        self.trail = [item for item in self.trail if now - item[1] < 0.42]
        for index, (point, born, color) in enumerate(self.trail):
            alpha = max(0.0, 1.0 - (now - born) / 0.42)
            radius = max(1, int(2 + 4 * alpha * (index + 1) / max(len(self.trail), 1)))
            dim = tuple(int(channel * alpha) for channel in color)
            cv2.circle(frame, point, radius, dim, -1, cv2.LINE_AA)

    def draw_cursor(self, frame: np.ndarray, point: tuple[int, int], color: tuple[int, int, int], drawing: bool, now: float) -> None:
        pulse = 2.0 * math.sin(now * 8.0)
        radius = int((18 if drawing else 13) + pulse)
        cv2.circle(frame, point, radius + 5, tuple(int(c * 0.24) for c in color), 3, cv2.LINE_AA)
        cv2.circle(frame, point, radius, color, 2, cv2.LINE_AA)
        cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_particles(self, frame: np.ndarray, now: float) -> None:
        alive: list[Particle] = []
        for particle in self.particles:
            age = now - particle.born_at
            if age >= particle.lifetime:
                continue
            alive.append(particle)
            position = particle.position + particle.velocity * age
            alpha = 1.0 - age / particle.lifetime
            color = tuple(int(channel * alpha) for channel in particle.color)
            cv2.circle(frame, tuple(position.astype(int)), max(1, int(3 * alpha)), color, -1, cv2.LINE_AA)
        self.particles = alive

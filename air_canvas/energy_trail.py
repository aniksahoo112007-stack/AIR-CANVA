"""Warm fire particles and curved landmark-anchored motion trails."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .config import MAX_ENERGY_TRAIL_POINTS, MAX_FIRE_PARTICLES, MAX_SPARK_STREAKS
from .gesture_detector import Gesture


WARM_PALETTE = ((180, 245, 255), (20, 170, 255), (0, 105, 235), (20, 200, 255))


@dataclass
class EnergyParticle:
    position: np.ndarray
    velocity: np.ndarray
    born: float
    lifetime: float
    size: float
    color: tuple[int, int, int]
    streak: bool


@dataclass
class TrailPoint:
    position: tuple[int, int]
    timestamp: float
    velocity: float


class EnergyTrailSystem:
    """Independent particle and trail collections for one tracking ID."""

    def __init__(self, tracking_id: int) -> None:
        self.tracking_id = tracking_id
        self.particles: list[EnergyParticle] = []
        self.trails: dict[str, deque[TrailPoint]] = defaultdict(lambda: deque(maxlen=MAX_ENERGY_TRAIL_POINTS))
        self.rng = np.random.default_rng(tracking_id)
        self.previous: dict[str, tuple[np.ndarray, float]] = {}
        self._pinch_latched = False

    def update(self, anchors: dict[str, tuple[int, int]], gesture: Gesture, confidence: float, now: float, intensity: float, particle_limit: int) -> None:
        velocities: dict[str, float] = {}
        for name, point in anchors.items():
            array = np.asarray(point, dtype=np.float32)
            previous = self.previous.get(name)
            velocity = 0.0 if previous is None else float(np.linalg.norm(array - previous[0]) / max(1e-3, now - previous[1]))
            velocities[name] = velocity
            self.trails[name].append(TrailPoint(point, now, velocity))
            self.previous[name] = (array, now)
        active = "index" if gesture in {Gesture.DRAW, Gesture.SELECT, Gesture.PINCH} else "palm"
        speed = velocities.get(active, 0.0)
        density = min(7, max(1, int((1 + speed / 180.0) * intensity * confidence)))
        if gesture in {Gesture.FIST, Gesture.PINCH}:
            density = 0
        self._spawn(anchors[active], now, density, particle_limit, converge=None)
        if gesture is Gesture.DRAW and speed > 80:
            self._spawn(anchors["index"], now, 2, particle_limit, converge=None, sparks=True)
        if gesture is Gesture.PINCH and not self._pinch_latched:
            self._pinch_latched = True
        elif gesture is not Gesture.PINCH:
            self._pinch_latched = False
        alive: list[EnergyParticle] = []
        for particle in self.particles:
            age = now - particle.born
            if age >= particle.lifetime:
                continue
            dt = min(0.05, max(0.0, now - max(particle.born, now - 0.05)))
            particle.velocity[1] -= 26.0 * dt
            particle.velocity *= 0.985
            alive.append(particle)
        self.particles = alive[-particle_limit:]

    def reset(self) -> None:
        self.particles.clear();self.trails.clear();self.previous.clear();self._pinch_latched=False

    def _spawn(self, point: tuple[int, int], now: float, count: int, limit: int, converge: np.ndarray | None, sparks: bool = False) -> None:
        room = max(0, min(limit, MAX_FIRE_PARTICLES) - len(self.particles))
        streaks = sum(item.streak for item in self.particles)
        for _ in range(min(count, room)):
            angle = self.rng.uniform(0, math.tau)
            speed = self.rng.uniform(16, 105 if sparks else 65)
            spawn = np.asarray(point, dtype=np.float32) + self.rng.normal(0, 3, 2)
            velocity = np.array((math.cos(angle) * speed, math.sin(angle) * speed - 22), dtype=np.float32)
            if converge is not None:
                radius = self.rng.uniform(22, 58)
                spawn = converge + np.array((math.cos(angle) * radius, math.sin(angle) * radius), dtype=np.float32)
                direction = converge - spawn
                velocity = direction / max(float(np.linalg.norm(direction)), 1.0) * self.rng.uniform(90, 180)
            streak = sparks and streaks < MAX_SPARK_STREAKS and self.rng.random() < 0.45
            streaks += int(streak)
            self.particles.append(EnergyParticle(
                spawn, velocity, now,
                float(self.rng.uniform(0.22, 0.72)), float(self.rng.uniform(1.2, 3.8)),
                WARM_PALETTE[int(self.rng.integers(0, len(WARM_PALETTE)))], streak,
            ))

    def burst(self, point: tuple[int, int], now: float, count: int = 20) -> None:
        self._spawn(point, now, count, MAX_FIRE_PARTICLES, converge=None, sparks=True)

    def render(self, sharp: np.ndarray, glow: np.ndarray, now: float, accent: tuple[int, int, int], trail_intensity: float, offset: tuple[int, int] = (0, 0), show_trails: bool = True, show_particles: bool = True) -> None:
        for name, trail in self.trails.items():
            points = [item for item in trail if now - item.timestamp < 0.55]
            self.trails[name] = deque(points, maxlen=MAX_ENERGY_TRAIL_POINTS)
            if len(points) < 2 or not show_trails or name != "index":
                continue
            for index in range(1, len(points)):
                age = now - points[index].timestamp
                alpha = max(0.0, (1.0 - age / 0.55) * trail_intensity)
                color = WARM_PALETTE[1] if name == "index" else accent
                dim = tuple(int(c * alpha) for c in color)
                start = (points[index - 1].position[0] - offset[0], points[index - 1].position[1] - offset[1])
                end = (points[index].position[0] - offset[0], points[index].position[1] - offset[1])
                cv2.line(glow, start, end, tuple(int(c * 0.3) for c in dim), max(2, int(7 * alpha)), cv2.LINE_AA)
                cv2.line(sharp, start, end, dim, max(1, int(2 * alpha)), cv2.LINE_AA)
        for particle in self.particles if show_particles else ():
            age = now - particle.born
            alpha = max(0.0, 1.0 - age / particle.lifetime)
            position = particle.position + particle.velocity * age
            point_global = tuple(np.rint(position).astype(int))
            point = (point_global[0] - offset[0], point_global[1] - offset[1])
            color = tuple(int(c * alpha) for c in particle.color)
            radius = max(1, int(particle.size * (0.55 + alpha)))
            cv2.circle(glow, point, radius * 3, tuple(int(c * 0.3) for c in color), -1, cv2.LINE_AA)
            if particle.streak:
                tail_global = tuple(np.rint(position - particle.velocity * 0.045).astype(int))
                tail = (tail_global[0] - offset[0], tail_global[1] - offset[1])
                cv2.line(sharp, tail, point, color, 1, cv2.LINE_AA)
            cv2.circle(sharp, point, radius, color, -1, cv2.LINE_AA)

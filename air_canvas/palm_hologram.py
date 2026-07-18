"""Perspective-scaled palm rings, HUD ticks, orbit points and local glow."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import ENABLE_ORBIT_PARTICLES, ENABLE_PALM_CORE, MAX_ORBIT_PARTICLES, PALM_RING_BASE_RADIUS, PALM_RING_COUNT
from .gesture_detector import Gesture
from .hand_effect_state import HandEffectState


PALM_INDICES = (0, 5, 9, 13, 17)


def palm_geometry(pixels: np.ndarray) -> tuple[tuple[int, int], float]:
    center = tuple(np.mean(pixels[list(PALM_INDICES)], axis=0).astype(int))
    scale = float(np.linalg.norm(pixels[0].astype(float) - pixels[9].astype(float)))
    return center, max(18.0, min(95.0, scale))


def draw_rotating_arc(layer: np.ndarray, center: tuple[int, int], radius: int, angle: float, span: int, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.ellipse(layer, center, (radius, max(4, int(radius * 0.64))), -8, angle, angle + span, color, thickness, cv2.LINE_AA)


def draw_segmented_ring(layer: np.ndarray, center: tuple[int, int], radius: int, angle: float, color: tuple[int, int, int], segments: int = 8) -> None:
    step = 360.0 / segments
    for index in range(segments):
        draw_rotating_arc(layer, center, radius, angle + index * step, int(step * 0.58), color)


def draw_hud_ticks(layer: np.ndarray, center: tuple[int, int], radius: int, angle: float, color: tuple[int, int, int]) -> None:
    for tick in range(0, 360, 30):
        radians = math.radians(tick + angle)
        inner = (int(center[0] + math.cos(radians) * (radius - 3)), int(center[1] + math.sin(radians) * (radius - 3) * 0.64))
        outer = (int(center[0] + math.cos(radians) * (radius + (5 if tick % 90 == 0 else 2))), int(center[1] + math.sin(radians) * (radius + (5 if tick % 90 == 0 else 2)) * 0.64))
        cv2.line(layer, inner, outer, color, 1, cv2.LINE_AA)


def draw_orbit_particles(layer: np.ndarray, center: tuple[int, int], radius: int, now: float, color: tuple[int, int, int]) -> None:
    if not ENABLE_ORBIT_PARTICLES:
        return
    count = min(6, MAX_ORBIT_PARTICLES)
    for index in range(count):
        angle = now * (0.8 + index * 0.13) * (-1 if index % 2 else 1) + index * math.tau / count
        point = (int(center[0] + math.cos(angle) * radius), int(center[1] + math.sin(angle) * radius * 0.64))
        cv2.circle(layer, point, 1 + index % 2, color, -1, cv2.LINE_AA)


def draw_radial_glow(layer: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int], strength: float) -> None:
    for fraction, alpha in ((1.0, 0.08), (0.62, 0.15), (0.3, 0.28)):
        cv2.circle(layer, center, max(2, int(radius * fraction)), tuple(int(c * alpha * strength) for c in color), -1, cv2.LINE_AA)


def render_palm_hologram(sharp: np.ndarray, glow: np.ndarray, pixels: np.ndarray, state: HandEffectState, color: tuple[int, int, int], now: float, quality: str) -> tuple[tuple[int, int], int]:
    center, hand_scale = palm_geometry(pixels)
    base = int(max(PALM_RING_BASE_RADIUS * 0.55, hand_scale * 0.52) * state.ring_expansion)
    faded = tuple(int(c * state.fade) for c in color)
    draw_radial_glow(glow, center, base, color, 1.0 + state.pulse_strength)
    ring_count = 2 if quality == "low" else 3 if quality == "medium" else PALM_RING_COUNT
    radii = (1.0, 0.78, 0.56, 1.2, 0.4)
    for index in range(ring_count):
        radius = max(6, int(base * radii[index]))
        draw_segmented_ring(sharp, center, radius, state.rotation_angles[index], faded, 7 + index * 2)
        draw_segmented_ring(glow, center, radius, state.rotation_angles[index], tuple(int(c * 0.28) for c in faded), 7 + index * 2)
    if state.current_gesture is Gesture.OPEN_PALM:
        draw_hud_ticks(sharp, center, int(base * 1.12), state.rotation_angles[1], (230, 240, 255))
        sweep_y = int(center[1] - base * 0.6 + ((now * 0.9) % 1.0) * base * 1.2)
        cv2.line(sharp, (center[0] - base, sweep_y), (center[0] + base, sweep_y), faded, 1, cv2.LINE_AA)
        cv2.ellipse(sharp, center, (int(base * 1.25), int(base * 0.82)), -8, -90, -90 + int(360 * state.clear_progress), (180, 225, 255), 2, cv2.LINE_AA)
    draw_orbit_particles(sharp, center, int(base * 1.08), now, faded)
    if ENABLE_PALM_CORE:
        core_radius = 3 + int(4 * state.pulse_strength + 5 * state.pinch_flash_timer)
        cv2.circle(glow, center, core_radius * 3, tuple(int(c * 0.3) for c in faded), -1, cv2.LINE_AA)
        cv2.circle(sharp, center, core_radius, (220, 245, 255), -1, cv2.LINE_AA)
    if state.current_gesture is Gesture.FIST:
        cv2.circle(sharp, center, int(base * (0.75 + state.pulse_strength * 0.15)), (20, 110, 255), 2, cv2.LINE_AA)
    return center, base

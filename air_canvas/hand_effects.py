"""Gesture-reactive arcane holograms rendered in bounded hand regions."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .config import (
    ACTIVE_FINGERTIP_MARKER_RADIUS, ENABLE_FIRE_TRAIL, ENABLE_HAND_CONNECTION_ARC,
    FX_DEGRADE_BELOW_FPS, FX_RECOVER_ABOVE_FPS, FX_RECOVERY_HOLD_SECONDS, GLOW_ALPHA,
    GLOW_BLUR_KERNEL, HAND_EFFECT_QUALITY, MAX_FIRE_PARTICLES, PASSIVE_FINGERTIP_MARKER_RADIUS,
    PRIMARY_EFFECT_COLOR, SECONDARY_EFFECT_COLOR, SHOW_ACTIVE_FINGERTIP_MARKER,
    SHOW_ALL_FINGERTIP_MARKERS, SHOW_BASE_SKELETON, SHOW_GESTURE_LABEL, SHOW_HAND_ROLE_LABEL,
    SHOW_TECHNICAL_LANDMARKS,
)
from .dual_hand_tracker import RoleTrackedHand
from .energy_trail import EnergyTrailSystem
from .gesture_detector import Gesture
from .hand_effect_state import HandEffectState
from .palm_hologram import PALM_INDICES, palm_geometry, render_palm_hologram


CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
HUD_LANDMARKS = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)


class HandEffectsRenderer:
    """Own all independent effect state; main only supplies tracked hands and time."""

    def __init__(self) -> None:
        self.quality = HAND_EFFECT_QUALITY.lower()
        self.states: dict[int, HandEffectState] = {}
        self.energy: dict[int, EnergyTrailSystem] = {}
        self.last_hands: dict[int, RoleTrackedHand] = {}
        self.performance_scale = 1.0
        self.particles_enabled = True
        self.trails_enabled = True
        self._low_fps_since: float | None = None
        self._lost_spawned: set[int] = set()
        self._degraded = False
        self._recovery_since: float | None = None

    def update(self, hands: list[RoleTrackedHand], now: float, fps: float = 30.0) -> None:
        visible_ids = {hand.tracking_id for hand in hands}
        for hand in hands:
            state = self.states.setdefault(hand.tracking_id, HandEffectState(hand.tracking_id))
            center, _ = palm_geometry(hand.pixels)
            stale = state.last_update > 0 and now-state.last_update > .45
            state.update(hand.gesture, center, now, True)
            system = self.energy.setdefault(hand.tracking_id, EnergyTrailSystem(hand.tracking_id))
            if stale:system.reset()
            anchors = {"index": hand.index_tip, "thumb": hand.thumb_tip, "wrist": tuple(hand.pixels[0]), "palm": center}
            limit = self._particle_limit()
            if ENABLE_FIRE_TRAIL and self.particles_enabled and not self._degraded:
                system.update(anchors, hand.gesture, hand.confidence, now, state.particle_intensity, limit)
                if hand.gesture is Gesture.OPEN_PALM and state.clear_progress >= 1.0 and not state.clear_burst_emitted:
                    system.burst(center, now, 32)
                    state.clear_burst_emitted = True
            self.last_hands[hand.tracking_id] = hand
            self._lost_spawned.discard(hand.tracking_id)
        for tracking_id, state in list(self.states.items()):
            if tracking_id not in visible_ids:
                state.update(state.current_gesture, tuple(state.last_position.astype(int)) if state.last_position is not None else (0, 0), now, False)
                if now - state.last_seen > 0.65:
                    self.states.pop(tracking_id, None)
                    self.energy.pop(tracking_id, None)
                    self.last_hands.pop(tracking_id, None)
                    self._lost_spawned.discard(tracking_id)
        self.adapt_to_fps(fps, now)

    def render(self, frame: np.ndarray, hands: list[RoleTrackedHand], now: float, hovered_target: tuple[int, int] | None = None) -> None:
        visible = {hand.tracking_id: hand for hand in hands}
        render_hands = list(hands)
        render_hands.extend(hand for tracking_id, hand in self.last_hands.items() if tracking_id not in visible and tracking_id in self.states)
        for hand in render_hands:
            state = self.states.get(hand.tracking_id)
            if state is None or state.fade < 0.02:
                continue
            self._render_hand_region(frame, hand, state, now, hovered_target if hand.is_primary else None)
        self._render_connection_arc(frame, hands, now)

    def _render_hand_region(self, frame: np.ndarray, hand: RoleTrackedHand, state: HandEffectState, now: float, hovered_target: tuple[int, int] | None) -> None:
        height, width = frame.shape[:2]
        minimum = hand.pixels.min(axis=0) - 75
        maximum = hand.pixels.max(axis=0) + 75
        x1, y1 = max(0, int(minimum[0])), max(0, int(minimum[1]))
        x2, y2 = min(width, int(maximum[0])), min(height, int(maximum[1]))
        if x2 <= x1 or y2 <= y1:
            return
        local_pixels = hand.pixels - np.array((x1, y1))
        sharp = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
        glow = np.zeros_like(sharp)
        color = PRIMARY_EFFECT_COLOR if hand.is_primary else SECONDARY_EFFECT_COLOR
        center, base_radius = render_palm_hologram(sharp, glow, local_pixels, state, color, now, self.quality, hand.is_primary)
        system = self.energy.get(hand.tracking_id)
        if system is not None and (self.particles_enabled or self.trails_enabled):
            show_particles=self.particles_enabled and not self._degraded and self.quality in {"high","ultra"}
            system.render(sharp, glow, now, color, state.trail_intensity, offset=(x1, y1), show_trails=self.trails_enabled and hand.gesture is Gesture.DRAW, show_particles=show_particles)
        self._draw_technical_hand(sharp, local_pixels, state, color, now, hand.gesture, hand.is_primary)
        if hand.gesture is Gesture.SELECT:
            for index in (8, 12):
                point = tuple(local_pixels[index])
                cv2.circle(sharp, point, int(9 + 3 * math.sin(now * 9 + index)), (235, 245, 255), 1, cv2.LINE_AA)
            if hovered_target is not None:
                target = (hovered_target[0] - x1, hovered_target[1] - y1)
                cv2.line(sharp, tuple(local_pixels[8]), target, tuple(int(c * 0.72) for c in color), 1, cv2.LINE_AA)
        if hand.gesture is Gesture.PINCH and state.pinch_flash_timer > 0:
            midpoint = tuple(np.mean(local_pixels[[4, 8]], axis=0).astype(int))
            radius = int(8 + (1.0 - state.pinch_flash_timer) * 35)
            cv2.circle(sharp, midpoint, radius, (220, 245, 255), 2, cv2.LINE_AA)
            cv2.circle(glow, midpoint, max(8, int(24 * state.pinch_flash_timer)), color, -1, cv2.LINE_AA)
        if self.quality != "low" and not self._degraded:
            glow = cv2.GaussianBlur(glow, (GLOW_BLUR_KERNEL, GLOW_BLUR_KERNEL), 0)
        roi = frame[y1:y2, x1:x2]
        cv2.addWeighted(roi, 1.0, glow, GLOW_ALPHA * state.fade, 0, roi)
        mask = np.any(sharp > 0, axis=2)
        mixed = cv2.addWeighted(roi, 0.35, sharp, 0.90 * state.fade, 0)
        roi[mask] = mixed[mask]
        role = "PRIMARY" if hand.is_primary else "SECONDARY"
        parts=[]
        if SHOW_HAND_ROLE_LABEL:parts.append(role)
        if SHOW_GESTURE_LABEL:parts.append(hand.gesture.value.upper())
        if parts:
            label="  ".join(parts);label_point=(max(3,center[0]-base_radius),min(sharp.shape[0]-8,center[1]+base_radius+18))
            text_size=cv2.getTextSize(label,cv2.FONT_HERSHEY_DUPLEX,.3,1)[0]
            chip=roi.copy();cv2.rectangle(chip,(label_point[0]-3,label_point[1]-11),(label_point[0]+text_size[0]+4,label_point[1]+4),(15,18,22),-1)
            cv2.addWeighted(chip,.55,roi,.45,0,roi);cv2.putText(roi,label,label_point,cv2.FONT_HERSHEY_DUPLEX,.3,color,1,cv2.LINE_AA)

    @staticmethod
    def _draw_technical_hand(layer: np.ndarray, pixels: np.ndarray, state: HandEffectState, color: tuple[int, int,int], now: float, gesture: Gesture, primary: bool) -> None:
        faded_white = tuple(int(205 * state.fade) for _ in range(3))
        if SHOW_BASE_SKELETON:
            for start, end in CONNECTIONS:
                cv2.line(layer, tuple(pixels[start]), tuple(pixels[end]), tuple(int(c * 0.32 * state.fade) for c in color), 1, cv2.LINE_AA)
        # The primary fingertip is rendered once by EffectsRenderer; only the
        # secondary hand needs a marker here.
        if SHOW_ACTIVE_FINGERTIP_MARKER and not primary:
            radius=ACTIVE_FINGERTIP_MARKER_RADIUS+(1 if math.sin(now*7)>0 else 0)
            cv2.circle(layer,tuple(pixels[8]),radius,color,1,cv2.LINE_AA)
        if gesture is Gesture.PINCH:cv2.circle(layer,tuple(pixels[4]),max(3,ACTIVE_FINGERTIP_MARKER_RADIUS-3),faded_white,1,cv2.LINE_AA)
        if SHOW_ALL_FINGERTIP_MARKERS:
            for index in (4,12,16,20):cv2.circle(layer,tuple(pixels[index]),PASSIVE_FINGERTIP_MARKER_RADIUS,tuple(int(c*.45) for c in faded_white),-1,cv2.LINE_AA)

    def _render_connection_arc(self, frame: np.ndarray, hands: list[RoleTrackedHand], now: float) -> None:
        if not ENABLE_HAND_CONNECTION_ARC or len(hands) != 2 or self.quality == "low":
            return
        centers = [palm_geometry(hand.pixels)[0] for hand in hands]
        if math.dist(*centers) > frame.shape[1] * 0.58:
            return
        a, b = centers
        midpoint = ((a[0] + b[0]) // 2, min(a[1], b[1]) - int(24 + 6 * math.sin(now * 4)))
        curve = []
        for index in range(25):
            t = index / 24.0
            x = int((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * midpoint[0] + t * t * b[0])
            y = int((1 - t) ** 2 * a[1] + 2 * (1 - t) * t * midpoint[1] + t * t * b[1])
            curve.append((x, y))
        for index in range(1, len(curve), 2):
            cv2.line(frame, curve[index - 1], curve[index], (160, 210, 255), 1, cv2.LINE_AA)

    def _particle_limit(self) -> int:
        base = 25 if self.quality == "low" else 50 if self.quality == "medium" else MAX_FIRE_PARTICLES
        return max(12, int(base * self.performance_scale))

    def adapt_to_fps(self, fps: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.performance_scale = 1.0 if fps >= FX_RECOVER_ABOVE_FPS else 0.62 if fps >= FX_DEGRADE_BELOW_FPS else 0.35
        if fps < FX_DEGRADE_BELOW_FPS:
            self._low_fps_since = self._low_fps_since or now
            self._degraded=True;self._recovery_since=None
        elif fps >= FX_RECOVER_ABOVE_FPS:
            self._low_fps_since = None
            self._recovery_since=self._recovery_since or now
            if now-self._recovery_since>=FX_RECOVERY_HOLD_SECONDS:self._degraded=False

    def cycle_quality(self) -> str:
        levels = ("low", "medium", "high", "ultra")
        self.quality = levels[(levels.index(self.quality) + 1) % len(levels)]
        return self.quality

    # Compatibility methods retained for callers outside main.py.
    def draw_particles(self, frame: np.ndarray, now: float) -> None:
        return

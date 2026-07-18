"""Air Canvas premium cyber interface entry point."""

from __future__ import annotations

import sys
import time
import ctypes
import argparse
import os

# Suppress native Google telemetry chatter without hiding Python exceptions or
# CameraManager diagnostics. These must be set before MediaPipe is imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

from air_canvas.animation_manager import AnimationManager
from air_canvas.camera_manager import CameraManager
from air_canvas.camera_selector import CameraSelector
from air_canvas.camera_view_controls import CameraViewControls
from air_canvas.drawing_assistant import DrawingAssistant
from air_canvas.drawing_assist_ui import DrawingAssistUI
from air_canvas.config import (
    APP_NAME, BRUSH_STEP, CAMERA_DISPLAY_MODE, CAMERA_READ_FAILURE_LIMIT, COLORS, DEBUG_CAMERA_UI, DEBUG_TOOLBAR_INTERACTION, DEFAULT_BRUSH_SIZE,
    DEBUG_DRAWING_PIPELINE, DEBUG_DRAWING_SMOOTHING, DEBUG_DUAL_HAND_TRACKING, DEBUG_GESTURES,
    DRAW_END_CONFIRM_FRAMES, DRAW_MIN_TRACKING_CONFIDENCE, DRAW_START_CONFIRM_FRAMES,
    ENABLE_HAND_SWIPE_HISTORY, ENABLE_SECONDARY_HAND_COMMANDS, ENABLE_TWO_HAND_CLEAR,
    enable_finger_trail, enable_glow, enable_hand_skeleton,
    enable_particles, FRAME_HEIGHT, FRAME_WIDTH, fullscreen, MAX_BRUSH_SIZE, MIN_BRUSH_SIZE, MIRROR_CAMERA,
    MODEL_PATH, OPEN_PALM_HOLD_SECONDS, OPEN_PALM_RESET_SECONDS, OUTPUT_DIR,
    TOOLBAR_DWELL_SECONDS, TOOLBAR_FAST_DWELL_SECONDS, TOOLBAR_PINCH_INSTANT_ACTIVATION,
    TOOLBAR_REENTRY_COOLDOWN_SECONDS, TOOLBAR_SAFE_INSET, VIEW_TRANSITION_SECONDS, WHITEBOARD_DARK_COLOR,
    WHITEBOARD_LIGHT_COLOR, DEFAULT_WHITEBOARD_THEME, WINDOW_NAME,
)
from air_canvas.drawing_canvas import DrawingCanvas
from air_canvas.dual_hand_tracker import DualHandTracker, RoleTrackedHand
from air_canvas.effects import EffectsRenderer
from air_canvas.gesture_detector import Gesture, GestureDetector
from air_canvas.hand_tracker import HandTracker, HandTrackerError
from air_canvas.history_manager import HistoryManager
from air_canvas.history_manager import CanvasSnapshot
from air_canvas.hand_effects import HandEffectsRenderer
from air_canvas.toolbar import Toolbar
from air_canvas.ui_renderer import UIRenderer
from air_canvas.ui_layout import CameraViewState, UILayout
from air_canvas.shape_recognizer import ShapeRecognitionResult, ShapeRecognizer, render_corrected_shape
from air_canvas.sketch_cleaner import CleanupResult, SketchCleaner
from air_canvas.config import UI_REFERENCE_WIDTH, UI_REFERENCE_HEIGHT
from air_canvas.config import (
    AUTO_SHAPE_DEFAULT, CLEANUP_DEFAULT, DEBUG_DRAW_ASSIST, DEBUG_SHAPE_RECOGNITION,
    DRAW_ASSIST_DEFAULT, SHAPE_CONFIDENCE_THRESHOLD, SHAPE_PREVIEW_TIMEOUT_SECONDS, ENABLE_DESKTOP_ANNOTATION,
)


class AirCanvasApp:
    """Coordinate capture, tracking, drawing, history, animation, and UI."""

    def __init__(self, camera_override: int | None = None) -> None:
        self.camera_manager = CameraManager()
        self.camera_selector = CameraSelector()
        self.camera_view_controls = CameraViewControls()
        self.drawing_assist_ui = DrawingAssistUI()
        self.camera_override = camera_override
        self.tracker: HandTracker | None = None
        self.dual_tracker: DualHandTracker | None = None
        self.canvas: DrawingCanvas | None = None
        self.toolbar: Toolbar | None = None
        self.gestures = GestureDetector()
        self.history = HistoryManager()
        self.animations = AnimationManager()
        self.effects = EffectsRenderer()
        self.hand_effects = HandEffectsRenderer()
        self.ui = UIRenderer()
        self.drawing_assistant = DrawingAssistant(DRAW_ASSIST_DEFAULT)
        self.shape_recognizer = ShapeRecognizer()
        self.sketch_cleaner = SketchCleaner()
        self.auto_shape = AUTO_SHAPE_DEFAULT
        self.cleanup_intensity = CLEANUP_DEFAULT
        self.pending_shape: dict[str, object] | None = None
        self.pending_cleanup: dict[str, object] | None = None
        self._stroke_before: CanvasSnapshot | None = None
        self._stroke_color: tuple[int, int, int] = (0, 0, 255)
        self._stroke_size = DEFAULT_BRUSH_SIZE
        self._shape_hold_anchor: tuple[int, int] | None = None
        self._shape_hold_started: float | None = None
        self.active_tool_id = "red"
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.enable_glow = enable_glow
        self.enable_skeleton = enable_hand_skeleton
        self.enable_particles = enable_particles
        self.enable_trail = enable_finger_trail
        self.whiteboard_mode = False
        self.whiteboard_theme = DEFAULT_WHITEBOARD_THEME.lower()
        self.fullscreen = fullscreen
        self.open_palm_started: float | None = None
        self.open_palm_triggered = False
        self.last_open_palm_seen = 0.0
        self.last_tool_selection = 0.0
        self.stroke_in_progress = False
        self.previous_gesture = Gesture.IDLE
        self.current_camera_frame: np.ndarray | None = None
        self.last_content_frame: np.ndarray | None = None
        self.transition_snapshot: np.ndarray | None = None
        self.transition_started = 0.0
        self.pinch_previous = False
        self.debug_fingertip: tuple[int, int] | None = None
        self.debug_pinch_distance = 0.0
        self.debug_pinch_active = False
        self.debug_hovered: str | None = None
        self.hovered_action_id: str | None = None
        self.hover_started_at: float | None = None
        self.last_activated_action_id: str | None = None
        self.last_toolbar_activation_at = -float("inf")
        self.toolbar_selection_locked = False
        self.toolbar_dwell_progress = 0.0
        self._last_dwell_log_percent = -1
        self._toolbar_cursor_sample: tuple[tuple[int, int], float] | None = None
        self._toolbar_cursor_speed = 0.0
        self._action_now = 0.0
        self._action_point: tuple[int, int] | None = None
        self._action_size = (FRAME_WIDTH, FRAME_HEIGHT)
        self.current_hands: list[RoleTrackedHand] = []
        self._dual_hands_visible = False
        self._secondary_wrist_history: list[tuple[float, int]] = []
        self._secondary_command_at = -float("inf")
        self._two_hand_clear_started: float | None = None
        self._two_hand_clear_locked = False
        self._secondary_open_locked = False
        self._secondary_pinch_locked = False
        self._two_fists_locked = False
        self._last_primary_hand_id: int | None = None
        self._draw_confirm_frames = 0
        self._draw_absent_frames = 0
        self._drawing_gesture_confirmed = False
        self._last_drawing_rejection = ""
        self.display_mode = CAMERA_DISPLAY_MODE
        self.layout = UILayout.create(UI_REFERENCE_WIDTH, UI_REFERENCE_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT, self.display_mode)
        self.camera_views: dict[int, CameraViewState] = {}
        self._last_view_update = time.monotonic()
        self._panning_camera = False
        self._pan_previous: tuple[int, int] | None = None
        self.desktop_annotation = None
        self.desktop_hotkeys = None
        self._desktop_window_hidden = False

    def run(self) -> int:
        """Run until exit and release all native resources on every path."""
        try:
            self._initialize()
            return self._event_loop()
        except (HandTrackerError, RuntimeError, OSError) as exc:
            print(f"{APP_NAME}: {exc}", file=sys.stderr)
            return 1
        finally:
            self._close_desktop_annotation()
            if self.desktop_hotkeys is not None:
                self.desktop_hotkeys.close()
            self.camera_manager.release()
            if self.tracker is not None:
                self.tracker.close()
            cv2.destroyAllWindows()

    def _initialize(self) -> None:
        self.tracker = HandTracker(MODEL_PATH)
        self.dual_tracker = DualHandTracker(self.tracker)
        opened = self.camera_manager.open_preferred_camera(self.camera_override)
        ok, frame = self.camera_manager.read_frame() if opened else (False, None)
        if not ok or frame is None or frame.size == 0:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        height, width = frame.shape[:2]
        self.canvas = DrawingCanvas(width, height)
        self.toolbar = Toolbar(width)
        self.camera_selector.set_cameras(self.camera_manager.detected_cameras, None if self.camera_manager.active_info is None else self.camera_manager.active_info.index)
        if not opened:
            self.camera_selector.show_no_feed()
        self._after_camera_change(time.monotonic())
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, UI_REFERENCE_WIDTH, UI_REFERENCE_HEIGHT)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        self._apply_fullscreen()
        if ENABLE_DESKTOP_ANNOTATION:
            try:
                from air_canvas.hotkey_manager import HotkeyManager
                self.desktop_hotkeys = HotkeyManager()
                if not self.desktop_hotkeys.register("desktop"):
                    print("[DESKTOP] F8 is already registered by another application", file=sys.stderr)
            except OSError as exc:
                print(f"[DESKTOP] Global hotkeys unavailable: {exc}", file=sys.stderr)

    def _event_loop(self) -> int:
        assert self.tracker is not None and self.dual_tracker is not None
        assert self.canvas is not None and self.toolbar is not None
        previous_frame_time = time.perf_counter()
        fps = 0.0
        invalid_frames = 0

        while True:
            ok, camera_frame = self.camera_manager.read_frame()
            frame_live = bool(ok and camera_frame is not None and camera_frame.size > 0)
            if not ok or camera_frame is None or camera_frame.size == 0:
                invalid_frames += 1
                if invalid_frames == CAMERA_READ_FAILURE_LIMIT:
                    self._end_stroke("camera disconnected")
                    self.animations.notify("CAMERA DISCONNECTED", time.monotonic(), 1.3, (40, 100, 255))
                    if self.camera_manager.recover_camera():
                        self._after_camera_change(time.monotonic())
                        invalid_frames = 0
                    else:
                        self.animations.notify("NO CAMERA AVAILABLE", time.monotonic(), 1.5, (40, 100, 255))
                elif invalid_frames > CAMERA_READ_FAILURE_LIMIT * 2:
                    invalid_frames = CAMERA_READ_FAILURE_LIMIT
                camera_frame = self.current_camera_frame.copy() if self.current_camera_frame is not None else np.zeros_like(self.canvas.image)
            else:
                invalid_frames = 0
            if frame_live and MIRROR_CAMERA:
                camera_frame = cv2.flip(camera_frame, 1)
            if frame_live:
                self.current_camera_frame = camera_frame.copy()
            height, width = camera_frame.shape[:2]
            self.canvas.resize_if_needed(width, height)
            now = time.monotonic()
            view_state = self._active_view_state()
            view_state.update(now - self._last_view_update)
            self._last_view_update = now
            window_width, window_height = self._window_size()
            view_zoom = 1.0 if self.whiteboard_mode else view_state.current_zoom
            view_pan_x = 0.0 if self.whiteboard_mode else view_state.pan_x
            view_pan_y = 0.0 if self.whiteboard_mode else view_state.pan_y
            self.layout = UILayout.create(window_width, window_height, width, height, self.display_mode, view_zoom, view_pan_x, view_pan_y)
            self.toolbar.update_layout(self.layout)
            self._process_camera_ui(now)

            hands = self.dual_tracker.detect(camera_frame, now) if frame_live else []
            self.current_hands = hands
            if self.desktop_hotkeys is not None:
                for desktop_action in self.desktop_hotkeys.poll():
                    self._handle_desktop_action(desktop_action, now)
            if self.desktop_annotation is not None and self.desktop_annotation.active:
                self.desktop_annotation.set_camera(-1 if self.camera_manager.active_info is None else self.camera_manager.active_info.index)
                try:
                    self.desktop_annotation.update(hands, camera_frame if frame_live else None, fps, now)
                except Exception as exc:
                    print(f"[DESKTOP] Overlay stopped: {exc}", file=sys.stderr)
                    self._close_desktop_annotation()
                    self.animations.notify("DESKTOP OVERLAY FAILED", now, 2.0, (50, 80, 255))
                current = time.perf_counter()
                instantaneous_fps = 1.0 / max(current - previous_frame_time, 1e-6)
                fps = instantaneous_fps if fps == 0.0 else fps * 0.88 + instantaneous_fps * 0.12
                previous_frame_time = current
                if self.desktop_annotation is not None and not self.desktop_annotation.active:
                    self._close_desktop_annotation()
                cv2.waitKeyEx(1)
                continue
            hand = next((item for item in hands if item.is_primary), None)
            gesture = Gesture.IDLE
            hover_tool: str | None = None
            index_tip: tuple[int, int] | None = None
            if hand is not None:
                if self._last_primary_hand_id is not None and hand.tracking_id != self._last_primary_hand_id:
                    self._end_stroke("primary hand changed")
                    self._reset_draw_confirmation()
                    if DEBUG_DRAWING_PIPELINE:
                        print(f"[DRAW] Primary hand changed: {self._last_primary_hand_id} -> {hand.tracking_id}")
                self._last_primary_hand_id = hand.tracking_id
                gesture = hand.gesture
                index_tip = hand.index_tip
                raw_tip = hand.raw_landmarks[8, :2]
                canvas_point = (
                    (int(round(float(raw_tip[0]) * width)), int(round(float(raw_tip[1]) * height)))
                    if np.isfinite(raw_tip).all() else (-1, -1)
                )
                self.toolbar.update_states(self.history.can_undo(), self.history.can_redo())
                ui_tip = self.layout.camera_to_window(index_tip, (width, height))
                hovered_button = self.toolbar.get_hovered_button(*ui_tip)
                hover_tool = None if hovered_button is None else hovered_button.action
                self.debug_fingertip = ui_tip
                self.debug_pinch_distance = hand.pinch_distance
                self.debug_pinch_active = hand.pinch_active
                self.debug_hovered = hover_tool
                self._handle_gesture(
                    gesture, ui_tip, hover_tool, hand.pinch_active,
                    hand.pinch_distance, now, (width, height), hand.tracking_id, hand.confidence, canvas_point,
                )
                self._update_shape_hold(canvas_point, now)
            else:
                self._end_stroke("hand lost")
                self._reset_draw_confirmation()
                self._last_primary_hand_id = None
                self._reset_open_palm_if_elapsed(now)
                self._reset_toolbar_interaction(clear_cursor=True)
            self._handle_multi_hand_commands(hands, now, (width, height))
            self.hand_effects.particles_enabled = self.enable_particles
            self.hand_effects.trails_enabled = self.enable_trail
            self.hand_effects.update(hands, now, fps)

            base = np.full_like(camera_frame, self._board_color()) if self.whiteboard_mode else camera_frame.copy()
            composed = self.canvas.overlay(base, glow=self.enable_glow)
            if self.enable_skeleton:
                hovered_target = None
                if hover_tool is not None:
                    hovered = next((button for button in self.toolbar.buttons if button.action == hover_tool), None)
                    if hovered is not None:
                        x1, y1, x2, y2 = hovered.rectangle
                        hovered_target = self.layout.window_to_camera(((x1 + x2) // 2, (y1 + y2) // 2), (width, height))
                self.hand_effects.render(composed, hands, now, hovered_target)
            cursor_color = self.active_color
            if index_tip is not None:
                if self.enable_trail:
                    self.effects.add_cursor_point(index_tip, now, cursor_color)
                    self.effects.draw_trail(composed, now)
                self.effects.draw_cursor(composed, index_tip, cursor_color, gesture is Gesture.DRAW, now)
            if self.enable_particles:
                self.effects.draw_particles(composed, now)
            composed = self._apply_view_transition(composed, now)
            if self.pending_shape is not None:
                result = self.pending_shape["result"]
                assert isinstance(result, ShapeRecognitionResult)
                render_corrected_shape(composed, np.zeros(composed.shape[:2], np.uint8), result, (80, 235, 255), max(2, self._stroke_size))
            if self.pending_cleanup is not None:
                cleanup = self.pending_cleanup["result"]
                assert isinstance(cleanup, CleanupResult)
                cleaned = base.copy(); visible = cleanup.mask > 0; cleaned[visible] = cleanup.image[visible]
                midpoint = composed.shape[1] // 2
                composed[:, midpoint:] = cleaned[:, midpoint:]
                cv2.line(composed, (midpoint, 0), (midpoint, composed.shape[0]), (80, 220, 255), 2, cv2.LINE_AA)
            self.last_content_frame = composed.copy()
            presented = self.layout.present(composed)

            current = time.perf_counter()
            instantaneous_fps = 1.0 / max(current - previous_frame_time, 1e-6)
            fps = instantaneous_fps if fps == 0.0 else fps * 0.88 + instantaneous_fps * 0.12
            previous_frame_time = current
            mode_label = self._mode_label(gesture)
            self.ui.draw(
                presented, layout=self.layout, fps=fps, tracking=bool(hands), hand_count=len(hands), tool=self.active_tool_id.title(),
                brush_size=self.brush_size, mode=mode_label, glow=self.enable_glow,
                skeleton=self.enable_skeleton, view=self._view_label(),
                history_position=self.history.current_position,
                history_total=self.history.total_states,
                quality=self.hand_effects.quality,
                assist=self.drawing_assistant.level, auto_shape=self.auto_shape, cleanup=self.cleanup_intensity,
            )
            self.toolbar.draw(
                presented, self.active_tool_id, self.hovered_action_id, self.toolbar_dwell_progress, now,
                can_undo=self.history.can_undo(), can_redo=self.history.can_redo(),
                whiteboard=self.whiteboard_mode,
            )
            self.drawing_assist_ui.render_controls(presented, self.layout, self.drawing_assistant.level, self.auto_shape, self.cleanup_intensity)
            self.ui.draw_help_panel(presented, self.layout)
            live_status = self.camera_manager.last_read_success and now - self.camera_manager.last_successful_frame_time < 1.0
            self.camera_selector.render(presented, live_status, self.layout)
            self.camera_view_controls.render(presented, self.layout, view_state.current_zoom)
            if self.pending_shape is not None:
                result = self.pending_shape["result"]
                assert isinstance(result, ShapeRecognitionResult)
                self.drawing_assist_ui.render_confirmation(presented, self.layout, f"{result.shape_type.upper()} DETECTED - HOLD TO ACCEPT")
            elif self.pending_cleanup is not None:
                self.drawing_assist_ui.render_confirmation(presented, self.layout, "SKETCH CLEANUP PREVIEW", "ENTER TO APPLY / ESC TO CANCEL")
            if DEBUG_CAMERA_UI:
                self._draw_camera_debug(presented)
            self.animations.draw(presented, now)
            if DEBUG_TOOLBAR_INTERACTION:
                self.ui.draw_toolbar_debug(
                    presented, fingertip=self.debug_fingertip,
                    gesture=gesture.value, pinch_distance=self.debug_pinch_distance,
                    pinch_active=self.debug_pinch_active, hovered=self.debug_hovered,
                    toolbar_y=self.toolbar.y_range,
                )
            if DEBUG_DUAL_HAND_TRACKING or DEBUG_DRAWING_SMOOTHING or DEBUG_GESTURES or DEBUG_DRAWING_PIPELINE or DEBUG_DRAW_ASSIST or DEBUG_SHAPE_RECOGNITION:
                self._draw_runtime_debug(presented)
            self._expire_pending_preview(now)
            cv2.imshow(WINDOW_NAME, presented)
            key = cv2.waitKeyEx(1)
            control, shift = self._keyboard_modifiers()
            if not self._handle_key(key, now, (width, height), control, shift):
                break
        return 0

    def _handle_gesture(
        self,
        gesture: Gesture,
        point: tuple[int, int],
        hover_tool: str | None,
        pinch_now: bool,
        pinch_distance: float,
        now: float,
        size: tuple[int, int],
        hand_id: int,
        confidence: float,
        canvas_point: tuple[int, int] | None = None,
    ) -> None:
        assert self.canvas is not None
        drawing_point = point if canvas_point is None else canvas_point

        self._action_now, self._action_point, self._action_size = now, point, size
        # Selection mode owns the cursor and always disables drawing.
        selection_mode = gesture in {Gesture.SELECT, Gesture.PINCH}
        new_pinch = pinch_now and not self.pinch_previous
        self.pinch_previous = pinch_now
        if new_pinch and (self.pending_shape is not None or self.pending_cleanup is not None):
            self._accept_pending_preview(now)
            return
        if new_pinch:
            self.animations.ripple(point, now, self.active_color)
        if selection_mode:
            self._end_stroke("toolbar selection")
            self._reset_draw_confirmation()
            self._update_toolbar_dwell(hover_tool, point, now)
            if TOOLBAR_PINCH_INSTANT_ACTIVATION and new_pinch and hover_tool and not self.toolbar_selection_locked:
                print(f"[TOOLBAR] Activating: {hover_tool}")
                self.execute_toolbar_action(hover_tool)
                self.last_activated_action_id = hover_tool
                self.toolbar_selection_locked = True
                self.last_toolbar_activation_at = now
        elif gesture is Gesture.DRAW and confidence >= DRAW_MIN_TRACKING_CONFIDENCE:
            self._draw_confirm_frames += 1
            self._draw_absent_frames = 0
            if not self._drawing_gesture_confirmed and self._draw_confirm_frames >= DRAW_START_CONFIRM_FRAMES:
                self._drawing_gesture_confirmed = True
            if self._drawing_gesture_confirmed and not self.stroke_in_progress:
                self._reject_pending_preview()
                self.history.begin_stroke(self.canvas.image, self.canvas.mask)
                self._stroke_before = self.history.snapshot(self.canvas.image, self.canvas.mask)
                self._stroke_color = self.active_color
                self._stroke_size = self.brush_size
                self.drawing_assistant.reset()
                assisted_point = self.drawing_assistant.add_point(drawing_point, now)
                self.stroke_in_progress = True
                if self.canvas.start_stroke(assisted_point, hand_id, now) and DEBUG_DRAWING_PIPELINE:
                    print(f"[DRAW] Stroke started: hand={hand_id}")
            elif self._drawing_gesture_confirmed:
                assisted_point = self.drawing_assistant.add_point(drawing_point, now)
                accepted = self.canvas.append_stroke_point(assisted_point, hand_id, now, self.active_color, self.brush_size, self.active_tool_id == "eraser")
                if not accepted and self.canvas.last_rejection_reason not in {"below minimum distance", "smoothed movement below minimum", ""}:
                    if self.canvas.last_rejection_reason != self._last_drawing_rejection and DEBUG_DRAWING_PIPELINE:
                        print(f"[DRAW] Point rejected: {self.canvas.last_rejection_reason}")
                    self._last_drawing_rejection = self.canvas.last_rejection_reason
        else:
            self._draw_confirm_frames = 0
            self._draw_absent_frames += 1
            force_end = gesture in {Gesture.OPEN_PALM, Gesture.FIST, Gesture.PINCH}
            if force_end or self._draw_absent_frames >= DRAW_END_CONFIRM_FRAMES:
                self._end_stroke(f"gesture {gesture.value}")
                self._drawing_gesture_confirmed = False

        if not selection_mode:
            self._reset_toolbar_interaction(clear_hover=True)

        if gesture is Gesture.OPEN_PALM and len(self.current_hands) < 2:
            self.last_open_palm_seen = now
            if self.open_palm_started is None:
                self.open_palm_started = now
            elif not self.open_palm_triggered and now - self.open_palm_started >= OPEN_PALM_HOLD_SECONDS:
                self.execute_toolbar_action("clear")
                self.open_palm_triggered = True
        else:
            self._reset_open_palm_if_elapsed(now)

        if gesture is not self.previous_gesture:
            messages = {Gesture.DRAW: "ERASER MODE" if self.active_tool_id == "eraser" else "DRAW MODE", Gesture.SELECT: "SELECT MODE", Gesture.FIST: "PAUSED"}
            if gesture in messages:
                self.animations.notify(messages[gesture], now, 1.0)
        self.previous_gesture = gesture

    def _reset_toolbar_interaction(self, clear_hover: bool = True, clear_cursor: bool = False) -> None:
        self.pinch_previous = False
        self.gestures.reset_pinch()
        self.debug_pinch_active = False
        self.hovered_action_id = None
        self.hover_started_at = None
        self.last_activated_action_id = None
        self.toolbar_selection_locked = False
        self.toolbar_dwell_progress = 0.0
        self._last_dwell_log_percent = -1
        self._toolbar_cursor_sample = None
        self._toolbar_cursor_speed = 0.0
        if clear_hover and self.toolbar is not None:
            self.toolbar.clear_hover()
            self.debug_hovered = None
        if clear_cursor:
            self.debug_fingertip = None

    def _end_stroke(self, reason: str = "ended") -> None:
        assert self.canvas is not None
        was_in_progress = self.stroke_in_progress
        assisted_points = self.drawing_assistant.finish_stroke() if was_in_progress else []
        if self.stroke_in_progress:
            self.history.commit_stroke(self.canvas.image, self.canvas.mask)
        self.canvas.end_stroke(reason)
        self.stroke_in_progress = False
        self._draw_confirm_frames = 0
        self._draw_absent_frames = 0
        self._drawing_gesture_confirmed = False
        if (was_in_progress and self.auto_shape and self.active_tool_id != "eraser" and
                self._stroke_before is not None and len(assisted_points) >= 6):
            result = self.shape_recognizer.recognize(assisted_points)
            if DEBUG_SHAPE_RECOGNITION:
                print(f"[SHAPE] {result.shape_type} confidence={result.confidence:.3f} bounds={result.bounding_box}")
            if result.shape_type != "freehand" and result.confidence >= SHAPE_CONFIDENCE_THRESHOLD:
                self.pending_shape = {
                    "result": result, "before": self._stroke_before,
                    "color": self._stroke_color, "size": self._stroke_size,
                    "expires": time.monotonic() + SHAPE_PREVIEW_TIMEOUT_SECONDS,
                }
                self.animations.notify(f"{result.shape_type.upper()} DETECTED", time.monotonic(), 1.0)
        self._stroke_before = None
        if was_in_progress and DEBUG_DRAWING_PIPELINE:
            print(f"[DRAW] Stroke ended: {reason}")

    def _reset_draw_confirmation(self) -> None:
        self._draw_confirm_frames = 0
        self._draw_absent_frames = 0
        self._drawing_gesture_confirmed = False

    def _reset_open_palm_if_elapsed(self, now: float) -> None:
        if now - self.last_open_palm_seen > OPEN_PALM_RESET_SECONDS:
            self.open_palm_started = None
            self.open_palm_triggered = False

    def _handle_multi_hand_commands(self, hands: list[RoleTrackedHand], now: float, size: tuple[int, int]) -> None:
        if len(hands) == 2 and not self._dual_hands_visible:
            self.animations.notify("DUAL HAND MODE", now, 1.2, (120, 210, 255))
        self._dual_hands_visible = len(hands) == 2
        primary = next((hand for hand in hands if hand.is_primary), None)
        secondary = next((hand for hand in hands if hand.is_secondary), None)
        two_fists = len(hands) == 2 and all(hand.gesture is Gesture.FIST for hand in hands)
        if two_fists and not self._two_fists_locked:
            self._end_stroke()
            self.animations.notify("PAUSED", now, 1.0, (40, 110, 255))
            self._two_fists_locked = True
        elif not two_fists:
            self._two_fists_locked = False
        if ENABLE_TWO_HAND_CLEAR and len(hands) == 2 and all(hand.gesture is Gesture.OPEN_PALM for hand in hands):
            if self._two_hand_clear_started is None:
                self._two_hand_clear_started = now
            progress = min(1.0, (now - self._two_hand_clear_started) / 1.0)
            self.animations.notify(f"CLEAR {int(progress * 100):02d}%", now, 0.12, (80, 220, 255))
            if progress >= 1.0 and not self._two_hand_clear_locked:
                self._action_now, self._action_point, self._action_size = now, None, size
                self.execute_toolbar_action("clear")
                self._two_hand_clear_locked = True
            return
        self._two_hand_clear_started = None
        self._two_hand_clear_locked = False
        if not ENABLE_SECONDARY_HAND_COMMANDS or secondary is None or primary is None or primary.gesture is Gesture.DRAW:
            self._secondary_wrist_history.clear()
            return
        if secondary.gesture is Gesture.OPEN_PALM and not self._secondary_open_locked:
            self.ui.help_visible = not self.ui.help_visible
            self.animations.notify("GESTURE HELP ON" if self.ui.help_visible else "GESTURE HELP OFF", now, 1.0)
            self._secondary_command_at = now
            self._secondary_open_locked = True
        elif secondary.gesture is not Gesture.OPEN_PALM:
            self._secondary_open_locked = False
        if secondary.gesture is Gesture.PINCH and not self._secondary_pinch_locked and now - self._secondary_command_at > 0.6:
            self.enable_glow = not self.enable_glow
            self.animations.notify(f"GLOW {'ON' if self.enable_glow else 'OFF'}", now, 1.0)
            self._secondary_command_at = now
            self._secondary_pinch_locked = True
        elif secondary.gesture is not Gesture.PINCH:
            self._secondary_pinch_locked = False
        wrist_x = int(secondary.pixels[0][0])
        self._secondary_wrist_history.append((now, wrist_x))
        self._secondary_wrist_history = [sample for sample in self._secondary_wrist_history if now - sample[0] <= 0.35]
        if ENABLE_HAND_SWIPE_HISTORY and len(self._secondary_wrist_history) >= 3 and now - self._secondary_command_at > 0.75:
            displacement = wrist_x - self._secondary_wrist_history[0][1]
            if abs(displacement) >= 130 and secondary.gesture not in {Gesture.FIST, Gesture.OPEN_PALM}:
                self._action_now, self._action_size = now, size
                self.execute_toolbar_action("redo" if displacement > 0 else "undo")
                self._secondary_command_at = now
                self._secondary_wrist_history.clear()

    @property
    def active_color(self) -> tuple[int, int, int]:
        return COLORS.get(self.active_tool_id.title(), (255, 255, 255))

    def _update_toolbar_dwell(self, action_id: str | None, point: tuple[int, int], now: float) -> None:
        if self._toolbar_cursor_sample is not None:
            previous, sampled_at = self._toolbar_cursor_sample
            dt = max(1e-3, now - sampled_at)
            instant_speed = float(np.linalg.norm(np.subtract(point, previous))) / dt
            self._toolbar_cursor_speed = self._toolbar_cursor_speed * 0.65 + instant_speed * 0.35
        self._toolbar_cursor_sample = (point, now)
        if action_id != self.hovered_action_id:
            self.hovered_action_id = action_id
            self._toolbar_cursor_sample = (point, now)
            self._toolbar_cursor_speed = 0.0
            cooldown_remaining = max(0.0, TOOLBAR_REENTRY_COOLDOWN_SECONDS - (now - self.last_toolbar_activation_at))
            self.hover_started_at = now + cooldown_remaining if action_id else None
            self.toolbar_selection_locked = False
            self.toolbar_dwell_progress = 0.0
            self._last_dwell_log_percent = -1
            if action_id:
                print(f"[TOOLBAR] Hover started: {action_id}")
            return
        if action_id is None or self.hover_started_at is None or self.toolbar_selection_locked:
            return
        assert self.toolbar is not None
        if not self.toolbar.is_inside_safe_area(action_id, *point):
            return
        elapsed = now - self.hover_started_at
        stable = self._toolbar_cursor_speed < 95.0
        dwell_seconds = TOOLBAR_FAST_DWELL_SECONDS if self._toolbar_cursor_speed < 35.0 else TOOLBAR_DWELL_SECONDS
        if not stable:
            self.hover_started_at = now
            self.toolbar_dwell_progress = 0.0
            return
        self.toolbar_dwell_progress = min(1.0, elapsed / dwell_seconds)
        percent = int(self.toolbar_dwell_progress * 100)
        bucket = percent // 25
        if bucket > self._last_dwell_log_percent // 25 and percent < 100:
            print(f"[TOOLBAR] Dwell progress: {action_id} {percent}%")
            self._last_dwell_log_percent = percent
        if elapsed >= dwell_seconds:
            print(f"[TOOLBAR] Activating: {action_id}")
            self.execute_toolbar_action(action_id)
            if self.last_activated_action_id != action_id:
                raise RuntimeError(f"Toolbar dwell completed but action did not execute: {action_id}")
            self.toolbar_selection_locked = True
            self.last_toolbar_activation_at = now

    def execute_toolbar_action(self, action_id: str) -> None:
        action_id = action_id.lower()
        now, point, size = self._action_now or time.monotonic(), self._action_point, self._action_size
        self._end_stroke()
        if action_id == "clear":
            self._clear(now, point, size)
        elif action_id == "save":
            self._save(False, now, size)
        elif action_id == "undo":
            self._undo(now)
        elif action_id == "redo":
            self._redo(now)
        elif action_id == "board":
            self._toggle_whiteboard(now)
        elif action_id in {"red", "blue", "green", "yellow", "white", "eraser"}:
            previous = self.active_tool_id
            self.active_tool_id = action_id
            print(f"[ACTION] Active tool changed from {previous} to {action_id}")
            self.animations.notify(f"TOOL: {action_id.upper()}", now, 1.1, self.active_color)
            self._warn_low_contrast(now, queue=True)
        else:
            raise ValueError(f"Unknown toolbar action: {action_id}")
        self.last_activated_action_id = action_id

    def _activate_tool(self, tool: str, now: float, point: tuple[int, int], size: tuple[int, int]) -> None:
        """Backward-compatible entry point; all behavior is centrally dispatched."""
        self._action_now, self._action_point, self._action_size = now, point, size
        self.execute_toolbar_action(tool)

    def _clear(self, now: float, point: tuple[int, int] | None, size: tuple[int, int]) -> None:
        assert self.canvas is not None
        self._end_stroke()
        self.history.begin_stroke(self.canvas.image, self.canvas.mask)
        self.canvas.clear()
        self.history.commit_stroke(self.canvas.image, self.canvas.mask)
        origin = point or (size[0] // 2, size[1] // 2)
        if self.enable_particles:
            self.effects.burst(origin, now, (80, 220, 255), 30)
        self.animations.notify("CANVAS CLEARED", now, 1.7, (80, 220, 255))

    def _save(self, include_camera: bool, now: float, size: tuple[int, int]) -> None:
        assert self.canvas is not None
        try:
            if include_camera:
                if self.whiteboard_mode:
                    board = np.full_like(self.canvas.image, self._board_color())
                    path = self.canvas.save_composite(board, OUTPUT_DIR)
                    self.animations.notify("WHITEBOARD COMPOSITE SAVED", now, 2.2, (80, 255, 160))
                else:
                    if self.current_camera_frame is None:
                        raise OSError("No camera frame is available yet.")
                    path = self.canvas.save_composite(self.current_camera_frame, OUTPUT_DIR)
                    self.animations.notify("DRAWING SAVED", now, 2.0, (80, 255, 160))
            else:
                background = self._board_color() if self.whiteboard_mode else None
                path = self.canvas.save(OUTPUT_DIR, background=background)
                self.animations.notify("DRAWING SAVED", now, 2.0, (80, 255, 160))
            if self.enable_particles:
                self.effects.burst((size[0] // 2, int(size[1] * 0.25)), now, (80, 255, 160), 18)
            print(f"Saved drawing to {path}")
        except OSError as exc:
            self.animations.notify("SAVE FAILED", now, 2.5, (80, 80, 255))
            print(f"{APP_NAME}: {exc}", file=sys.stderr)

    def _handle_key(self, key: int, now: float, size: tuple[int, int], control: bool = False, shift: bool = False) -> bool:
        assert self.canvas is not None
        if key in (10, 13) and (self.pending_shape is not None or self.pending_cleanup is not None):
            self._accept_pending_preview(now)
            return True
        if key == 0x770000 or key == 0x77:
            self._toggle_desktop_annotation(now)
            return True
        if key == 27 and (self.pending_shape is not None or self.pending_cleanup is not None):
            self._reject_pending_preview()
            self.animations.notify("PREVIEW CANCELLED", now, 0.9)
            return True
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("c"), ord("C")):
            self._action_now, self._action_point, self._action_size = now, None, size
            self.execute_toolbar_action("clear")
        elif key == ord("S"):
            self._save(True, now, size)
        elif key == ord("s"):
            self.execute_toolbar_action("save")
        elif key in (26, ord("z"), ord("Z")):
            self.execute_toolbar_action("redo" if (key == 26 and shift) or (control and shift) else "undo")
        elif key in (25, ord("y"), ord("Y")):
            self.execute_toolbar_action("redo")
        elif key in (ord("h"), ord("H")):
            self.enable_skeleton = not self.enable_skeleton
            self.animations.notify(f"HAND SKELETON {'ON' if self.enable_skeleton else 'OFF'}", now, 1.1)
        elif key in (ord("t"), ord("T")):
            self.enable_trail = not self.enable_trail
            self.animations.notify(f"TRAILS {'ON' if self.enable_trail else 'OFF'}", now, 1.1)
        elif key in (ord("k"), ord("K")):
            self.enable_particles = not self.enable_particles
            self.animations.notify(f"PARTICLES {'ON' if self.enable_particles else 'OFF'}", now, 1.1)
        elif key in (ord("p"), ord("P")):
            quality = self.hand_effects.cycle_quality()
            self.animations.notify(f"EFFECT QUALITY: {quality.upper()}", now, 1.1)
        elif key in (ord("m"), ord("M")):
            self.display_mode = "fit" if self.display_mode == "cover" else "cover"
            self.animations.notify(f"DISPLAY: {self.display_mode.upper()}", now, 1.1)
        elif key in (ord("j"), ord("J")):
            self.ui.help_visible = not self.ui.help_visible
            self.animations.notify(f"GESTURE HELP {'ON' if self.ui.help_visible else 'OFF'}", now, 1.1)
        elif key in (ord("a"), ord("A")):
            self._cycle_drawing_assist(now)
        elif key in (ord("n"), ord("N")):
            self._toggle_auto_shape(now)
        elif key in (ord("l"), ord("L")):
            self._request_cleanup(now)
        elif key in (ord("+"), ord("=")):
            self._adjust_camera_zoom(1, now)
        elif key in (ord("-"), ord("_")):
            self._adjust_camera_zoom(-1, now)
        elif key == ord("0"):
            self._reset_camera_view(now)
        elif key in (2424832, 2555904, 2490368, 2621440):
            dx = -0.06 if key == 2424832 else 0.06 if key == 2555904 else 0.0
            dy = -0.06 if key == 2490368 else 0.06 if key == 2621440 else 0.0
            self._active_view_state().pan(dx, dy)
        elif key in (ord("x"), ord("X")):
            if self.dual_tracker is not None and self.dual_tracker.swap_roles(self.current_hands):
                self._end_stroke("hand roles swapped")
                self._reset_draw_confirmation()
                self.animations.notify("HAND ROLES SWAPPED", now, 1.3, (120, 210, 255))
                primary = next((hand for hand in self.current_hands if hand.is_primary), None)
                if primary is not None:
                    self.animations.ripple(tuple(primary.pixels[0]), now, (120, 210, 255))
        elif key in (ord("v"), ord("V")):
            self._switch_camera(-1 if shift else 1, now)
        elif key in (ord("r"), ord("R")):
            self._refresh_cameras(now)
        elif key in (ord("g"), ord("G")):
            self.enable_glow = not self.enable_glow
            self.animations.notify(f"GLOW {'ON' if self.enable_glow else 'OFF'}", now, 1.1)
        elif key in (ord("w"), ord("W")):
            self.execute_toolbar_action("board")
        elif key in (ord("b"), ord("B")):
            self._toggle_board_theme(now)
        elif key in (ord("f"), ord("F")):
            self.fullscreen = not self.fullscreen
            self._apply_fullscreen()
        elif ord("0") <= key <= ord("9"):
            number_actions = {
                ord("1"): "red", ord("2"): "blue", ord("3"): "green",
                ord("4"): "yellow", ord("5"): "white", ord("6"): "eraser",
                ord("7"): "undo", ord("8"): "redo", ord("9"): "board",
                ord("0"): "clear",
            }
            self._activate_tool(number_actions[key], now, (size[0] // 2, size[1] // 2), size)
        return True

    def _switch_camera(self, direction: int, now: float) -> None:
        self._end_stroke("camera switch")
        if self.camera_manager.switch_camera(direction):
            self._after_camera_change(now)
        else:
            self.animations.notify("CAMERA SWITCH FAILED", now, 1.3, (40, 100, 255))

    def _use_camera(self, index: int, now: float) -> None:
        current = None if self.camera_manager.active_info is None else self.camera_manager.active_info.index
        if index == current:
            self.animations.notify(f"CAMERA {index} ACTIVE", now, 1.0, (80, 230, 255))
            return
        self._end_stroke("mouse camera switch")
        if self.camera_manager.switch_to(index):
            self._after_camera_change(now)
            self.animations.notify(f"CAMERA {index} ACTIVE", now, 1.4, (80, 230, 255))
        else:
            self.animations.notify(f"CAMERA {index} FAILED", now, 1.4, (40, 100, 255))

    def _refresh_cameras(self, now: float) -> None:
        cameras = self.camera_manager.discover_cameras()
        active = None if self.camera_manager.active_info is None else self.camera_manager.active_info.index
        self.camera_selector.set_cameras(self.camera_manager.detected_cameras, active)
        self.animations.notify(f"{len(cameras)} CAMERAS FOUND" if cameras else "NO CAMERA AVAILABLE", now, 1.2)

    def _process_camera_ui(self, now: float) -> None:
        if self.camera_selector.get_refresh_requested():
            self._refresh_cameras(now)
        test_index = self.camera_selector.get_test_camera()
        if test_index is not None:
            info, preview = self.camera_manager.test_camera(test_index)
            if preview is not None and MIRROR_CAMERA:
                preview = cv2.flip(preview, 1)
            self.camera_selector.set_test_result(test_index, info, preview)
        selected = self.camera_selector.get_selected_camera()
        if selected is not None:
            self._use_camera(selected, now)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _userdata: object = None) -> None:
        assist_action = self.drawing_assist_ui.update_mouse(event, x, y)
        if assist_action is not None:
            now = time.monotonic()
            if assist_action == "assist":
                self._cycle_drawing_assist(now)
            elif assist_action == "shape":
                self._toggle_auto_shape(now)
            elif assist_action == "clean":
                self._request_cleanup(now)
            elif assist_action == "desktop":
                self._toggle_desktop_annotation(now)
            elif assist_action == "accept":
                self._accept_pending_preview(now)
            elif assist_action == "cancel":
                self._reject_pending_preview()
                self.animations.notify("PREVIEW CANCELLED", now, 0.9)
            return
        action = self.camera_view_controls.update_mouse(event, x, y)
        if action is not None:
            now = time.monotonic()
            if action == "in":
                self._adjust_camera_zoom(1, now)
            elif action == "out":
                self._adjust_camera_zoom(-1, now)
            else:
                self._reset_camera_view(now)
            return
        inside_content = (self.layout.content_rect.x <= x <= self.layout.content_rect.x2 and
                          self.layout.content_rect.y <= y <= self.layout.content_rect.y2)
        if event == cv2.EVENT_MOUSEWHEEL and inside_content:
            self._adjust_camera_zoom(1 if self._mouse_wheel_delta(flags) > 0 else -1, time.monotonic())
            return
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN) and inside_content:
            self._panning_camera = True
            self._pan_previous = (x, y)
            return
        if event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self._panning_camera = False
            self._pan_previous = None
            return
        if event == cv2.EVENT_MOUSEMOVE and self._panning_camera and self._pan_previous is not None:
            previous_x, previous_y = self._pan_previous
            self._active_view_state().pan(2.0 * (x - previous_x) / max(1, self.layout.content_rect.width),
                                          2.0 * (y - previous_y) / max(1, self.layout.content_rect.height))
            self._pan_previous = (x, y)
            return
        self.camera_selector.update_mouse(event, x, y, flags)

    def _toggle_desktop_annotation(self, now: float) -> None:
        if self.desktop_annotation is not None and self.desktop_annotation.active:
            self._close_desktop_annotation()
            self.animations.notify("DESKTOP ANNOTATION OFF", now, 1.2)
            return
        try:
            from air_canvas.annotation_controller import AnnotationController
            camera_index = -1 if self.camera_manager.active_info is None else self.camera_manager.active_info.index
            self.desktop_annotation = AnnotationController(camera_index)
            self.desktop_annotation.enter()
            if self.desktop_hotkeys is not None:
                failed = self.desktop_hotkeys.register_active()
                if failed:
                    print(f"[DESKTOP] Hotkeys unavailable: {', '.join(failed)}", file=sys.stderr)
            cv2.moveWindow(WINDOW_NAME, self.desktop_annotation.monitor.right + 40, self.desktop_annotation.monitor.bottom + 40)
            self._desktop_window_hidden = True
        except Exception as exc:
            self._close_desktop_annotation()
            self.animations.notify("DESKTOP OVERLAY FAILED", now, 2.0, (50, 80, 255))
            print(f"[DESKTOP] Initialization failed: {exc}", file=sys.stderr)

    def _handle_desktop_action(self, action: str, now: float) -> None:
        if action == "desktop":
            self._toggle_desktop_annotation(now); return
        controller = self.desktop_annotation
        if controller is None or not controller.active:
            return
        if action == "exit": self._close_desktop_annotation()
        elif action == "input": controller.toggle_input()
        elif action == "laser": controller.toggle_laser()
        elif action == "preview":
            controller.preview_visible = not controller.preview_visible
            controller.notify(f"CAMERA PREVIEW {'ON' if controller.preview_visible else 'OFF'}")
        elif action == "calibrate": controller.start_calibration()
        elif action == "save":
            try: controller.save()
            except Exception as exc: print(f"[DESKTOP] Save failed: {exc}", file=sys.stderr)
        elif action == "toolbar": controller.toolbar.visible = not controller.toolbar.visible
        elif action == "undo": controller.renderer.undo(); controller.notify("UNDO")
        elif action == "redo": controller.renderer.redo(); controller.notify("REDO")
        elif action == "clear": controller.request_clear()
        elif action == "monitor": controller.cycle_monitor()
        elif action == "camera": self._switch_camera(1, now)

    def _close_desktop_annotation(self) -> None:
        if self.desktop_annotation is not None:
            self.desktop_annotation.close()
            self.desktop_annotation = None
        if self.desktop_hotkeys is not None:
            self.desktop_hotkeys.unregister_active()
        if self._desktop_window_hidden:
            try:
                cv2.moveWindow(WINDOW_NAME, 60, 60)
                cv2.resizeWindow(WINDOW_NAME, UI_REFERENCE_WIDTH, UI_REFERENCE_HEIGHT)
            except cv2.error:
                pass
            self._desktop_window_hidden = False

    def _active_view_state(self) -> CameraViewState:
        index = -1 if self.camera_manager.active_info is None else self.camera_manager.active_info.index
        return self.camera_views.setdefault(index, CameraViewState())

    @staticmethod
    def _mouse_wheel_delta(flags: int) -> int:
        delta = (flags >> 16) & 0xFFFF
        return delta - 0x10000 if delta >= 0x8000 else delta

    def _adjust_camera_zoom(self, direction: int, now: float) -> None:
        state = self._active_view_state()
        state.zoom_in() if direction > 0 else state.zoom_out()
        self.animations.notify(f"ZOOM {state.target_zoom:.2f}x", now, 0.9)

    def _reset_camera_view(self, now: float) -> None:
        self._active_view_state().reset_view()
        self.animations.notify("CAMERA VIEW RESET", now, 1.0)

    def _cycle_drawing_assist(self, now: float) -> None:
        self._end_stroke("assistance changed")
        level = self.drawing_assistant.cycle_level()
        self.animations.notify(f"ASSIST {level.upper()}", now, 1.0)

    def _toggle_auto_shape(self, now: float) -> None:
        self._end_stroke("auto shape changed")
        self.auto_shape = not self.auto_shape
        self._reject_pending_preview()
        self.animations.notify(f"AUTO SHAPE {'ON' if self.auto_shape else 'OFF'}", now, 1.0)

    def _request_cleanup(self, now: float) -> None:
        assert self.canvas is not None
        self._end_stroke("cleanup requested")
        self._reject_pending_preview()
        if not np.any(self.canvas.mask):
            self.animations.notify("NOTHING TO CLEAN", now, 1.0)
            return
        result = self.sketch_cleaner.clean(self.canvas.image, self.canvas.mask, self.cleanup_intensity)
        source = self.history.snapshot(self.canvas.image, self.canvas.mask)
        self.pending_cleanup = {"result": result, "source": source, "expires": now + SHAPE_PREVIEW_TIMEOUT_SECONDS}
        self.animations.notify("CLEANUP PREVIEW", now, 1.0)

    def _accept_pending_preview(self, now: float) -> None:
        assert self.canvas is not None
        if self.pending_shape is not None:
            pending = self.pending_shape
            result, before = pending["result"], pending["before"]
            assert isinstance(result, ShapeRecognitionResult) and isinstance(before, CanvasSnapshot)
            rough = self.history.snapshot(self.canvas.image, self.canvas.mask)
            corrected_image, corrected_mask = before.image.copy(), before.mask.copy()
            render_corrected_shape(corrected_image, corrected_mask, result, pending["color"], int(pending["size"]))  # type: ignore[arg-type]
            self.history.begin_stroke(rough.image, rough.mask)
            self.canvas.restore(CanvasSnapshot(corrected_image, corrected_mask))
            self.history.commit_stroke(self.canvas.image, self.canvas.mask)
            self.animations.notify(f"{result.shape_type.upper()} APPLIED", now, 1.1)
        elif self.pending_cleanup is not None:
            result = self.pending_cleanup["result"]
            assert isinstance(result, CleanupResult)
            self.history.begin_stroke(self.canvas.image, self.canvas.mask)
            self.canvas.restore(CanvasSnapshot(result.image, result.mask))
            self.history.commit_stroke(self.canvas.image, self.canvas.mask)
            self.animations.notify("CLEANUP APPLIED", now, 1.1)
        self._reject_pending_preview()

    def _reject_pending_preview(self) -> None:
        self.pending_shape = None
        self.pending_cleanup = None
        self._shape_hold_anchor = None
        self._shape_hold_started = None

    def _expire_pending_preview(self, now: float) -> None:
        pending = self.pending_shape or self.pending_cleanup
        if pending is not None and now >= float(pending["expires"]):
            self._reject_pending_preview()

    def _update_shape_hold(self, point: tuple[int, int], now: float) -> None:
        if self.pending_shape is None:
            self._shape_hold_anchor = None; self._shape_hold_started = None
            return
        if self._shape_hold_anchor is None or np.linalg.norm(np.subtract(point, self._shape_hold_anchor)) > 8.0:
            self._shape_hold_anchor, self._shape_hold_started = point, now
        elif self._shape_hold_started is not None and now - self._shape_hold_started >= 0.5:
            self._accept_pending_preview(now)

    def _after_camera_change(self, now: float) -> None:
        self._end_stroke("camera changed")
        self._last_primary_hand_id = None
        self.current_hands.clear()
        self.current_camera_frame = None
        self.previous_gesture = Gesture.IDLE
        self.gestures.reset_pinch()
        self._draw_confirm_frames = 0
        self._draw_absent_frames = 0
        self._drawing_gesture_confirmed = False
        if self.dual_tracker is not None:
            self.dual_tracker.reset_tracking()
        info = self.camera_manager.active_info
        if info is None:
            self.camera_selector.set_cameras(self.camera_manager.detected_cameras, None)
            self.camera_selector.show_no_feed()
            self.animations.notify("NO CAMERA AVAILABLE", now, 1.4, (40, 100, 255))
            return
        self.camera_selector.set_cameras(self.camera_manager.detected_cameras, info.index)
        if info.index == 0:
            label = "BUILT-IN CAMERA ACTIVE"
        elif info.index == 1:
            label = "USB CAMERA / CAMERA 1 ACTIVE"
        else:
            label = f"CAMERA {info.index} ACTIVE"
        self.animations.notify(label, now, 1.4, (80, 230, 255))

    def _undo(self, now: float) -> None:
        assert self.canvas is not None
        self._end_stroke()
        snapshot = self.history.undo(self.canvas.image, self.canvas.mask)
        if snapshot is None:
            self.animations.notify("NOTHING TO UNDO", now, 1.2, (90, 150, 255))
            return
        self.canvas.restore(snapshot)
        self.animations.notify("UNDO", now, 0.9)

    def _redo(self, now: float) -> None:
        assert self.canvas is not None
        self._end_stroke()
        snapshot = self.history.redo(self.canvas.image, self.canvas.mask)
        if snapshot is None:
            self.animations.notify("NOTHING TO REDO", now, 1.2, (90, 150, 255))
            return
        self.canvas.restore(snapshot)
        self.animations.notify("REDO", now, 0.9)

    def _toggle_whiteboard(self, now: float) -> None:
        self._end_stroke()
        self._reset_toolbar_interaction(clear_hover=False)
        self._start_view_transition(now)
        self.whiteboard_mode = not self.whiteboard_mode
        self.animations.notify("WHITEBOARD MODE" if self.whiteboard_mode else "CAMERA MODE", now, 1.3)
        self._warn_low_contrast(now, queue=True)

    def _toggle_board_theme(self, now: float) -> None:
        self._end_stroke()
        self._reset_toolbar_interaction(clear_hover=False)
        self._start_view_transition(now)
        self.whiteboard_theme = "dark" if self.whiteboard_theme == "light" else "light"
        if not self.whiteboard_mode:
            self.whiteboard_mode = True
        self.animations.notify("DARK BOARD" if self.whiteboard_theme == "dark" else "LIGHT BOARD", now, 1.3)
        self._warn_low_contrast(now, queue=True)

    def _board_color(self) -> tuple[int, int, int]:
        return WHITEBOARD_DARK_COLOR if self.whiteboard_theme == "dark" else WHITEBOARD_LIGHT_COLOR

    def _view_label(self) -> str:
        if not self.whiteboard_mode:
            return "Camera"
        return f"Whiteboard {self.whiteboard_theme.title()}"

    def _warn_low_contrast(self, now: float, queue: bool = False) -> None:
        if not self.whiteboard_mode or self.active_tool_id == "eraser":
            return
        color = COLORS.get(self.active_tool_id.title())
        if color is None:
            return
        board = self._board_color()
        luminance = lambda bgr: 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
        if abs(luminance(color) - luminance(board)) < 72:
            self.animations.notify("LOW CONTRAST COLOR", now, 1.5, (40, 190, 255), queue=queue)

    def _start_view_transition(self, now: float) -> None:
        self.transition_snapshot = None if self.last_content_frame is None else self.last_content_frame.copy()
        self.transition_started = now

    def _apply_view_transition(self, frame: np.ndarray, now: float) -> np.ndarray:
        if self.transition_snapshot is None or self.transition_snapshot.shape != frame.shape:
            return frame
        progress = (now - self.transition_started) / VIEW_TRANSITION_SECONDS
        if progress >= 1.0:
            self.transition_snapshot = None
            return frame
        eased = max(0.0, min(1.0, progress))
        return cv2.addWeighted(self.transition_snapshot, 1.0 - eased, frame, eased, 0)

    @staticmethod
    def _keyboard_modifiers() -> tuple[bool, bool]:
        if sys.platform != "win32":
            return False, False
        control = bool(ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000)
        shift = bool(ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000)
        return control, shift

    def _apply_fullscreen(self) -> None:
        mode = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)

    def _window_size(self) -> tuple[int, int]:
        try:
            _x, _y, width, height = cv2.getWindowImageRect(WINDOW_NAME)
            if width >= 640 and height >= 480:
                return width, height
        except cv2.error:
            pass
        return self.layout.window_width, self.layout.window_height

    def _mode_label(self, gesture: Gesture) -> str:
        if gesture is Gesture.DRAW and self.active_tool_id == "eraser":
            return "Eraser"
        return gesture.value

    def _draw_runtime_debug(self, frame: np.ndarray) -> None:
        lines = [f"HANDS: {len(self.current_hands)}"]
        if DEBUG_DUAL_HAND_TRACKING or DEBUG_GESTURES:
            lines.extend(
                f"ID {hand.tracking_id} {'PRIMARY' if hand.is_primary else 'SECONDARY'} {hand.handedness} {hand.confidence:.2f} {hand.gesture.value} RAW {tuple(hand.raw_landmarks[8, :2].round(3))} SMOOTH {hand.index_tip}"
                for hand in self.current_hands
            )
        if DEBUG_DRAWING_SMOOTHING and self.canvas is not None:
            smoother = self.canvas.smoother
            lines.append(f"VELOCITY {smoother.velocity:.1f}  REJECTED {smoother.rejected_outliers}")
        if DEBUG_DRAWING_PIPELINE and self.canvas is not None:
            raw = self.canvas.previous_raw_point
            smoothed = self.canvas.previous_point
            lines.extend((
                f"DRAW HAND {self.canvas.active_hand_id} RAW {raw} SMOOTH {smoothed}",
                f"ACCEPTED {self.canvas.last_point_accepted} REASON {self.canvas.last_rejection_reason or 'none'}",
                f"DIST {self.canvas.last_distance:.1f} VEL {self.canvas.smoother.velocity:.1f} ACTIVE {self.stroke_in_progress}",
                f"DRAW CONFIRM {self._draw_confirm_frames}/{DRAW_START_CONFIRM_FRAMES}",
            ))
            if raw is not None:
                cv2.circle(frame, raw, 3, (0, 0, 255), -1, cv2.LINE_AA)
            if smoothed is not None:
                cv2.circle(frame, smoothed, 3, (0, 255, 0), -1, cv2.LINE_AA)
        if DEBUG_DRAW_ASSIST and self.canvas is not None:
            native_size = (self.canvas.image.shape[1], self.canvas.image.shape[0])
            for point in self.drawing_assistant.raw_points:
                cv2.circle(frame, self.layout.camera_to_window(point, native_size), 2, (70, 70, 255), -1, cv2.LINE_AA)
            for point in self.drawing_assistant.stabilized_points:
                cv2.circle(frame, self.layout.camera_to_window(point, native_size), 2, (70, 255, 120), -1, cv2.LINE_AA)
            lines.append(f"ASSIST {self.drawing_assistant.level.upper()} RAW {len(self.drawing_assistant.raw_points)} STABLE {len(self.drawing_assistant.stabilized_points)}")
        if DEBUG_SHAPE_RECOGNITION and self.pending_shape is not None and self.canvas is not None:
            result = self.pending_shape["result"]
            assert isinstance(result, ShapeRecognitionResult)
            native_size = (self.canvas.image.shape[1], self.canvas.image.shape[0])
            x, y, w, h = result.bounding_box
            p1 = self.layout.camera_to_window((x, y), native_size); p2 = self.layout.camera_to_window((x+w, y+h), native_size)
            cv2.rectangle(frame, p1, p2, (80, 220, 255), 1, cv2.LINE_AA)
            lines.append(f"SHAPE {result.shape_type.upper()} CONF {result.confidence:.3f}")
        lines.append(f"DWELL {self.toolbar_dwell_progress:.0%}")
        y = 370
        for line in lines:
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 245, 255), 1, cv2.LINE_AA)
            y += 18

    def _draw_camera_debug(self, frame: np.ndarray) -> None:
        status = self.camera_manager.get_camera_status()
        lines = (
            f"CAMERA {status['index']} BACKEND {status['backend']}",
            f"READ {status['read_success']} SHAPE {status['frame_shape']}",
            f"MEAN {status['mean_brightness']:.2f} STD {status['frame_stddev']:.2f}",
            f"FAILURES {status['consecutive_failures']} LAST {status['last_successful_frame_time']:.2f}",
        )
        y = 285
        for line in lines:
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 220, 255), 1, cv2.LINE_AA)
            y += 17


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Air Canvas")
    parser.add_argument("--camera", type=int, help="override automatic camera selection")
    parser.add_argument("--list-cameras", action="store_true", help="list working cameras and exit")
    args = parser.parse_args(argv)
    if args.list_cameras:
        manager = CameraManager()
        try:
            cameras = manager.discover_cameras()
            if not cameras:
                print("[CAMERA] No working cameras found")
                return 1
            preferred = manager.preferred_camera_info()
            if preferred is not None:
                print(f"[CAMERA] Preferred automatic selection: index {preferred.index}, {preferred.width}x{preferred.height}, backend {preferred.backend_name}")
            return 0
        finally:
            manager.release()
    return AirCanvasApp(camera_override=args.camera).run()


if __name__ == "__main__":
    raise SystemExit(main())

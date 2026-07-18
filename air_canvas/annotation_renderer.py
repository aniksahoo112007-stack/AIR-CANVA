"""Vector-backed desktop annotation layer with independent history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import (DESKTOP_ERASER_WIDTH, DESKTOP_HIGHLIGHTER_ALPHA, DESKTOP_HIGHLIGHTER_WIDTH,
                     DESKTOP_HISTORY_LIMIT, DESKTOP_PEN_WIDTH)


@dataclass
class AnnotationAction:
    tool: str
    points: list[tuple[int, int]] = field(default_factory=list)
    color: tuple[int, int, int] = (0, 0, 255)
    width: int = DESKTOP_PEN_WIDTH


class AnnotationRenderer:
    def __init__(self, width: int, height: int, history_limit: int = DESKTOP_HISTORY_LIMIT) -> None:
        self.width, self.height = width, height
        self.actions: list[AnnotationAction] = []
        self._undo: list[list[AnnotationAction]] = []
        self._redo: list[list[AnnotationAction]] = []
        self.history_limit = history_limit
        self.current: AnnotationAction | None = None
        self.preview: AnnotationAction | None = None
        self.layer = np.zeros((height, width, 4), np.uint8)
        self.dirty = True

    @staticmethod
    def _copy_actions(actions: list[AnnotationAction]) -> list[AnnotationAction]:
        return [AnnotationAction(a.tool, list(a.points), a.color, a.width) for a in actions]

    def resize(self, width: int, height: int) -> None:
        if (width, height) == (self.width, self.height): return
        sx, sy = width/max(1,self.width), height/max(1,self.height)
        for action in self.actions:
            action.points = [(round(x*sx), round(y*sy)) for x,y in action.points]
        self.width, self.height = width, height
        self.layer = np.zeros((height,width,4),np.uint8); self.dirty = True

    def _checkpoint(self) -> None:
        self._undo.append(self._copy_actions(self.actions)); self._undo = self._undo[-self.history_limit:]; self._redo.clear()

    def begin(self, tool: str, point: tuple[int,int], color: tuple[int,int,int], width: int | None = None) -> None:
        if self.current is not None: return
        widths = {"pen":DESKTOP_PEN_WIDTH,"highlighter":DESKTOP_HIGHLIGHTER_WIDTH,"eraser":DESKTOP_ERASER_WIDTH}
        self.current = AnnotationAction(tool,[point],color,width or widths.get(tool,DESKTOP_PEN_WIDTH))
        if tool in {"arrow","rectangle","circle"}: self.preview = self.current

    def append(self, point: tuple[int,int]) -> None:
        if self.current is None: return
        if self.current.points and np.linalg.norm(np.subtract(point,self.current.points[-1])) < 2: return
        if self.current.tool in {"arrow","rectangle","circle"}:
            self.current.points = [self.current.points[0],point]
        else: self.current.points.append(point)
        self.dirty = True

    def finish(self) -> bool:
        action, self.current, self.preview = self.current, None, None
        if action is None or len(action.points)<2: return False
        self._checkpoint(); self.actions.append(action); self.dirty = True; return True

    def cancel(self) -> None: self.current=None; self.preview=None; self.dirty=True

    def undo(self) -> bool:
        if not self._undo: return False
        self._redo.append(self._copy_actions(self.actions)); self.actions=self._undo.pop(); self.dirty=True; return True

    def redo(self) -> bool:
        if not self._redo: return False
        self._undo.append(self._copy_actions(self.actions)); self.actions=self._redo.pop(); self.dirty=True; return True

    def clear(self) -> bool:
        if not self.actions: return False
        self._checkpoint(); self.actions=[]; self.dirty=True; return True

    def render(self) -> np.ndarray:
        if self.dirty:
            self.layer.fill(0)
            for action in self.actions: self._draw_action(self.layer,action)
            self.dirty=False
        output=self.layer.copy()
        if self.current is not None: self._draw_action(output,self.current)
        return output

    @staticmethod
    def _draw_action(layer: np.ndarray, action: AnnotationAction) -> None:
        if len(action.points)<2:return
        color=(*action.color, round(255*DESKTOP_HIGHLIGHTER_ALPHA) if action.tool=="highlighter" else 255)
        if action.tool=="eraser": color=(0,0,0,0)
        pts=np.asarray(action.points,np.int32)
        if action.tool in {"pen","highlighter","eraser"}:
            for a,b in zip(pts[:-1],pts[1:]): cv2.line(layer,tuple(a),tuple(b),color,action.width,cv2.LINE_AA)
        elif action.tool=="rectangle": cv2.rectangle(layer,tuple(pts[0]),tuple(pts[-1]),color,action.width,cv2.LINE_AA)
        elif action.tool=="circle":
            center=tuple(((pts[0]+pts[-1])/2).astype(int)); radius=max(1,round(float(np.linalg.norm(pts[-1]-pts[0]))/2))
            cv2.circle(layer,center,radius,color,action.width,cv2.LINE_AA)
        elif action.tool=="arrow": cv2.arrowedLine(layer,tuple(pts[0]),tuple(pts[-1]),color,action.width,cv2.LINE_AA,tipLength=.18)

    def save_layer(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True,exist_ok=True)
        path=output_dir/datetime.now().strftime("annotation_layer_%Y%m%d_%H%M%S.png")
        if not cv2.imwrite(str(path),self.render()): raise OSError(f"Could not save {path}")
        return path

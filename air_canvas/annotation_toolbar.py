"""Compact mouse/hand hit-tested floating desktop palette."""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class ToolbarRect:
    x1:int;y1:int;x2:int;y2:int
    def contains(self,x:int,y:int)->bool:return self.x1<=x<=self.x2 and self.y1<=y<=self.y2


class AnnotationToolbar:
    tools=("pen","highlight","eraser","laser","arrow","rectangle","circle","undo","redo","clear","save","exit")
    def __init__(self)->None:
        self.position=[28,28];self.visible=True;self.rects:dict[str,ToolbarRect]={};self.dragging=False;self.drag_offset=(0,0)

    @property
    def bounds(self)->ToolbarRect:return ToolbarRect(self.position[0],self.position[1],self.position[0]+610,self.position[1]+76)
    def layout(self)->None:
        x,y=self.position; self.rects={}
        for i,tool in enumerate(self.tools):
            row,col=divmod(i,6); self.rects[tool]=ToolbarRect(x+8+col*99,y+8+row*32,x+98+col*99,y+36+row*32)
    def hit_test(self,x:int,y:int)->str|None:
        self.layout();return next((name for name,rect in self.rects.items() if rect.contains(x,y)),None)
    def render(self,layer:np.ndarray,active:str,input_mode:str)->None:
        if not self.visible:return
        self.layout(); x,y=self.position
        overlay=layer.copy();cv2.rectangle(overlay,(x,y),(x+610,y+76),(10,18,28,225),-1);cv2.addWeighted(overlay,.88,layer,.12,0,layer)
        cv2.rectangle(layer,(x,y),(x+610,y+76),(80,220,255,255),1,cv2.LINE_AA)
        for name,rect in self.rects.items():
            fill=(38,92,72,240) if name==active else (28,43,58,230)
            cv2.rectangle(layer,(rect.x1,rect.y1),(rect.x2,rect.y2),fill,-1);cv2.rectangle(layer,(rect.x1,rect.y1),(rect.x2,rect.y2),(70,120,145,255),1)
            label="HILITE" if name=="highlight" else name.upper(); cv2.putText(layer,label,(rect.x1+7,rect.y1+19),cv2.FONT_HERSHEY_SIMPLEX,.32,(230,240,248,255),1,cv2.LINE_AA)
        cv2.putText(layer,f"INPUT: {input_mode.upper()}",(x+430,y+72),cv2.FONT_HERSHEY_SIMPLEX,.3,(100,230,255,255),1,cv2.LINE_AA)

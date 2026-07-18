"""Clean perspective-aware palm HUD with a strict three-ring hierarchy."""
from __future__ import annotations
import math
import cv2
import numpy as np

from .config import (PALM_PRIMARY_RING_COUNT, PALM_RING_LINE_THICKNESS, PRIMARY_HUD_ACCENT,
                     PRIMARY_HUD_CORE, PRIMARY_HUD_LINE, SECONDARY_HUD_ACCENT,
                     SECONDARY_HUD_CORE, SECONDARY_HUD_LINE)
from .gesture_detector import Gesture
from .hand_effect_state import HandEffectState

PALM_INDICES=(0,5,9,13,17)

def palm_geometry(pixels:np.ndarray)->tuple[tuple[int,int],float]:
    center=tuple(np.rint(np.mean(pixels[list(PALM_INDICES)],axis=0)).astype(int))
    scale=float(np.linalg.norm(pixels[5].astype(float)-pixels[17].astype(float)))
    return center,max(18.0,min(105.0,scale*.62))

def oriented_geometry(pixels:np.ndarray,state:HandEffectState)->tuple[tuple[int,int],tuple[int,int],float]:
    center,_=palm_geometry(pixels)
    across=pixels[5].astype(float)-pixels[17].astype(float)
    length=max(1.0,float(np.linalg.norm(across)))
    angle=math.degrees(math.atan2(across[1],across[0]))
    height=max(12.0,float(np.linalg.norm(pixels[0].astype(float)-pixels[9].astype(float))))
    # The ellipse follows actual across-palm and wrist-to-MCP dimensions.
    axes=(max(15.0,min(82.0,length*.48)),max(10.0,min(62.0,height*.48)))
    state.update_geometry(angle,axes)
    smooth=state.palm_axes if state.palm_axes is not None else np.asarray(axes)
    return center,(max(6,int(round(smooth[0]))),max(5,int(round(smooth[1])))),state.palm_angle

def _arc(layer,center,axes,rotation,start,end,color,thickness=1):
    cv2.ellipse(layer,center,axes,rotation,start,end,color,thickness,cv2.LINE_AA)

def _segments(layer,center,axes,rotation,phase,color,count=7,span=.58):
    step=360.0/count
    for index in range(count):
        start=phase+index*step;_arc(layer,center,axes,rotation,start,start+step*span,color,PALM_RING_LINE_THICKNESS)

def _ticks(layer,center,axes,rotation,phase,color,count=8):
    theta=math.radians(rotation);cs,sn=math.cos(theta),math.sin(theta)
    for index in range(count):
        a=math.radians(phase+index*360/count);ca,sa=math.cos(a),math.sin(a)
        points=[]
        for extra in (0,3):
            x=(axes[0]+extra)*ca;y=(axes[1]+extra)*sa
            points.append((int(round(center[0]+x*cs-y*sn)),int(round(center[1]+x*sn+y*cs))))
        cv2.line(layer,points[0],points[1],color,1,cv2.LINE_AA)

def render_palm_hologram(sharp:np.ndarray,glow:np.ndarray,pixels:np.ndarray,state:HandEffectState,
                          _color:tuple[int,int,int],now:float,quality:str,primary:bool=True)->tuple[tuple[int,int],int]:
    center,axes,rotation=oriented_geometry(pixels,state)
    expansion=state.ring_expansion
    axes=(max(6,int(axes[0]*expansion)),max(5,int(axes[1]*expansion)))
    line,accent,core=(PRIMARY_HUD_LINE,PRIMARY_HUD_ACCENT,PRIMARY_HUD_CORE) if primary else (SECONDARY_HUD_LINE,SECONDARY_HUD_ACCENT,SECONDARY_HUD_CORE)
    gesture=state.current_gesture
    quality_count={'low':1,'medium':2,'high':3,'ultra':3}.get(quality,2)
    ring_count=min(PALM_PRIMARY_RING_COUNT,quality_count)
    if gesture is Gesture.IDLE:ring_count=0
    elif gesture is Gesture.DRAW:ring_count=min(1,ring_count)
    # Layer 1: a fresh local mask receives only compact glow primitives.
    cv2.ellipse(glow,center,(max(4,int(axes[0]*.34)),max(3,int(axes[1]*.34))),rotation,0,360,tuple(int(c*.35) for c in accent),-1,cv2.LINE_AA)
    # Layers 2/3: each sharp primary ring is rendered exactly once.
    if ring_count>=1:_segments(sharp,center,axes,rotation,state.rotation_angles[0],tuple(int(c*.70) for c in line),7)
    middle=(max(5,int(axes[0]*.72)),max(4,int(axes[1]*.72)))
    if ring_count>=2:
        _segments(sharp,center,middle,rotation,state.rotation_angles[1],tuple(int(c*.85) for c in accent),6,.38)
        if min(axes)>=13:_ticks(sharp,center,middle,rotation,state.rotation_angles[1],tuple(int(c*.58) for c in line),8)
    inner=(max(4,int(axes[0]*.37)),max(3,int(axes[1]*.37)))
    if ring_count>=3:_arc(sharp,center,inner,rotation,0,360,tuple(int(c*.90) for c in line),1)
    core_radius=max(2,min(5,int(min(axes)*.14+state.pulse_strength)))
    cv2.circle(sharp,center,core_radius,core,-1,cv2.LINE_AA)
    # Layer 5: one gesture accent, never another full duplicate ring.
    if gesture is Gesture.OPEN_PALM:
        _arc(sharp,center,(axes[0]+4,axes[1]+4),rotation,-90,-90+int(360*state.clear_progress),accent,1)
    elif gesture is Gesture.FIST:
        _arc(sharp,center,(max(5,int(axes[0]*.55)),max(4,int(axes[1]*.55))),rotation,25,335,accent,1)
    return center,max(axes)

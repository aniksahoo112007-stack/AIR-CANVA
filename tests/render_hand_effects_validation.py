"""Render deterministic gesture/resolution samples for visual QA."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from air_canvas.gesture_detector import Gesture
from air_canvas.hand_effects import HandEffectsRenderer
from test_hand_effects import effect_hand

def render(width:int,height:int)->np.ndarray:
    gestures=(Gesture.OPEN_PALM,Gesture.DRAW,Gesture.PINCH,Gesture.FIST,Gesture.IDLE)
    cells=[]
    for gesture in gestures:
        renderer=HandEffectsRenderer();hand=effect_hand(1,width//2,gesture,True)
        scale=height/500;hand.pixels[:]=np.rint(hand.pixels*scale).astype(np.int32)
        hand.index_tip=tuple(hand.pixels[8]);hand.thumb_tip=tuple(hand.pixels[4])
        for now in (1.0,1.1,1.2):renderer.update([hand],now,30)
        cell=np.full((height,width,3),(22,28,34),np.uint8);renderer.render(cell,[hand],1.2)
        cv2.putText(cell,gesture.name,(16,28),cv2.FONT_HERSHEY_DUPLEX,.55,(220,235,245),1,cv2.LINE_AA);cells.append(cell)
    return np.hstack(cells)

if __name__=='__main__':
    output=Path(__file__).resolve().parents[1]/'outputs'/'hand_effects_validation.png'
    top=render(640,480);bottom=cv2.resize(render(1280,720),(top.shape[1],480))
    cv2.imwrite(str(output),np.vstack((top,bottom)));print(output)

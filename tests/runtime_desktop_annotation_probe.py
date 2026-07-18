"""Real Windows smoke probe for the production desktop overlay controller."""
from __future__ import annotations
import ctypes, time
from types import SimpleNamespace
import numpy as np

from air_canvas.annotation_controller import AnnotationController
from air_canvas.gesture_detector import Gesture

def hand(x:float,y:float,gesture:Gesture=Gesture.DRAW):
    landmarks=np.zeros((21,3),np.float32);landmarks[8,:2]=(x,y)
    return SimpleNamespace(is_primary=True,smoothed_landmarks=landmarks,gesture=gesture,pinch_active=False)

def main()->int:
    controller=AnnotationController(0);controller.enter();overlay_handle=controller.overlay.hwnd
    frame=np.full((480,640,3),(35,55,75),np.uint8);now=time.monotonic()
    try:
        controller.update([hand(.25,.3)],frame,30,now)
        controller.update([hand(.55,.55)],frame,30,now+.05)
        controller.update([],frame,30,now+.1)
        assert len(controller.renderer.actions)==1
        controller.toggle_laser();controller.update([hand(.6,.5,Gesture.IDLE)],frame,30,now+.15)
        assert len(controller.renderer.actions)==1
        controller.toggle_input();assert ctypes.windll.user32.GetWindowLongW(overlay_handle,-20)&0x20
        paths=controller.save();assert all(path.is_file() for path in paths)
    finally:controller.close()
    assert not ctypes.windll.user32.IsWindow(overlay_handle)
    print('PASS real overlay/draw/laser/click-through/export/cleanup')
    return 0

if __name__=='__main__':raise SystemExit(main())

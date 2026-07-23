"""Isolated state, gestures, rendering and export for desktop annotation."""
from __future__ import annotations
from datetime import datetime
import time
import cv2
import numpy as np

from .annotation_renderer import AnnotationRenderer
from .annotation_toolbar import AnnotationToolbar
from .config import COLORS, DESKTOP_CAMERA_PREVIEW_DEFAULT, DESKTOP_DEFAULT_COLOR, DESKTOP_DEFAULT_TOOL, DESKTOP_OVERLAY_CLICK_THROUGH_DEFAULT, DESKTOP_TOOLBAR_VISIBLE_DEFAULT, OUTPUT_DIR
from .desktop_mapper import DesktopMapper
from .desktop_overlay import DesktopOverlay
from .gesture_detector import Gesture
from .monitor_manager import MonitorManager
from .pointer_renderer import PointerRenderer

class AnnotationController:
    def __init__(self, camera_index:int=0)->None:
        self.monitors=MonitorManager(); self.monitor=self.monitors.active
        self.mapper=DesktopMapper(); self.mapper.set_context(camera_index,self.monitor.index)
        self.renderer=AnnotationRenderer(self.monitor.width,self.monitor.height); self.pointer=PointerRenderer(); self.toolbar=AnnotationToolbar()
        self.toolbar.visible=DESKTOP_TOOLBAR_VISIBLE_DEFAULT; self.preview_visible=DESKTOP_CAMERA_PREVIEW_DEFAULT
        self.tool=DESKTOP_DEFAULT_TOOL; self.color=DESKTOP_DEFAULT_COLOR; self.click_through=DESKTOP_OVERLAY_CLICK_THROUGH_DEFAULT; self.laser_mode=False
        self.overlay:DesktopOverlay|None=None; self.active=False; self.exit_requested=False; self.message=''; self.message_until=0.0
        self._drawing=False; self._toolbar_down=None; self._clear_armed_until=0.0
        self._calibration:list[tuple[float,float]]=[]; self._calibration_started=None; self._calibration_point=None
        self._paused=False;self._secondary_open_latched=False;self._secondary_wrist=[];self._secondary_action_at=0.0;self._both_open_started=None;self._both_open_latched=False

    def notify(self,text:str,duration:float=1.5)->None:self.message=text;self.message_until=time.monotonic()+duration
    def enter(self)->None:
        if self.active:return
        try:
            self.exit_requested=False; self.overlay=DesktopOverlay(self.monitor,self.click_through); self.active=True; self.notify('DESKTOP ANNOTATION ON')
        except Exception:
            self.close(); raise
    def close(self)->None:
        self.renderer.cancel();self.pointer.clear();self.active=False
        if self.overlay is not None:self.overlay.close();self.overlay=None
    def toggle_input(self)->None:
        self.click_through=not self.click_through
        if self.overlay:self.overlay.set_click_through(self.click_through)
        self.notify('INPUT: CLICK-THROUGH' if self.click_through else 'INPUT: DRAWING')
    def toggle_laser(self)->None:self.laser_mode=not self.laser_mode;self.renderer.finish();self.notify(f"LASER POINTER {'ON' if self.laser_mode else 'OFF'}")
    def cycle_monitor(self)->None:
        old=self.overlay; self.monitor=self.monitors.cycle(); self.mapper.set_context(self.mapper.camera_index,self.monitor.index)
        self.renderer.resize(self.monitor.width,self.monitor.height)
        if old: old.close(); self.overlay=DesktopOverlay(self.monitor,self.click_through)
        self.notify(f'MONITOR {self.monitor.index+1} ACTIVE')
    def set_camera(self,index:int)->None:self.mapper.set_context(index,self.monitor.index)

    def update(self,hands:list,frame:np.ndarray|None,fps:float,now:float)->None:
        if not self.active or self.overlay is None:return
        for event,x,y in self.overlay.poll_events():self._mouse(event,x,y)
        if not self.active:
            self.close(); return
        primary=next((h for h in hands if h.is_primary),None)
        self._update_multi_hand(hands,now)
        if primary is None:
            self.renderer.finish();self._drawing=False;self.pointer.update(None,now)
        else:
            raw=primary.smoothed_landmarks[8,:2]; absolute=self.mapper.camera_to_desktop((float(raw[0]),float(raw[1])),self.monitor)
            point=(absolute[0]-self.monitor.left,absolute[1]-self.monitor.top)
            paused=primary.gesture is Gesture.FIST or (len(hands)>=2 and all(h.gesture is Gesture.FIST for h in hands[:2]))
            if paused != self._paused:self._paused=paused;self.notify('OVERLAY PAUSED' if paused else 'OVERLAY ACTIVE')
            laser=self.laser_mode or self.tool=='laser'
            self.pointer.update(point if laser and not paused else None,now)
            if not laser and not paused and primary.gesture is Gesture.DRAW:
                render_tool = "highlighter" if self.tool == "highlight" else self.tool
                if not self._drawing:self.renderer.begin(render_tool,point,COLORS.get(self.color,(0,0,255)));self._drawing=True
                else:self.renderer.append(point)
            else:
                if self._drawing:self.renderer.finish();self._drawing=False
            if primary.gesture in {Gesture.SELECT,Gesture.PINCH} and primary.pinch_active:
                action=self.toolbar.hit_test(*point) if self.toolbar.visible else None
                if action and action!=self._toolbar_down:self._activate(action)
                self._toolbar_down=action
            elif not primary.pinch_active:self._toolbar_down=None
            if self._calibration_started is not None:self._update_calibration((float(raw[0]),float(raw[1])),now)
        layer=self.renderer.render(); self.pointer.render(layer,(self.monitor.left,self.monitor.top),now)
        self.toolbar.render(layer,self.tool,'click-through' if self.click_through else 'drawing')
        self._render_preview(layer,frame,fps,len(hands)); self._render_status(layer,now)
        self.overlay.update(layer)

    def _update_multi_hand(self,hands:list,now:float)->None:
        secondary=next((h for h in hands if h.is_secondary),None)
        secondary_open=secondary is not None and secondary.gesture is Gesture.OPEN_PALM
        if secondary_open and not self._secondary_open_latched:
            self.toolbar.visible=not self.toolbar.visible;self._secondary_open_latched=True;self.notify(f"TOOLBAR {'ON' if self.toolbar.visible else 'OFF'}")
        elif not secondary_open:self._secondary_open_latched=False
        if secondary is not None:
            self._secondary_wrist.append((now,float(secondary.smoothed_landmarks[0,0])));self._secondary_wrist=[p for p in self._secondary_wrist if now-p[0]<.65]
            if len(self._secondary_wrist)>2 and now-self._secondary_action_at>.9:
                dx=self._secondary_wrist[-1][1]-self._secondary_wrist[0][1]
                if abs(dx)>.22:
                    (self.renderer.redo() if dx>0 else self.renderer.undo());self.notify('REDO' if dx>0 else 'UNDO');self._secondary_action_at=now;self._secondary_wrist=[]
        else:self._secondary_wrist=[]
        both_open=len(hands)>=2 and all(h.gesture is Gesture.OPEN_PALM for h in hands[:2])
        if both_open and self._both_open_started is None:self._both_open_started=now;self.notify('HOLD TO CLEAR 2s',2)
        if both_open and not self._both_open_latched and now-float(self._both_open_started or now)>=2.0:self.renderer.clear();self._both_open_latched=True;self.notify('ANNOTATIONS CLEARED')
        if not both_open:self._both_open_started=None;self._both_open_latched=False

    def _mouse(self,event:str,x:int,y:int)->None:
        if event=='down':
            action=self.toolbar.hit_test(x,y) if self.toolbar.visible else None
            if action:self._toolbar_down=action
            elif self.toolbar.visible and self.toolbar.bounds.contains(x,y):self.toolbar.dragging=True;self.toolbar.drag_offset=(x-self.toolbar.position[0],y-self.toolbar.position[1])
        elif event=='move' and self.toolbar.dragging:
            ox,oy=self.toolbar.drag_offset;self.toolbar.position=[max(0,min(self.monitor.width-610,x-ox)),max(0,min(self.monitor.height-76,y-oy))]
        elif event=='up':
            action=self.toolbar.hit_test(x,y) if self.toolbar.visible else None
            if action and action==self._toolbar_down:self._activate(action)
            self._toolbar_down=None;self.toolbar.dragging=False

    def _activate(self,action:str)->None:
        if action in {'pen','highlight','eraser','laser','arrow','rectangle','circle'}:self.tool=action;self.laser_mode=action=='laser';self.notify(f'TOOL: {action.upper()}')
        elif action=='undo':self.renderer.undo();self.notify('UNDO')
        elif action=='redo':self.renderer.redo();self.notify('REDO')
        elif action=='clear':self.request_clear()
        elif action=='save':self.save()
        elif action=='exit':self.notify('DESKTOP ANNOTATION OFF');self.exit_requested=True;self.active=False

    def request_clear(self)->None:
        now=time.monotonic()
        if now<=self._clear_armed_until:self.renderer.clear();self._clear_armed_until=0;self.notify('ANNOTATIONS CLEARED')
        else:self._clear_armed_until=now+2.0;self.notify('CLEAR? PRESS AGAIN',2.0)
    def start_calibration(self)->None:self._calibration=[];self._calibration_started=time.monotonic();self._calibration_point=None;self.notify('CALIBRATION 1/4')
    def _update_calibration(self,p:tuple[float,float],now:float)->None:
        if self._calibration_point is None:self._calibration_point=p;self._calibration_started=now;return
        if now-float(self._calibration_started)<0.65:return
        self._calibration.append(p);self._calibration_point=None
        if len(self._calibration)==4:self.mapper.calibrate(self._calibration);self._calibration_started=None;self.notify('CALIBRATION COMPLETE',2)
        else:self.notify(f'CALIBRATION {len(self._calibration)+1}/4')

    def save(self)->tuple[object,object]:
        if self.overlay:self.overlay.hide()
        try:
            from PIL import Image,ImageGrab
            stamp=datetime.now().strftime('%Y%m%d_%H%M%S');OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
            shot=ImageGrab.grab(bbox=(self.monitor.left,self.monitor.top,self.monitor.right,self.monitor.bottom),all_screens=True).convert('RGBA')
            layer_bgra=self.renderer.render(); layer=Image.fromarray(cv2.cvtColor(layer_bgra,cv2.COLOR_BGRA2RGBA))
            screen_path=OUTPUT_DIR/f'desktop_annotation_{stamp}.png'; layer_path=OUTPUT_DIR/f'annotation_layer_{stamp}.png'
            Image.alpha_composite(shot,layer).convert('RGB').save(screen_path);layer.save(layer_path);self.notify('ANNOTATED SCREEN SAVED');return screen_path,layer_path
        except Exception:self.notify('SAVE FAILED');raise
        finally:
            if self.overlay:self.overlay.show()

    def _render_preview(self,layer,frame,fps,count):
        if not self.preview_visible or frame is None:return
        thumb=cv2.resize(frame,(256,144)); thumb=np.dstack((thumb,np.full(thumb.shape[:2],235,np.uint8)));x=self.monitor.width-272;y=18
        layer[y:y+144,x:x+256]=thumb;cv2.rectangle(layer,(x,y),(x+256,y+144),(80,220,255,255),1);cv2.putText(layer,f'CAM {self.mapper.camera_index} | {fps:.0f} FPS | HANDS {count}',(x+7,y+136),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255,255),1,cv2.LINE_AA)
    def _render_status(self,layer,now):
        if self._calibration_started is not None:
            corners=[(35,35),(self.monitor.width-35,35),(self.monitor.width-35,self.monitor.height-35),(35,self.monitor.height-35)];p=corners[len(self._calibration)];cv2.drawMarker(layer,p,(80,240,255,255),cv2.MARKER_CROSS,30,3)
        if now<self.message_until:
            size=cv2.getTextSize(self.message,cv2.FONT_HERSHEY_DUPLEX,.65,1)[0];x=(self.monitor.width-size[0])//2;cv2.rectangle(layer,(x-16,self.monitor.height-64),(x+size[0]+16,self.monitor.height-25),(8,20,30,220),-1);cv2.putText(layer,self.message,(x,self.monitor.height-38),cv2.FONT_HERSHEY_DUPLEX,.65,(100,235,255,255),1,cv2.LINE_AA)

"""Scoped Windows global hotkey registration and polling."""
from __future__ import annotations
import ctypes, sys
from ctypes import wintypes

MOD_CONTROL=2; MOD_ALT=1; WM_HOTKEY=0x0312
KEYS={'preview':(0,0x75),'input':(0,0x76),'desktop':(0,0x77),'laser':(0,0x78),'calibrate':(0,0x79),'save':(0,0x7B),'toolbar':(0,0x09),'exit':(0,0x1B),'undo':(MOD_CONTROL,ord('Z')),'redo':(MOD_CONTROL,ord('Y')),'clear':(0,0x2E),'monitor':(MOD_CONTROL|MOD_ALT,ord('M')),'camera':(MOD_CONTROL|MOD_ALT,ord('C'))}

class HotkeyManager:
    def __init__(self)->None:
        if sys.platform!='win32': raise OSError('Global hotkeys require Windows')
        self.user32=ctypes.windll.user32; self.registered:dict[int,str]={}; self._next=0xA100
        self.fallback:set[str]=set(); self._fallback_down:set[str]=set()
    def register(self,name:str)->bool:
        if name in self.registered.values():return True
        mods,key=KEYS[name]; ident=self._next; self._next+=1
        if not self.user32.RegisterHotKey(None,ident,mods,key):
            self.fallback.add(name); return True
        self.registered[ident]=name; return True
    def register_active(self)->list[str]:
        return [name for name in KEYS if name!='desktop' and not self.register(name)]
    def unregister_active(self)->None:
        for ident,name in list(self.registered.items()):
            if name!='desktop':self.user32.UnregisterHotKey(None,ident);self.registered.pop(ident,None)
        self.fallback.intersection_update({'desktop'}); self._fallback_down.intersection_update({'desktop'})
    def poll(self)->list[str]:
        msg=wintypes.MSG(); result=[]
        while self.user32.PeekMessageW(ctypes.byref(msg),None,WM_HOTKEY,WM_HOTKEY,1):
            name=self.registered.get(int(msg.wParam))
            if name:result.append(name)
        for name in self.fallback:
            mods,key=KEYS[name]
            pressed=bool(self.user32.GetAsyncKeyState(key)&0x8000)
            pressed=pressed and (not mods&MOD_CONTROL or bool(self.user32.GetAsyncKeyState(0x11)&0x8000))
            pressed=pressed and (not mods&MOD_ALT or bool(self.user32.GetAsyncKeyState(0x12)&0x8000))
            if pressed and name not in self._fallback_down:result.append(name)
            if pressed:self._fallback_down.add(name)
            else:self._fallback_down.discard(name)
        return result
    def close(self)->None:
        for ident in list(self.registered):self.user32.UnregisterHotKey(None,ident)
        self.registered.clear()
        self.fallback.clear();self._fallback_down.clear()

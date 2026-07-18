"""DPI-aware, non-activating Win32 per-pixel-alpha desktop overlay."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
from collections import deque
import numpy as np

from .monitor_manager import MonitorBounds

WM_DESTROY=0x0002; WM_MOUSEMOVE=0x0200; WM_LBUTTONDOWN=0x0201; WM_LBUTTONUP=0x0202
WS_POPUP=0x80000000; WS_EX_LAYERED=0x80000; WS_EX_TRANSPARENT=0x20
WS_EX_TOOLWINDOW=0x80; WS_EX_TOPMOST=0x8; WS_EX_NOACTIVATE=0x08000000
GWL_EXSTYLE=-20; SW_SHOWNA=8; SW_HIDE=0; HWND_TOPMOST=-1
SWP_NOACTIVATE=0x0010; ULW_ALPHA=2; AC_SRC_OVER=0; AC_SRC_ALPHA=1

class POINT(ctypes.Structure): _fields_=(('x',wintypes.LONG),('y',wintypes.LONG))
class SIZE(ctypes.Structure): _fields_=(('cx',wintypes.LONG),('cy',wintypes.LONG))
class BLENDFUNCTION(ctypes.Structure): _fields_=(('BlendOp',ctypes.c_byte),('BlendFlags',ctypes.c_byte),('SourceConstantAlpha',ctypes.c_byte),('AlphaFormat',ctypes.c_byte))
class BITMAPINFOHEADER(ctypes.Structure):
    _fields_=(('biSize',wintypes.DWORD),('biWidth',wintypes.LONG),('biHeight',wintypes.LONG),('biPlanes',wintypes.WORD),('biBitCount',wintypes.WORD),('biCompression',wintypes.DWORD),('biSizeImage',wintypes.DWORD),('biXPelsPerMeter',wintypes.LONG),('biYPelsPerMeter',wintypes.LONG),('biClrUsed',wintypes.DWORD),('biClrImportant',wintypes.DWORD))
class BITMAPINFO(ctypes.Structure): _fields_=(('bmiHeader',BITMAPINFOHEADER),('bmiColors',wintypes.DWORD*3))

class DesktopOverlay:
    def __init__(self, monitor: MonitorBounds, click_through: bool=False) -> None:
        if sys.platform != 'win32': raise OSError('Desktop overlay requires Windows')
        self.user32=ctypes.windll.user32; self.gdi32=ctypes.windll.gdi32
        self.user32.DefWindowProcW.argtypes=(wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM); self.user32.DefWindowProcW.restype=wintypes.LPARAM
        self.user32.CreateWindowExW.restype=wintypes.HWND
        self.user32.GetDC.argtypes=(wintypes.HWND,); self.user32.GetDC.restype=wintypes.HDC
        self.user32.ReleaseDC.argtypes=(wintypes.HWND,wintypes.HDC)
        self.gdi32.CreateCompatibleDC.argtypes=(wintypes.HDC,); self.gdi32.CreateCompatibleDC.restype=wintypes.HDC
        self.gdi32.CreateDIBSection.argtypes=(wintypes.HDC,ctypes.POINTER(BITMAPINFO),wintypes.UINT,ctypes.POINTER(ctypes.c_void_p),wintypes.HANDLE,wintypes.DWORD); self.gdi32.CreateDIBSection.restype=wintypes.HBITMAP
        self.gdi32.SelectObject.argtypes=(wintypes.HDC,wintypes.HANDLE); self.gdi32.SelectObject.restype=wintypes.HANDLE
        self.gdi32.DeleteDC.argtypes=(wintypes.HDC,); self.gdi32.DeleteObject.argtypes=(wintypes.HANDLE,)
        self.user32.UpdateLayeredWindow.argtypes=(wintypes.HWND,wintypes.HDC,ctypes.POINTER(POINT),ctypes.POINTER(SIZE),wintypes.HDC,ctypes.POINTER(POINT),wintypes.COLORREF,ctypes.POINTER(BLENDFUNCTION),wintypes.DWORD)
        self.monitor=monitor; self.hwnd=0; self.memdc=0; self.bitmap=0; self.old_bitmap=0; self.bits=ctypes.c_void_p()
        self.events: deque[tuple[str,int,int]]=deque(); self._class_name=f'AirCanvasOverlay_{id(self)}'; self._click_through=click_through
        WNDPROC=ctypes.WINFUNCTYPE(ctypes.c_longlong,wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM)
        self._wndproc=WNDPROC(self._window_proc)
        class WNDCLASS(ctypes.Structure):
            _fields_=(('style',wintypes.UINT),('lpfnWndProc',WNDPROC),('cbClsExtra',ctypes.c_int),('cbWndExtra',ctypes.c_int),('hInstance',wintypes.HINSTANCE),('hIcon',wintypes.HICON),('hCursor',wintypes.HANDLE),('hbrBackground',wintypes.HBRUSH),('lpszMenuName',wintypes.LPCWSTR),('lpszClassName',wintypes.LPCWSTR))
        self._instance=ctypes.windll.kernel32.GetModuleHandleW(None)
        wc=WNDCLASS(); wc.lpfnWndProc=self._wndproc; wc.hInstance=self._instance; wc.lpszClassName=self._class_name; wc.hCursor=self.user32.LoadCursorW(None,32512)
        if not self.user32.RegisterClassW(ctypes.byref(wc)): raise ctypes.WinError()
        ex=WS_EX_LAYERED|WS_EX_TOOLWINDOW|WS_EX_TOPMOST|WS_EX_NOACTIVATE|(WS_EX_TRANSPARENT if click_through else 0)
        self.hwnd=self.user32.CreateWindowExW(ex,self._class_name,'Air Canvas Desktop Annotation',WS_POPUP,monitor.left,monitor.top,monitor.width,monitor.height,None,None,self._instance,None)
        if not self.hwnd: self.user32.UnregisterClassW(self._class_name,self._instance); raise ctypes.WinError()
        self._create_surface(); self.user32.ShowWindow(self.hwnd,SW_SHOWNA)

    def _window_proc(self, hwnd, msg, wp, lp):
        if msg in (WM_MOUSEMOVE,WM_LBUTTONDOWN,WM_LBUTTONUP):
            x=ctypes.c_short(lp & 0xffff).value; y=ctypes.c_short((lp>>16)&0xffff).value
            self.events.append(({WM_MOUSEMOVE:'move',WM_LBUTTONDOWN:'down',WM_LBUTTONUP:'up'}[msg],x,y))
            return 0
        if msg==WM_DESTROY:return 0
        return self.user32.DefWindowProcW(hwnd,msg,wp,lp)

    def _create_surface(self):
        screen=self.user32.GetDC(None); self.memdc=self.gdi32.CreateCompatibleDC(screen)
        info=BITMAPINFO(); info.bmiHeader.biSize=ctypes.sizeof(BITMAPINFOHEADER); info.bmiHeader.biWidth=self.monitor.width; info.bmiHeader.biHeight=-self.monitor.height; info.bmiHeader.biPlanes=1; info.bmiHeader.biBitCount=32
        self.bitmap=self.gdi32.CreateDIBSection(screen,ctypes.byref(info),0,ctypes.byref(self.bits),None,0)
        self.user32.ReleaseDC(None,screen)
        if not self.memdc or not self.bitmap: raise ctypes.WinError()
        self.old_bitmap=self.gdi32.SelectObject(self.memdc,self.bitmap)

    def update(self, bgra: np.ndarray) -> None:
        if not self.hwnd:return
        expected=(self.monitor.height,self.monitor.width,4)
        if bgra.shape!=expected: raise ValueError(f'Overlay frame {bgra.shape} does not match {expected}')
        data=np.ascontiguousarray(bgra,np.uint8).copy()
        alpha=data[:,:,3:4].astype(np.uint16); data[:,:,:3]=((data[:,:,:3].astype(np.uint16)*alpha+127)//255).astype(np.uint8)
        ctypes.memmove(self.bits,data.ctypes.data,data.nbytes)
        dst=POINT(self.monitor.left,self.monitor.top); src=POINT(0,0); size=SIZE(self.monitor.width,self.monitor.height); blend=BLENDFUNCTION(AC_SRC_OVER,0,255,AC_SRC_ALPHA)
        if not self.user32.UpdateLayeredWindow(self.hwnd,None,ctypes.byref(dst),ctypes.byref(size),self.memdc,ctypes.byref(src),0,ctypes.byref(blend),ULW_ALPHA): raise ctypes.WinError()

    def poll_events(self)->list[tuple[str,int,int]]:
        msg=wintypes.MSG()
        while self.user32.PeekMessageW(ctypes.byref(msg),self.hwnd,0,0,1): self.user32.TranslateMessage(ctypes.byref(msg)); self.user32.DispatchMessageW(ctypes.byref(msg))
        result=list(self.events); self.events.clear(); return result

    def set_click_through(self, enabled: bool)->None:
        self._click_through=enabled; style=self.user32.GetWindowLongW(self.hwnd,GWL_EXSTYLE)
        self.user32.SetWindowLongW(self.hwnd,GWL_EXSTYLE,style|WS_EX_TRANSPARENT if enabled else style & ~WS_EX_TRANSPARENT)

    def show(self)->None:self.user32.ShowWindow(self.hwnd,SW_SHOWNA)
    def hide(self)->None:self.user32.ShowWindow(self.hwnd,SW_HIDE)
    def close(self)->None:
        if self.memdc:
            if self.old_bitmap:self.gdi32.SelectObject(self.memdc,self.old_bitmap)
            if self.bitmap:self.gdi32.DeleteObject(self.bitmap)
            self.gdi32.DeleteDC(self.memdc); self.memdc=0
        if self.hwnd:self.user32.DestroyWindow(self.hwnd); self.hwnd=0
        if getattr(self,'_class_name',None): self.user32.UnregisterClassW(self._class_name,self._instance)
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

"""DPI-aware Windows monitor discovery and selection."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import sys
from ctypes import wintypes


@dataclass(frozen=True)
class MonitorBounds:
    index: int
    left: int
    top: int
    width: int
    height: int
    primary: bool = False

    @property
    def right(self) -> int: return self.left + self.width
    @property
    def bottom(self) -> int: return self.top + self.height


class MonitorManager:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Desktop annotation is supported only on Windows")
        self._set_dpi_awareness()
        self.monitors = self._enumerate()
        if not self.monitors:
            raise OSError("Windows did not report any display monitors")
        self.active_index = next((m.index for m in self.monitors if m.primary), 0)

    @staticmethod
    def _set_dpi_awareness() -> None:
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            try: ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except (AttributeError, OSError): pass

    @staticmethod
    def _enumerate() -> list[MonitorBounds]:
        class MONITORINFO(ctypes.Structure):
            _fields_ = (("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD))
        found: list[MonitorBounds] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
        def callback(handle: int, _dc: int, _rect: object, _data: int) -> bool:
            info = MONITORINFO(); info.cbSize = ctypes.sizeof(info)
            if ctypes.windll.user32.GetMonitorInfoW(handle, ctypes.byref(info)):
                rect = info.rcMonitor
                found.append(MonitorBounds(len(found), rect.left, rect.top, rect.right-rect.left, rect.bottom-rect.top, bool(info.dwFlags & 1)))
            return True
        ctypes.windll.user32.EnumDisplayMonitors(0, None, callback_type(callback), 0)
        return found

    @property
    def active(self) -> MonitorBounds: return self.monitors[self.active_index]

    def set_monitor(self, index: int) -> MonitorBounds:
        self.active_index = index % len(self.monitors)
        return self.active

    def cycle(self) -> MonitorBounds: return self.set_monitor(self.active_index + 1)

    def monitor_for_point(self, x: int, y: int) -> MonitorBounds:
        return next((m for m in self.monitors if m.left <= x < m.right and m.top <= y < m.bottom), self.active)

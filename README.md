# Air Canvas

## Windows Desktop Annotation

On Windows 10/11, select **DESKTOP** or press `F8` to place a transparent annotation layer above the active monitor while camera hand tracking continues in the background. The desktop layer has independent drawing/history state and never changes the normal Air Canvas canvas.

Shortcuts: `F6` preview, `F7` drawing/click-through, `F8` desktop mode, `F9` laser, `F10` calibration, `F12` screenshot, `Tab` palette, `Escape` exit, `Ctrl+Z`/`Ctrl+Y` history, `Delete` clear confirmation, `Ctrl+Alt+M` monitor, and `Ctrl+Alt+C` camera cycle.

Calibration is saved as numeric normalized bounds in `desktop_calibration.json`, separately for each camera/monitor pair. No camera frames are stored. Exports are written to `outputs/` as an annotated desktop PNG and an annotation-only transparent PNG.

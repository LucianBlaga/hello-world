"""Video capture and pan/tilt/zoom control for the OBSBOT Tiny 2 (any UVC PTZ camera).

The Tiny 2 exposes its gimbal and zoom as standard UVC camera controls, so no
vendor SDK is required:
  * Linux  : v4l2 controls pan_absolute / tilt_absolute / zoom_absolute (arc-seconds)
  * Windows: DirectShow IAMCameraControl, reachable through OpenCV CAP_PROP_PAN/TILT/ZOOM

IMPORTANT: turn OFF the camera's own AI tracking in OBSBOT Center (or with the
remote/gesture) - otherwise the camera and this program fight over the gimbal.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import threading
from collections import deque
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)


def windows_camera_names() -> list[str]:
    """DirectShow camera names in index order (same order OpenCV's CAP_DSHOW uses)."""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        log.warning("Camera names unavailable: python -m pip install pygrabber")
        return []
    try:
        return list(FilterGraph().get_input_devices())
    except Exception as exc:
        log.warning("Could not list cameras: %s", exc)
        return []


def resolve_device(device):
    """"0" -> 0; on Windows a name such as "OBSBOT Tiny 2" -> its index."""
    if isinstance(device, str) and device.strip().isdigit():
        return int(device)
    if isinstance(device, str) and platform.system() == "Windows":
        names = windows_camera_names()
        matches = [i for i, n in enumerate(names) if device.lower() in n.lower()]
        # Prefer the real camera over OBSBOT Center's virtual camera.
        real = [i for i in matches if "virtual" not in names[i].lower()]
        if real or matches:
            idx = (real or matches)[0]
            log.info("Using camera %d: %s", idx, names[idx])
            return idx
        raise RuntimeError(f"No camera named like {device!r}. Found: {names or 'none'}")
    return device


def _open_capture(cfg: CameraConfig) -> cv2.VideoCapture:
    device = resolve_device(cfg.device)
    system = platform.system()
    if isinstance(device, int) and system == "Windows":
        cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    elif system == "Linux":
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {cfg.device!r}")
    # MJPG is required for 1080p/4K at 30 fps over USB.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Camera opened at %dx%d @ %.0f fps", w, h, cap.get(cv2.CAP_PROP_FPS))
    if (w, h) != (cfg.width, cfg.height):
        log.warning("Camera refused %dx%d, using %dx%d", cfg.width, cfg.height, w, h)
    return cap


class FrameSource:
    """Reads frames on a background thread and always hands out the newest one.

    Works with a live camera or a video file (for testing without hardware).
    """

    def __init__(self, cfg: CameraConfig, video_file: str | None = None):
        self.lock = threading.Lock()          # also guards property writes (PTZ)
        self.is_file = video_file is not None
        if video_file:
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video {video_file!r}")
            self._file_delay = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        else:
            self.cap = _open_capture(cfg)
            self._file_delay = 0.0
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._seq = 0
        self._running = True
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="frames", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        failures = 0
        while self._running:
            with self.lock:
                ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    self._running = False
                    break
                failures += 1
                if failures > 50:
                    log.error("Camera stopped delivering frames")
                    self._running = False
                    break
                time.sleep(0.05)
                continue
            failures = 0
            with self._cond:
                self._frame, self._stamp = frame, time.time()
                self._seq += 1
                self._cond.notify_all()
            if self._file_delay:
                time.sleep(self._file_delay)
        with self._cond:
            self._cond.notify_all()

    @property
    def running(self) -> bool:
        return self._running

    def next(self, last_seq: int, timeout: float = 1.0):
        """Block until a frame newer than `last_seq` exists. Returns (seq, stamp, frame)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or not self._running, timeout)
            return self._seq, self._stamp, self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


# --------------------------------------------------------------------------- PTZ


@dataclass
class PTZState:
    pan: float = 0.0    # degrees, + = right
    tilt: float = 0.0   # degrees, + = up
    zoom: float = 1.0   # ratio, 1.0 = widest


class PTZ:
    """Base class: keeps the commanded state and converts it to raw driver units."""

    default_raw_pan = (-130, 130)
    default_raw_tilt = (-90, 90)
    default_raw_zoom = (0, 100)

    def __init__(self, cfg: CameraConfig, clock=time.time):
        self.cfg = cfg
        self.clock = clock
        self.state = PTZState(cfg.home_pan, cfg.home_tilt, cfg.home_zoom)   # last command
        self.est = PTZState(cfg.home_pan, cfg.home_tilt, cfg.home_zoom)     # where it really points
        self._est_t = clock()
        self._cmds: deque = deque([(self._est_t, self.state)])
        self._last_cmd = 0.0
        self.moving_until = 0.0     # frames before this time may be motion-blurred

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _map(value, src, dst):
        (a, b), (c, d) = src, dst
        t = (value - a) / (b - a) if b != a else 0.0
        return c + t * (d - c)

    def clamp(self, st: PTZState) -> PTZState:
        c = self.cfg
        return PTZState(
            pan=float(np.clip(st.pan, *c.pan_limits)),
            tilt=float(np.clip(st.tilt, *c.tilt_limits)),
            zoom=float(np.clip(st.zoom, 1.0, c.max_zoom_ratio)),
        )

    def to_raw(self, st: PTZState) -> tuple[int, int, int]:
        c = self.cfg
        pan = -st.pan if c.invert_pan else st.pan
        tilt = -st.tilt if c.invert_tilt else st.tilt
        raw_pan = self._map(pan, c.pan_limits, c.raw_pan_range or self.default_raw_pan)
        raw_tilt = self._map(tilt, c.tilt_limits, c.raw_tilt_range or self.default_raw_tilt)
        raw_zoom = self._map(st.zoom, (1.0, c.max_zoom_ratio), c.raw_zoom_range or self.default_raw_zoom)
        return int(round(raw_pan)), int(round(raw_tilt)), int(round(raw_zoom))

    def update_estimate(self, now: float | None = None) -> PTZState:
        """Advance the model of the real gimbal: it starts moving `ptz_latency_s`
        after a command, turns at most `pan_speed_dps` and zooms at `zoom_speed`.

        Image positions must be interpreted with where the camera REALLY points;
        using the last command instead makes fast targets run away.
        """
        c = self.cfg
        now = self.clock() if now is None else now
        dt = max(0.0, now - self._est_t)
        self._est_t = now
        target = self._cmds[0][1]
        for t_cmd, st in self._cmds:
            if t_cmd <= now - c.ptz_latency_s:
                target = st
        while len(self._cmds) > 1 and self._cmds[1][0] <= now - c.ptz_latency_s:
            self._cmds.popleft()
        step = c.pan_speed_dps * dt
        zstep = c.zoom_speed * dt

        def toward(a, b, lim):
            return b if abs(b - a) <= lim else a + (lim if b > a else -lim)

        self.est = PTZState(toward(self.est.pan, target.pan, step),
                            toward(self.est.tilt, target.tilt, step),
                            toward(self.est.zoom, target.zoom, zstep))
        return self.est

    def fov(self) -> tuple[float, float]:
        """Current horizontal/vertical field of view in degrees (at the estimated real zoom)."""
        z = max(self.est.zoom, 1.0)
        h = 2 * np.degrees(np.arctan(np.tan(np.radians(self.cfg.hfov_deg / 2)) / z))
        v = 2 * np.degrees(np.arctan(np.tan(np.radians(self.cfg.vfov_deg / 2)) / z))
        return float(h), float(v)

    # -- public API ----------------------------------------------------------
    def move(self, st: PTZState, force: bool = False) -> bool:
        """Command an absolute position. Rate limited unless `force`."""
        now = self.clock()
        if not force and now - self._last_cmd < self.cfg.command_interval_s:
            return False
        st = self.clamp(st)
        self.update_estimate(now)
        self._send(*self.to_raw(st))
        self.state = st
        self._last_cmd = now
        self._cmds.append((now, st))
        # When the real camera will have arrived, per the same model.
        c = self.cfg
        travel = max(abs(st.pan - self.est.pan), abs(st.tilt - self.est.tilt))
        settle = (c.ptz_latency_s + travel / max(c.pan_speed_dps, 1e-3)
                  + abs(st.zoom - self.est.zoom) / max(c.zoom_speed, 1e-3) + c.ptz_settle_margin_s)
        self.moving_until = max(self.moving_until, now + settle)
        return True

    def home(self) -> None:
        c = self.cfg
        self.move(PTZState(c.home_pan, c.home_tilt, c.home_zoom), force=True)
        self.moving_until += 0.3

    def _send(self, pan: int, tilt: int, zoom: int) -> None:
        raise NotImplementedError


class NullPTZ(PTZ):
    """No hardware control (fixed camera or video-file testing)."""

    def _send(self, pan, tilt, zoom):
        pass


class OpenCVPTZ(PTZ):
    """Windows/macOS: DirectShow camera-control properties through OpenCV."""

    default_raw_pan = (-130, 130)
    default_raw_tilt = (-90, 90)

    def __init__(self, cfg: CameraConfig, source: FrameSource):
        super().__init__(cfg)
        self.source = source

    def _send(self, pan, tilt, zoom):
        cap = self.source.cap
        with self.source.lock:
            cap.set(cv2.CAP_PROP_PAN, pan)
            cap.set(cv2.CAP_PROP_TILT, tilt)
            cap.set(cv2.CAP_PROP_ZOOM, zoom)


class V4L2PTZ(PTZ):
    """Linux: UVC controls via v4l2-ctl (package v4l-utils)."""

    default_raw_pan = (-468000, 468000)     # +/-130 deg in arc-seconds
    default_raw_tilt = (-324000, 324000)    # +/-90 deg

    def __init__(self, cfg: CameraConfig):
        super().__init__(cfg)
        if not shutil.which("v4l2-ctl"):
            raise RuntimeError("v4l2-ctl not found: sudo apt install v4l-utils")
        dev = cfg.device
        self.dev = f"/dev/video{dev}" if isinstance(dev, int) or str(dev).isdigit() else str(dev)

    def _send(self, pan, tilt, zoom):
        cmd = ["v4l2-ctl", "-d", self.dev, "-c",
               f"pan_absolute={pan},tilt_absolute={tilt},zoom_absolute={zoom}"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            log.warning("v4l2-ctl failed: %s", res.stderr.strip())


# DirectShow IAMCameraControl (strmif.h). Property ids and flags:
CC_PAN, CC_TILT, CC_ZOOM = 0, 1, 3
CC_FLAGS_MANUAL = 2


def _camera_control(index: int):
    """Open the IAMCameraControl interface of DirectShow video device `index`."""
    from ctypes import HRESULT, POINTER, c_long

    from comtypes import COMMETHOD, GUID, IUnknown
    from pygrabber.dshow_graph import SystemDeviceEnum
    from pygrabber.dshow_ids import DeviceCategories

    class IAMCameraControl(IUnknown):
        _iid_ = GUID("{C6E13370-30AC-11d0-A18C-00A0C9118956}")
        _methods_ = [
            COMMETHOD([], HRESULT, "GetRange",
                      (["in"], c_long, "Property"),
                      (["out"], POINTER(c_long), "pMin"),
                      (["out"], POINTER(c_long), "pMax"),
                      (["out"], POINTER(c_long), "pSteppingDelta"),
                      (["out"], POINTER(c_long), "pDefault"),
                      (["out"], POINTER(c_long), "pCapsFlags")),
            COMMETHOD([], HRESULT, "Set",
                      (["in"], c_long, "Property"),
                      (["in"], c_long, "lValue"),
                      (["in"], c_long, "Flags")),
            COMMETHOD([], HRESULT, "Get",
                      (["in"], c_long, "Property"),
                      (["out"], POINTER(c_long), "lValue"),
                      (["out"], POINTER(c_long), "Flags")),
        ]

    filt, name = SystemDeviceEnum().get_filter_by_index(DeviceCategories.VideoInputDevice, index)
    return filt.QueryInterface(IAMCameraControl), name


def camera_control_ranges(ctrl) -> dict:
    """{property: (min, max, step, default)} for pan/tilt/zoom the device supports."""
    ranges = {}
    for prop in (CC_PAN, CC_TILT, CC_ZOOM):
        try:
            mn, mx, step, default, _caps = ctrl.GetRange(prop)
            if mx > mn:
                ranges[prop] = (mn, mx, step, default)
        except Exception:
            pass
    return ranges


class DShowPTZ(PTZ):
    """Windows: DirectShow camera control, talking to the driver directly.

    Uses the ranges the camera itself reports, so no calibration is needed.
    """

    def __init__(self, cfg: CameraConfig, index: int):
        super().__init__(cfg)
        self.ctrl, self.name = _camera_control(index)
        self.ranges = camera_control_ranges(self.ctrl)
        if not self.ranges:
            raise RuntimeError(f"Camera {index} ({self.name}) has no pan/tilt/zoom controls")
        if CC_PAN in self.ranges:
            self.default_raw_pan = self.ranges[CC_PAN][:2]
        if CC_TILT in self.ranges:
            self.default_raw_tilt = self.ranges[CC_TILT][:2]
        if CC_ZOOM in self.ranges:
            self.default_raw_zoom = self.ranges[CC_ZOOM][:2]
        log.info("PTZ via DirectShow on %s: %s", self.name,
                 {k: v[:2] for k, v in zip(("pan", "tilt", "zoom"), (self.ranges.get(p) for p in (CC_PAN, CC_TILT, CC_ZOOM))) if v})

    def to_raw(self, st):
        # The device's own ranges are authoritative on this backend.
        c = self.cfg
        pan = -st.pan if c.invert_pan else st.pan
        tilt = -st.tilt if c.invert_tilt else st.tilt
        return (int(round(self._map(pan, c.pan_limits, self.default_raw_pan))),
                int(round(self._map(tilt, c.tilt_limits, self.default_raw_tilt))),
                int(round(self._map(st.zoom, (1.0, c.max_zoom_ratio), self.default_raw_zoom))))

    def _send(self, pan, tilt, zoom):
        for prop, value in ((CC_PAN, pan), (CC_TILT, tilt), (CC_ZOOM, zoom)):
            if prop not in self.ranges:
                continue
            mn, mx = self.ranges[prop][:2]
            try:
                self.ctrl.Set(prop, int(min(max(value, mn), mx)), CC_FLAGS_MANUAL)
            except Exception as exc:
                log.warning("Camera control %d=%d failed: %s", prop, value, exc)


def make_ptz(cfg: CameraConfig, source: FrameSource) -> PTZ:
    backend = cfg.ptz_backend
    if source.is_file or backend == "none":
        return NullPTZ(cfg)
    if backend == "auto":
        if platform.system() == "Linux":
            backend = "v4l2"
        elif platform.system() == "Windows":
            try:
                return DShowPTZ(cfg, resolve_device(cfg.device))
            except Exception as exc:
                log.warning("DirectShow camera control unavailable (%s); trying OpenCV", exc)
            backend = "opencv"
        else:
            backend = "opencv"
    if backend == "dshow":
        return DShowPTZ(cfg, resolve_device(cfg.device))
    if backend == "v4l2":
        return V4L2PTZ(cfg)
    if backend == "opencv":
        return OpenCVPTZ(cfg, source)
    raise ValueError(f"Unknown ptz_backend {cfg.ptz_backend!r}")


def probe(cfg: CameraConfig) -> str:
    """Report what the camera/driver supports, to fill in the raw_* ranges."""
    lines = []
    if platform.system() == "Linux" and shutil.which("v4l2-ctl"):
        dev = cfg.device
        dev = f"/dev/video{dev}" if isinstance(dev, int) or str(dev).isdigit() else str(dev)
        for args in (["--list-ctrls"], ["--list-formats-ext"]):
            res = subprocess.run(["v4l2-ctl", "-d", dev, *args], capture_output=True, text=True)
            lines.append(res.stdout or res.stderr)
    else:
        cap = _open_capture(cfg)
        for name in ("FRAME_WIDTH", "FRAME_HEIGHT", "FPS"):
            lines.append(f"{name:14s} = {cap.get(getattr(cv2, 'CAP_PROP_' + name))}")
        cap.release()
        if platform.system() == "Windows":
            try:
                ctrl, name = _camera_control(resolve_device(cfg.device))
                lines.append(f"DirectShow camera control on: {name}")
                ranges = camera_control_ranges(ctrl)
                for label, prop in (("pan", CC_PAN), ("tilt", CC_TILT), ("zoom", CC_ZOOM)):
                    r = ranges.get(prop)
                    lines.append(f"  {label:5s}: " + (f"min={r[0]} max={r[1]} step={r[2]} default={r[3]}" if r else "NOT SUPPORTED"))
            except Exception as exc:
                lines.append(f"DirectShow camera control failed: {exc}")
    return "\n".join(lines)


def list_devices() -> str:
    """List cameras and whether each one can pan/tilt/zoom."""
    if platform.system() != "Windows":
        return "On Linux use: v4l2-ctl --list-devices"
    names = windows_camera_names()
    if not names:
        return "No cameras found (or pygrabber missing: python -m pip install pygrabber)."
    lines = ["Cameras (number: name -> pan/tilt/zoom support):"]
    for i, name in enumerate(names):
        try:
            ranges = camera_control_ranges(_camera_control(i)[0])
            ptz = ", ".join(l for l, p in (("pan", CC_PAN), ("tilt", CC_TILT), ("zoom", CC_ZOOM)) if p in ranges) or "none"
        except Exception as exc:
            ptz = f"none ({exc.__class__.__name__})"
        lines.append(f"  {i}: {name}  ->  {ptz}")
    lines.append("")
    lines.append("Put the right one in Settings > Camera device (number or name, e.g. OBSBOT Tiny 2).")
    try:
        from .audio import list_input_devices
        mics = sorted({name for _, name, _ in list_input_devices()})
        lines.append("")
        lines.append("Microphones:")
        lines.extend(f"  {m}" for m in mics)
        lines.append("Settings > Audio > Microphone takes part of a name, e.g. OBSBOT.")
    except Exception as exc:
        lines.append(f"(microphones: {exc})")
    return "\n".join(lines)

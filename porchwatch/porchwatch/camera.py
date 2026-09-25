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
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)


def _open_capture(cfg: CameraConfig) -> cv2.VideoCapture:
    device = cfg.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
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

    def __init__(self, cfg: CameraConfig, clock=time.time):
        self.cfg = cfg
        self.clock = clock
        self.state = PTZState(cfg.home_pan, cfg.home_tilt, cfg.home_zoom)
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
        raw_zoom = self._map(st.zoom, (1.0, c.max_zoom_ratio), c.raw_zoom_range)
        return int(round(raw_pan)), int(round(raw_tilt)), int(round(raw_zoom))

    def fov(self) -> tuple[float, float]:
        """Current horizontal/vertical field of view in degrees."""
        z = max(self.state.zoom, 1.0)
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
        prev = self.state
        self._send(*self.to_raw(st))
        self.state = st
        self._last_cmd = now
        # Rough settle time: gimbal ~120 deg/s plus fixed latency.
        travel = max(abs(st.pan - prev.pan), abs(st.tilt - prev.tilt))
        settle = 0.08 + travel / 120.0 + (0.15 if st.zoom != prev.zoom else 0.0)
        self.moving_until = max(self.moving_until, now + settle)
        return True

    def home(self) -> None:
        c = self.cfg
        self.move(PTZState(c.home_pan, c.home_tilt, c.home_zoom), force=True)
        self.moving_until = self.clock() + 1.5

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


def make_ptz(cfg: CameraConfig, source: FrameSource) -> PTZ:
    backend = cfg.ptz_backend
    if source.is_file or backend == "none":
        return NullPTZ(cfg)
    if backend == "auto":
        backend = "v4l2" if platform.system() == "Linux" else "opencv"
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
        for name in ("PAN", "TILT", "ZOOM", "FOCUS", "EXPOSURE", "FRAME_WIDTH", "FRAME_HEIGHT", "FPS"):
            lines.append(f"{name:14s} = {cap.get(getattr(cv2, 'CAP_PROP_' + name))}")
        cap.release()
    return "\n".join(lines)

"""Physics of the virtual camera's zoom head (the black ball).

Numbers from the datasheet: pan 0-360 deg, tilt -15..90 deg, about 30 deg/s on
both axes, 20x zoom (4.7-94 mm). Command latency, zoom speed and autofocus time
are guesses until the real camera is measured (the probe tool does that); they
are parameters so tests can try faster and slower heads.

ONVIF generic coordinates used by the simulator (also a guess, to be confirmed):
pan  x in [-1, 1]  <->  0..360 deg        (x = -1 is 0 deg)
tilt y in [-1, 1]  <->  -15..90 deg
zoom z in [0, 1]   <->  focal 4.7..94 mm  (1x..20x, linear in focal length)
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


@dataclass
class PTZLimits:
    pan_speed: float = 30.0          # deg/s at full speed
    tilt_speed: float = 30.0
    tilt_min: float = -15.0
    tilt_max: float = 90.0
    zoom_max: float = 20.0
    zoom_full_s: float = 4.0         # time for 1x -> 20x
    latency_s: float = 0.15          # command -> motion starts
    af_base_s: float = 0.30          # autofocus after any move
    af_per_zoom_s: float = 0.03      # extra AF time per zoom step (x)


def pan_to_norm(deg: float) -> float:
    return (deg % 360.0) / 180.0 - 1.0


def norm_to_pan(x: float) -> float:
    return ((x + 1.0) * 180.0) % 360.0


def tilt_to_norm(deg: float, lim: PTZLimits = PTZLimits()) -> float:
    return (deg - lim.tilt_min) / (lim.tilt_max - lim.tilt_min) * 2.0 - 1.0


def norm_to_tilt(y: float, lim: PTZLimits = PTZLimits()) -> float:
    return lim.tilt_min + (y + 1.0) / 2.0 * (lim.tilt_max - lim.tilt_min)


def zoom_to_norm(zoom: float, lim: PTZLimits = PTZLimits()) -> float:
    return (zoom - 1.0) / (lim.zoom_max - 1.0)


def norm_to_zoom(z: float, lim: PTZLimits = PTZLimits()) -> float:
    return 1.0 + min(max(z, 0.0), 1.0) * (lim.zoom_max - 1.0)


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


class SimPTZ:
    """Pan/tilt/zoom state that moves with (simulated or real) time."""

    def __init__(self, clock=time.monotonic, limits: PTZLimits | None = None,
                 pan: float = 0.0, tilt: float = 5.0, zoom: float = 1.0):
        self.clock = clock
        self.lim = limits or PTZLimits()
        self.pan, self.tilt, self.zoom = pan % 360.0, tilt, zoom
        self._t = clock()
        self._lock = threading.RLock()
        self._pending: list[tuple[float, str, tuple]] = []     # (activate_at, kind, args)
        self._mode = "idle"                                      # idle | velocity | target
        self._vel = (0.0, 0.0, 0.0)                              # deg/s, deg/s, ln(zoom)/s
        self._vel_until = math.inf
        self._target: tuple[float, float, float] | None = None
        self._speed_frac = 1.0
        self.last_motion_end = self._t - 10.0
        self.presets: dict[int, tuple[float, float, float]] = {}

    # ------------------------------------------------------------------ commands
    def _queue(self, kind: str, *args) -> None:
        with self._lock:
            self.update()
            self._pending.append((self.clock() + self.lim.latency_s, kind, args))

    def joystick(self, pan_dir: float, tilt_dir: float, speed_frac: float, duration_s: float | None) -> None:
        """pan_dir/tilt_dir in {-1, 0, 1}; speed_frac 0..1; duration None = until stop."""
        self._queue("velocity", pan_dir * speed_frac * self.lim.pan_speed,
                    tilt_dir * speed_frac * self.lim.tilt_speed, 0.0, duration_s)

    def zoom_move(self, direction: float, duration_s: float | None, speed_frac: float = 1.0) -> None:
        rate = direction * speed_frac * math.log(self.lim.zoom_max) / self.lim.zoom_full_s
        self._queue("velocity", 0.0, 0.0, rate, duration_s)

    def absolute(self, pan: float | None, tilt: float | None, zoom: float | None, speed_frac: float = 1.0) -> None:
        self._queue("target", pan, tilt, zoom, speed_frac)

    def stop(self) -> None:
        self._queue("stop")

    def save_preset(self, n: int) -> None:
        with self._lock:
            self.update()
            self.presets[n] = (self.pan, self.tilt, self.zoom)

    def goto_preset(self, n: int) -> bool:
        if n not in self.presets:
            return False
        self.absolute(*self.presets[n])
        return True

    # ------------------------------------------------------------------ integration
    def _apply(self, kind: str, args: tuple, t: float) -> None:
        if kind == "stop":
            if self._mode != "idle":
                self.last_motion_end = t
            self._mode, self._vel, self._target = "idle", (0.0, 0.0, 0.0), None
        elif kind == "velocity":
            vp, vt, vz, dur = args
            self._mode, self._vel = "velocity", (vp, vt, vz)
            self._vel_until = t + dur if dur else math.inf
            self._target = None
        elif kind == "target":
            pan, tilt, zoom, frac = args
            self._target = (self.pan if pan is None else pan % 360.0,
                            self.tilt if tilt is None else min(max(tilt, self.lim.tilt_min), self.lim.tilt_max),
                            self.zoom if zoom is None else min(max(zoom, 1.0), self.lim.zoom_max))
            self._speed_frac = min(max(frac, 0.05), 1.0)
            self._mode = "target"

    def _advance(self, t0: float, t1: float) -> None:
        dt = t1 - t0
        if dt <= 0 or self._mode == "idle":
            return
        if self._mode == "velocity":
            end = min(t1, self._vel_until)
            d = max(0.0, end - t0)
            vp, vt, vz = self._vel
            self.pan = (self.pan + vp * d) % 360.0
            self.tilt = min(max(self.tilt + vt * d, self.lim.tilt_min), self.lim.tilt_max)
            self.zoom = min(max(self.zoom * math.exp(vz * d), 1.0), self.lim.zoom_max)
            if t1 >= self._vel_until:
                self._mode, self._vel = "idle", (0.0, 0.0, 0.0)
                self.last_motion_end = self._vel_until
            return
        tp, tt, tz = self._target
        f = self._speed_frac
        dp = _wrap180(tp - self.pan)
        dtl = tt - self.tilt
        dz = math.log(tz) - math.log(self.zoom)
        zr = math.log(self.lim.zoom_max) / self.lim.zoom_full_s
        need = max(abs(dp) / (self.lim.pan_speed * f), abs(dtl) / (self.lim.tilt_speed * f), abs(dz) / (zr * f))
        if need <= dt:
            self.pan, self.tilt, self.zoom = tp, tt, tz
            self._mode, self._target = "idle", None
            self.last_motion_end = t0 + need
            return
        sp = min(abs(dp), self.lim.pan_speed * f * dt)
        st = min(abs(dtl), self.lim.tilt_speed * f * dt)
        sz = min(abs(dz), zr * f * dt)
        self.pan = (self.pan + math.copysign(sp, dp)) % 360.0
        self.tilt += math.copysign(st, dtl)
        self.zoom = math.exp(math.log(self.zoom) + math.copysign(sz, dz))

    def update(self) -> None:
        with self._lock:
            now = self.clock()
            self._pending.sort(key=lambda p: p[0])
            while self._pending and self._pending[0][0] <= now:
                at, kind, args = self._pending.pop(0)
                self._advance(self._t, at)
                self._t = max(self._t, at)
                self._apply(kind, args, at)
            self._advance(self._t, now)
            self._t = now

    # ------------------------------------------------------------------ queries
    def state(self) -> dict:
        with self._lock:
            self.update()
            moving = self._mode != "idle" or bool(self._pending)
            return {"pan": self.pan, "tilt": self.tilt, "zoom": self.zoom, "moving": moving,
                    "pan_speed": abs(self._vel[0]) if self._mode == "velocity" else
                    (self.lim.pan_speed * self._speed_frac if self._mode == "target" else 0.0)}

    def sharpness(self) -> float:
        """1.0 when focused; drops while moving and until autofocus settles."""
        st = self.state()
        if st["moving"]:
            return 0.3
        settle = self.lim.af_base_s + self.lim.af_per_zoom_s * self.zoom
        since = self.clock() - self.last_motion_end
        return 1.0 if since >= settle else 0.3 + 0.7 * since / settle

    def hfov(self, sensor_w_mm: float = 5.12, f_wide_mm: float = 4.7) -> float:
        """Horizontal field of view of the zoom lens, degrees."""
        f = f_wide_mm * self.state()["zoom"]
        return math.degrees(2 * math.atan(sensor_w_mm / 2 / f))

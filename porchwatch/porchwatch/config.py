"""Configuration: dataclass defaults, optionally overridden by a YAML file."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    # Device index, a camera name on Windows (e.g. "OBSBOT Tiny 2"; list them with
    # `python -m porchwatch devices`) or a path such as /dev/video0 on Linux.
    device: Any = 0
    width: int = 1920          # 3840x2160 gives much better plate reads, costs CPU/USB.
    height: int = 1080
    fps: int = 30
    # "auto" -> DirectShow camera control on Windows (falls back to OpenCV),
    # v4l2 on Linux. "none" disables PTZ.
    ptz_backend: str = "auto"
    # Physical limits in degrees and the zoom ratio at zoom 0 / zoom max.
    pan_limits: tuple = (-130.0, 130.0)
    tilt_limits: tuple = (-90.0, 90.0)
    max_zoom_ratio: float = 4.0
    # Raw control ranges for the opencv/v4l2 backends (see `python -m porchwatch probe`).
    # The Windows DirectShow backend reads them from the camera itself.
    # Linux UVC: pan/tilt in arc-seconds. DirectShow: degrees.
    raw_pan_range: tuple | None = None    # None -> backend default
    raw_tilt_range: tuple | None = None
    raw_zoom_range: tuple | None = None   # None -> backend default (0-100)
    invert_pan: bool = False
    invert_tilt: bool = False
    # Field of view at 1x zoom (degrees). Tiny 2: ~86 deg diagonal.
    hfov_deg: float = 76.0
    vfov_deg: float = 47.0
    # Where the camera rests while watching (the "wide" view of the street).
    home_pan: float = 0.0
    home_tilt: float = 0.0
    home_zoom: float = 1.0
    # How the real gimbal responds (used to know where it points while moving):
    ptz_latency_s: float = 0.25     # delay before it starts moving after a command
    pan_speed_dps: float = 60.0     # turning speed, degrees per second
    zoom_speed: float = 2.0         # zoom ratio change per second (1x -> 3x takes 1 s)
    ptz_settle_margin_s: float = 0.15   # extra wait before trusting the picture after a move
    # Minimum seconds between PTZ commands (UVC control transfers are slow).
    command_interval_s: float = 0.08


@dataclass
class DetectionConfig:
    model: str = "yolo26n.pt"       # downloaded automatically by ultralytics
    device: str = ""                # "" = auto, "cpu", "cuda:0", "mps"
    imgsz: int = 640
    person_conf: float = 0.45
    vehicle_conf: float = 0.40
    # A vehicle is "moving" when its centre travels this fraction of the frame
    # width within `motion_window_s`, measured while the camera is at home.
    motion_min_travel: float = 0.03
    motion_window_s: float = 0.6
    # Ignore detections whose centre lies in these normalised [x1, y1, x2, y2]
    # boxes (e.g. your own driveway, a neighbour's window).
    ignore_zones: list = field(default_factory=list)
    # Only react to things inside this normalised box (default: whole frame).
    watch_zone: tuple = (0.0, 0.0, 1.0, 1.0)


@dataclass
class TrackingConfig:
    gain: float = 0.45              # fraction of the angular error corrected per command
    deadband: float = 0.04          # ignore centring errors below this (fraction of frame)
    zoom_step: float = 0.35         # zoom ratio change per step
    # Zoom until the face is this tall / the plate this wide (fraction of frame).
    # Before a face/plate is found, it is estimated from the person/vehicle box.
    face_fill: float = 0.20
    plate_fill: float = 0.18
    vehicle_fill: float = 0.80      # never zoom a vehicle wider than this
    # Aim point inside the box before a face/plate is found (fraction down the box).
    person_aim_y: float = 0.10
    vehicle_aim_y: float = 0.70
    # Lead moving targets by this many seconds (compensates camera latency).
    lead_s: float = 0.2             # extra lead: centres moving targets during the pause
    lost_timeout_s: float = 1.2     # (time spent waiting for the camera to stop doesn't count)
    # Before chasing, a person / vehicle must have been seen this many times with
    # at least this average confidence (filters one-frame ghosts).
    min_sightings: int = 6
    min_avg_conf: float = 0.5
    min_age_s: float = 0.5          # ...and for at least this long (people)
    max_track_s: float = 15.0
    # Keep following a person / car until it leaves the picture instead of
    # returning to watch/patrol as soon as a face or plate is captured.
    follow_until_gone: bool = False
    follow_max_s: float = 120.0     # safety limit in follow mode (e.g. someone standing still)
    capture_window_s: float = 4.0   # how long to keep collecting once zoomed in
    cooldown_s: float = 2.0         # after returning home, before picking a new target
    # Don't chase the same person/vehicle again for this long after a capture
    # (matched by where it should be by now, given its speed).
    recapture_after_s: float = 30.0
    prefer_vehicles: bool = True    # moving cars leave frame faster than walkers


@dataclass
class PatrolConfig:
    # Sweep between two pan angles, stopping at each position to look.
    enabled: bool = False
    left_pan: float = -45.0
    right_pan: float = 45.0
    stops: int = 3                  # positions between (and including) the two edges
    dwell_s: float = 3.0            # seconds to hold still and look at each stop


@dataclass
class CaptureConfig:
    output_dir: str = "captures"
    min_face_px: int = 80
    min_sharpness: float = 60.0     # variance of Laplacian; lower = blurrier
    plate_min_conf: float = 0.70
    plate_votes: int = 3            # identical reads needed before accepting a plate
    save_context_frame: bool = True
    retention_days: int = 30        # 0 = keep forever


@dataclass
class RecordingConfig:
    # "events": record only while something is detected/tracked (plus pre/post roll).
    # "continuous": always record. "off": snapshots only.
    mode: str = "events"
    trigger_people: bool = True     # events mode: record when a person is detected
    trigger_vehicles: bool = True   # ...when a vehicle is moving (parked cars never trigger)
    trigger_frames: int = 3         # sightings within 1 s needed before a person starts a recording
    width: int = 1280
    height: int = 720
    fps: int = 15
    # h264 / h265: efficient, via ffmpeg (small files). mp4v / MJPG / XVID: legacy OpenCV codecs.
    codec: str = "h264"
    encoder: str = "auto"           # auto (NVIDIA GPU if available, else CPU), nvidia, cpu
    crf: int = 28                   # compression: higher = smaller files / lower quality (18-40)
    output_dir: str = "recordings"
    segment_minutes: int = 10       # continuous mode: start a new file every N minutes
    pre_record_s: float = 5.0
    post_record_s: float = 10.0
    timestamp_overlay: bool = True
    retention_days: int = 14        # 0 = keep forever
    max_storage_gb: float = 50.0    # oldest files are deleted above this; 0 = no limit


@dataclass
class AudioConfig:
    enabled: bool = False
    # Part of the microphone name (e.g. "OBSBOT"), an index, or "" for the system default.
    device: Any = "OBSBOT"
    sample_rate: int = 48000
    gain_db: float = 12.0           # boost; the limiter keeps loud sounds from distorting
    high_pass_hz: int = 120         # cut wind / traffic rumble below this; 0 = off
    noise_gate: bool = False        # quiet the background hiss between sounds
    gate_threshold_db: float = -50.0
    limiter: bool = True
    bitrate_kbps: int = 128
    sync_offset_ms: int = 0         # shift audio later (+) or earlier (-) if lips don't match


@dataclass
class WebConfig:
    enabled: bool = True
    # 127.0.0.1 = only this computer. Use 0.0.0.0 to reach it from your phone
    # on the home network (set a password first!).
    host: str = "127.0.0.1"
    port: int = 8080
    username: str = "admin"
    password: str = ""              # empty = no login required
    # Extra host names allowed in the browser address bar (IP addresses and
    # "localhost" always work), e.g. ["mypc.local"].
    allowed_hosts: list = field(default_factory=list)
    stream_fps: int = 10
    stream_width: int = 960


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    patrol: PatrolConfig = field(default_factory=PatrolConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    web: WebConfig = field(default_factory=WebConfig)
    tracking_enabled: bool = True   # False = fixed camera, detection + recording only
    show_preview: bool = True       # local OpenCV window


def _merge(obj: Any, data: dict, path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{path}{key} must be a mapping")
            _merge(current, value, f"{path}{key}.")
        else:
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, key, value)


def update_config(cfg: Config, data: dict) -> None:
    """Apply a (possibly partial) nested dict of settings, validating keys."""
    _merge(cfg, data)


def config_to_dict(cfg: Any) -> dict:
    out = {}
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        if is_dataclass(value):
            out[f.name] = config_to_dict(value)
        elif isinstance(value, tuple):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out


_save_lock = threading.Lock()


def save_config(cfg: Config, path: str | Path) -> None:
    """Atomic write; safe to call from the web thread and the main loop at once."""
    import yaml

    with _save_lock:
        tmp = Path(f"{path}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
        os.replace(tmp, path)


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path and Path(path).exists():
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge(cfg, data)
    return cfg

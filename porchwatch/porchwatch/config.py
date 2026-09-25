"""Configuration: dataclass defaults, optionally overridden by a YAML file."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    # Device index (Windows/macOS) or path such as /dev/video0 (Linux).
    device: Any = 0
    width: int = 1920          # 3840x2160 gives much better plate reads, costs CPU/USB.
    height: int = 1080
    fps: int = 30
    # "auto" -> v4l2 on Linux, OpenCV/DirectShow elsewhere. "none" disables PTZ.
    ptz_backend: str = "auto"
    # Physical limits in degrees and the zoom ratio at zoom 0 / zoom max.
    pan_limits: tuple = (-130.0, 130.0)
    tilt_limits: tuple = (-90.0, 90.0)
    max_zoom_ratio: float = 4.0
    # Raw control ranges the driver reports (see `python -m porchwatch probe`).
    # Linux UVC: pan/tilt in arc-seconds. DirectShow: degrees.
    raw_pan_range: tuple | None = None    # None -> backend default
    raw_tilt_range: tuple | None = None
    raw_zoom_range: tuple = (0, 100)
    invert_pan: bool = False
    invert_tilt: bool = False
    # Field of view at 1x zoom (degrees). Tiny 2: ~86 deg diagonal.
    hfov_deg: float = 76.0
    vfov_deg: float = 47.0
    # Where the camera rests while watching (the "wide" view of the street).
    home_pan: float = 0.0
    home_tilt: float = 0.0
    home_zoom: float = 1.0
    # Minimum seconds between PTZ commands (UVC control transfers are slow).
    command_interval_s: float = 0.08


@dataclass
class DetectionConfig:
    model: str = "yolo11n.pt"       # downloaded automatically by ultralytics
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
    lead_s: float = 0.25
    lost_timeout_s: float = 1.2
    max_track_s: float = 15.0
    capture_window_s: float = 4.0   # how long to keep collecting once zoomed in
    cooldown_s: float = 2.0         # after returning home, before picking a new target
    # Don't chase the same person/vehicle again for this long after a capture
    # (matched by where it should be by now, given its speed).
    recapture_after_s: float = 30.0
    prefer_vehicles: bool = True    # moving cars leave frame faster than walkers


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
    width: int = 1280
    height: int = 720
    fps: int = 15
    codec: str = "mp4v"             # mp4v (.mp4), avc1 (.mp4, needs H.264 build), MJPG (.avi), XVID (.avi)
    output_dir: str = "recordings"
    segment_minutes: int = 10       # continuous mode: start a new file every N minutes
    pre_record_s: float = 5.0
    post_record_s: float = 10.0
    timestamp_overlay: bool = True
    retention_days: int = 14        # 0 = keep forever
    max_storage_gb: float = 50.0    # oldest files are deleted above this; 0 = no limit


@dataclass
class WebConfig:
    enabled: bool = True
    # 127.0.0.1 = only this computer. Use 0.0.0.0 to reach it from your phone
    # on the home network (set a password first!).
    host: str = "127.0.0.1"
    port: int = 8080
    username: str = "admin"
    password: str = ""              # empty = no login required
    stream_fps: int = 10
    stream_width: int = 960


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
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


def save_config(cfg: Config, path: str | Path) -> None:
    import yaml

    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
    tmp.replace(path)


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path and Path(path).exists():
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge(cfg, data)
    return cfg

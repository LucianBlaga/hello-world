"""Local web dashboard: live view, events, recordings, PTZ pad and settings."""
from __future__ import annotations

import copy
import functools
import logging
import threading
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from .config import config_to_dict, save_config, update_config
from .storage import disk_usage

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"

RESOLUTIONS_CAMERA = ["3840x2160", "2560x1440", "1920x1080", "1280x720", "960x540", "640x360"]
RESOLUTIONS_RECORD = ["3840x2160", "2560x1440", "1920x1080", "1280x720", "854x480", "640x360"]
FPS = [60, 30, 25, 24, 20, 15, 12, 10, 5]

# Each field: key (dotted path; a list for resolution pairs), label, type, extra.
# restart=True -> the camera/recorder pipeline is reopened when it changes.
SCHEMA = [
    {"title": "Camera", "fields": [
        {"key": ["camera.width", "camera.height"], "label": "Capture resolution", "type": "resolution",
         "options": RESOLUTIONS_CAMERA, "restart": True,
         "help": "What the camera sends. 4K gives far better licence-plate and face detail but needs a strong PC."},
        {"key": "camera.fps", "label": "Capture frame rate", "type": "select", "options": [60, 30, 25, 15],
         "restart": True, "help": "Tiny 2: 4K up to 30 fps, 1080p up to 60 fps."},
        {"key": "camera.device", "label": "Camera device", "type": "text", "restart": True,
         "help": "Number (0, 1, ...) or, on Windows, a name like OBSBOT Tiny 2. See: python -m porchwatch devices"},
        {"key": "camera.ptz_backend", "label": "PTZ control", "type": "select",
         "options": ["auto", "dshow", "opencv", "v4l2", "none"], "restart": True},
        {"key": "camera.home_pan", "label": "Home pan (deg)", "type": "number", "min": -130, "max": 130, "step": 0.5},
        {"key": "camera.home_tilt", "label": "Home tilt (deg)", "type": "number", "min": -90, "max": 90, "step": 0.5},
        {"key": "camera.home_zoom", "label": "Home zoom (x)", "type": "number", "min": 1, "max": 4, "step": 0.1},
        {"key": "camera.invert_pan", "label": "Invert pan", "type": "bool", "help": "Enable if the camera is mounted upside-down or turns the wrong way."},
        {"key": "camera.invert_tilt", "label": "Invert tilt", "type": "bool"},
    ]},
    {"title": "Recording", "fields": [
        {"key": "recording.mode", "label": "Recording mode", "type": "select",
         "options": ["events", "continuous", "off"], "restart": True,
         "help": "events = only when someone/something is detected. continuous = 24/7."},
        {"key": ["recording.width", "recording.height"], "label": "Recording resolution", "type": "resolution",
         "options": RESOLUTIONS_RECORD, "restart": True},
        {"key": "recording.fps", "label": "Recording frame rate", "type": "select", "options": FPS, "restart": True},
        {"key": "recording.codec", "label": "Codec", "type": "select",
         "options": ["mp4v", "avc1", "MJPG", "XVID"], "restart": True,
         "help": "mp4v works everywhere. avc1 (H.264) is smaller but needs an OpenCV build with H.264."},
        {"key": "recording.pre_record_s", "label": "Pre-record (s)", "type": "number", "min": 0, "max": 30, "step": 1, "restart": True},
        {"key": "recording.post_record_s", "label": "Post-record (s)", "type": "number", "min": 0, "max": 120, "step": 1},
        {"key": "recording.segment_minutes", "label": "File length (min)", "type": "number", "min": 1, "max": 120, "step": 1},
        {"key": "recording.timestamp_overlay", "label": "Timestamp overlay", "type": "bool"},
        {"key": "recording.output_dir", "label": "Recordings folder", "type": "text", "restart": True},
    ]},
    {"title": "Storage", "fields": [
        {"key": "recording.retention_days", "label": "Keep recordings (days)", "type": "number", "min": 0, "max": 365, "step": 1,
         "help": "0 = forever. Old files are deleted automatically."},
        {"key": "recording.max_storage_gb", "label": "Max recordings size (GB)", "type": "number", "min": 0, "max": 10000, "step": 1},
        {"key": "capture.retention_days", "label": "Keep snapshots (days)", "type": "number", "min": 0, "max": 365, "step": 1},
        {"key": "capture.output_dir", "label": "Snapshots folder", "type": "text", "restart": True},
    ]},
    {"title": "Detection", "fields": [
        {"key": "detection.person_conf", "label": "Person sensitivity", "type": "range", "min": 0.2, "max": 0.9, "step": 0.05,
         "invert": True, "help": "Higher = more detections but more false alarms."},
        {"key": "detection.vehicle_conf", "label": "Vehicle sensitivity", "type": "range", "min": 0.2, "max": 0.9, "step": 0.05, "invert": True},
        {"key": "detection.motion_min_travel", "label": "Moving-vehicle threshold", "type": "range", "min": 0.005, "max": 0.1, "step": 0.005,
         "help": "How far (fraction of frame width) a vehicle must move to count as moving. Parked cars are ignored."},
        {"key": "detection.model", "label": "Detection model", "type": "select",
         "options": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], "restart": True,
         "help": "n = fastest, m = most accurate (needs a GPU for real time)."},
        {"key": "detection.imgsz", "label": "Detection image size", "type": "select", "options": [480, 640, 960, 1280], "restart": True},
        {"key": "detection.device", "label": "Compute device", "type": "select", "options": ["", "cpu", "cuda:0", "mps"],
         "restart": True, "help": "Empty = automatic."},
        {"key": "zones", "label": "Zones", "type": "zones",
         "help": "Drag on the picture to draw. Only the watch zone is monitored; ignore zones are skipped (your own driveway, a neighbour's window)."},
    ]},
    {"title": "Tracking", "fields": [
        {"key": "tracking_enabled", "label": "Auto-tracking (pan/tilt/zoom)", "type": "bool"},
        {"key": "tracking.prefer_vehicles", "label": "Prioritise moving vehicles", "type": "bool"},
        {"key": "tracking.gain", "label": "Follow speed", "type": "range", "min": 0.1, "max": 0.9, "step": 0.05,
         "help": "Too high makes the camera overshoot and wobble."},
        {"key": "tracking.lead_s", "label": "Lead moving targets (s)", "type": "number", "min": 0, "max": 1, "step": 0.05},
        {"key": "tracking.face_fill", "label": "Face zoom target", "type": "range", "min": 0.05, "max": 0.5, "step": 0.01,
         "help": "Face height as a fraction of the picture."},
        {"key": "tracking.plate_fill", "label": "Plate zoom target", "type": "range", "min": 0.05, "max": 0.5, "step": 0.01},
        {"key": "tracking.capture_window_s", "label": "Time zoomed in (s)", "type": "number", "min": 1, "max": 30, "step": 0.5},
        {"key": "tracking.max_track_s", "label": "Max chase time (s)", "type": "number", "min": 3, "max": 60, "step": 1},
        {"key": "tracking.lost_timeout_s", "label": "Give up when lost (s)", "type": "number", "min": 0.3, "max": 5, "step": 0.1},
        {"key": "tracking.recapture_after_s", "label": "Don't re-capture same target for (s)", "type": "number",
         "min": 0, "max": 600, "step": 5},
        {"key": "tracking.cooldown_s", "label": "Cool-down at home (s)", "type": "number", "min": 0, "max": 30, "step": 0.5},
    ]},
    {"title": "Snapshots", "fields": [
        {"key": "capture.min_face_px", "label": "Min face size (px)", "type": "number", "min": 30, "max": 400, "step": 5},
        {"key": "capture.min_sharpness", "label": "Min sharpness", "type": "number", "min": 0, "max": 500, "step": 5,
         "help": "Blur filter. Lower it if night shots never save."},
        {"key": "capture.plate_min_conf", "label": "Plate OCR min confidence", "type": "range", "min": 0.3, "max": 0.99, "step": 0.01},
        {"key": "capture.plate_votes", "label": "Plate confirmations", "type": "number", "min": 1, "max": 10, "step": 1,
         "help": "Same text must be read this many times before it is accepted."},
        {"key": "capture.save_context_frame", "label": "Save full scene too", "type": "bool"},
    ]},
    {"title": "Web & access", "fields": [
        {"key": "web.host", "label": "Listen on", "type": "select", "options": ["127.0.0.1", "0.0.0.0"],
         "help": "0.0.0.0 makes the dashboard reachable from other devices on your network (takes effect after restarting the program)."},
        {"key": "web.port", "label": "Port", "type": "number", "min": 1024, "max": 65535, "step": 1},
        {"key": "web.username", "label": "Username", "type": "text"},
        {"key": "web.password", "label": "Password", "type": "password", "help": "Leave empty for no login (only safe on 127.0.0.1)."},
        {"key": "web.stream_fps", "label": "Live view fps", "type": "number", "min": 1, "max": 30, "step": 1},
        {"key": "web.stream_width", "label": "Live view width", "type": "select", "options": [640, 960, 1280, 1920]},
        {"key": "show_preview", "label": "Local preview window", "type": "bool", "restart": True},
    ]},
]


def _restart_keys():
    keys = set()
    for sec in SCHEMA:
        for f in sec["fields"]:
            if f.get("restart"):
                ks = f["key"] if isinstance(f["key"], list) else [f["key"]]
                keys.update(k for k in ks if k)
    return keys


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def create_app(ctx) -> Flask:
    """`ctx` is the running porchwatch.app.App instance."""
    app = Flask(__name__, static_folder=None)

    def auth(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            web = ctx.cfg.web
            if web.password:
                a_ = request.authorization
                if not a_ or a_.username != web.username or a_.password != web.password:
                    return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="PorchWatch"'})
            return fn(*a, **kw)
        return wrapper

    @app.get("/")
    @auth
    def index():
        return send_file(STATIC / "index.html")

    @app.get("/stream.mjpg")
    @auth
    def stream():
        def gen():
            last = None
            while True:
                jpg = ctx.latest_jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(1.0 / max(1, ctx.cfg.web.stream_fps))
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot.jpg")
    @auth
    def snapshot():
        jpg = ctx.latest_raw_jpeg or ctx.latest_jpeg
        if jpg is None:
            abort(503)
        return Response(jpg, mimetype="image/jpeg")

    @app.get("/api/status")
    @auth
    def status():
        rec = ctx.recorder
        return jsonify({
            **ctx.status,
            "fps": round(ctx.fps, 1),
            "recording": bool(rec and rec.recording),
            "recording_file": str(rec.current_file) if rec and rec.current_file else None,
            "paused": ctx.paused,
            "error": ctx.error,
            "storage": disk_usage(ctx.cfg.recording.output_dir),
            "log": list(ctx.storage.messages)[-30:] if ctx.storage else [],
        })

    @app.get("/api/events")
    @auth
    def events():
        return jsonify(ctx.storage.recent_events(int(request.args.get("limit", 60))) if ctx.storage else [])

    @app.get("/captures/<path:p>")
    @auth
    def captures(p):
        return send_from_directory(Path(ctx.cfg.capture.output_dir).resolve(), p)

    @app.get("/api/recordings")
    @auth
    def recordings():
        root = Path(ctx.cfg.recording.output_dir)
        files = sorted((f for f in root.rglob("*") if f.suffix in (".mp4", ".avi")),
                       key=lambda f: f.stat().st_mtime, reverse=True)[:200] if root.exists() else []
        return jsonify([{"path": f.relative_to(root).as_posix(), "size_mb": round(f.stat().st_size / 1e6, 1),
                         "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))}
                        for f in files])

    @app.get("/recordings/<path:p>")
    @auth
    def recording_file(p):
        return send_from_directory(Path(ctx.cfg.recording.output_dir).resolve(), p)

    @app.get("/api/settings")
    @auth
    def get_settings():
        return jsonify({"schema": SCHEMA, "values": config_to_dict(ctx.cfg)})

    @app.post("/api/settings")
    @auth
    def post_settings():
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "Expected a JSON object"}), 400
        new_cfg = copy.deepcopy(ctx.cfg)
        try:
            update_config(new_cfg, data)
        except (ValueError, TypeError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        before, after = _flatten(config_to_dict(ctx.cfg)), _flatten(config_to_dict(new_cfg))
        changed = {k for k in after if before.get(k) != after[k]}
        restart = bool(changed & _restart_keys())
        ctx.apply_config(new_cfg, restart=restart)
        save_config(new_cfg, ctx.config_path)
        return jsonify({"ok": True, "changed": sorted(changed), "restarted": restart})

    @app.post("/api/ptz")
    @auth
    def ptz():
        data = request.get_json(force=True, silent=True) or {}
        ok = ctx.manual_ptz(data.get("action", ""))
        return jsonify({"ok": ok})

    @app.post("/api/pause")
    @auth
    def pause():
        data = request.get_json(force=True, silent=True) or {}
        ctx.paused = bool(data.get("paused", not ctx.paused))
        return jsonify({"paused": ctx.paused})

    return app


def start_web(ctx) -> threading.Thread | None:
    web = ctx.cfg.web
    if not web.enabled:
        return None
    if web.host not in ("127.0.0.1", "localhost") and not web.password:
        log.warning("Web dashboard is reachable from the network WITHOUT a password. Set web.password!")
    from werkzeug.serving import make_server

    server = make_server(web.host, web.port, create_app(ctx), threaded=True)
    th = threading.Thread(target=server.serve_forever, name="web", daemon=True)
    th.start()
    log.info("Dashboard: http://%s:%d", "localhost" if web.host == "0.0.0.0" else web.host, web.port)
    return th

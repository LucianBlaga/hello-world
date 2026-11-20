"""Alerts pushed by the camera (doc chapter 3).

The camera POSTs JSON to http://<our PC>:<port>[prefix]/HAPI/V1.0/Event/Notification
for as long as our subscription lasts (30..3600 s), so a background thread keeps
renewing it and re-registers when the camera forgets it (e.g. after a reboot).

Each alert says what (AlarmType/AlarmSubType), start or end (OccurFlag), which
lens (ChannelNo) and may carry a JPEG. Object alerts carry no position
("ObjectDetectInfo: Null"), so they are triggers, not trackers.

Note: Table 1 of the doc lost every "on" in translation (MotiDetect, RegiDetect,
FaceRecogniti...). The real names are restored below; the doc's own example
message uses "MotionDetect"/"MotionAlarm".
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .hapi import HapiClient, HapiError

log = logging.getLogger(__name__)

MAX_BODY = 8 * 1024 * 1024          # a JPEG in base64 fits easily

_NAME_FIXES = {
    "MotiDetect": "MotionDetect", "MotiAlarm": "MotionAlarm", "RegiDetect": "RegionDetect",
    "LoiteringDetecti": "LoiteringDetection", "FaceRecogniti": "FaceRecognition",
    "FaceRecognitiMatchlistAlarm": "FaceRecognitionMatchlistAlarm",
    "FaceRecognitiMismatchlistAlarm": "FaceRecognitionMismatchlistAlarm",
    "FallingObjectsDetectiAlarm": "FallingObjectsDetectionAlarm",
    "TemperatureDetectiAlarm": "TemperatureDetectionAlarm", "NetworkDiscnected": "NetworkDisconnected",
    "IPCflict": "IPConflict", "TelephingAlarm": "TelephoningAlarm",
}


def fix_name(name: str) -> str:
    return _NAME_FIXES.get(name, name)


@dataclass
class CameraEvent:
    alarm_type: str
    sub_type: str
    active: bool                    # OccurFlag: True = started, False = ended
    channel: int
    timestamp: float                # camera's UTC seconds
    device_id: str
    picture: bytes | None
    info: dict | None
    source_ip: str
    received_at: float = field(default_factory=time.time)

    @property
    def is_person(self) -> bool:
        return self.sub_type in ("HumanShapeDetect", "FaceIsDetected")

    @property
    def is_vehicle(self) -> bool:
        return self.sub_type == "VehicleDetect"


def _b64_jpeg(value) -> bytes | None:
    """Find a base64 JPEG in the Picture block, whatever its exact shape."""
    candidates = []
    if isinstance(value, str):
        candidates.append(value)
    elif isinstance(value, dict):
        candidates += [v for v in value.values() if isinstance(v, str) and len(v) > 100]
    for c in candidates:
        try:
            raw = base64.b64decode(c, validate=False)
        except (binascii.Error, ValueError):
            continue
        if raw[:2] == b"\xff\xd8":
            return raw
    return None


def parse_event(body: bytes, source_ip: str) -> CameraEvent:
    doc = json.loads(body.decode("utf-8", errors="replace"))
    if not isinstance(doc, dict) or "AlarmType" not in doc:
        raise ValueError("not an alarm message")
    occur = doc.get("OccurFlag", True)
    active = occur if isinstance(occur, bool) else str(occur).strip().lower() == "true"
    info = doc.get("AlarmInfo") if isinstance(doc.get("AlarmInfo"), dict) else None
    picture = _b64_jpeg(doc.get("Picture"))
    if picture is None and info is not None:
        picture = _b64_jpeg(info.get("image"))
    return CameraEvent(
        alarm_type=fix_name(str(doc.get("AlarmType", ""))), sub_type=fix_name(str(doc.get("AlarmSubType", ""))),
        active=active, channel=int(doc.get("ChannelNo", doc.get("Channel", 0)) or 0),
        timestamp=float(doc.get("TimeStamp") or 0), device_id=str(doc.get("DeviceID", "")),
        picture=picture, info=info, source_ip=source_ip)


class EventReceiver:
    """Small HTTP server that accepts alerts only from the camera's own IPs."""

    def __init__(self, on_event: Callable[[CameraEvent], None], *, port: int = 9998,
                 bind: str = "0.0.0.0", allowed_ips: set[str] | None = None):
        self.on_event = on_event
        self.allowed_ips = set(allowed_ips or ())
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):          # keep the console quiet
                log.debug("event http: " + fmt, *args)

            def _reply(self, code: int) -> None:
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                ip = self.client_address[0]
                if receiver.allowed_ips and ip not in receiver.allowed_ips:
                    return self._reply(403)
                if not self.path.split("?")[0].rstrip("/").endswith("/HAPI/V1.0/Event/Notification"):
                    return self._reply(404)
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    return self._reply(400)
                if n <= 0 or n > MAX_BODY:
                    return self._reply(413 if n > MAX_BODY else 400)
                body = self.rfile.read(n)
                try:
                    ev = parse_event(body, ip)
                except (ValueError, TypeError, json.JSONDecodeError) as e:
                    log.warning("Ignored a malformed alert from %s: %s", ip, e)
                    return self._reply(400)
                self._reply(200)
                try:
                    receiver.on_event(ev)
                except Exception:                       # never let a bad handler kill the server
                    log.exception("Event handler failed")

            def do_GET(self):
                self._reply(405)

        self.server = ThreadingHTTPServer((bind, port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self._thread = threading.Thread(target=self.server.serve_forever, name="event-receiver", daemon=True)

    def start(self) -> "EventReceiver":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def local_ip_towards(host: str) -> str:
    """Our address on the interface that reaches `host` (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((host, 9))
        return s.getsockname()[0]


class Subscription:
    """Keeps the camera's alert subscription alive."""

    def __init__(self, hapi: HapiClient, port: int, *, server: str | None = None,
                 duration: int = 300, renew_every: float = 60.0, event_types: str = "all",
                 prefix: str = ""):
        self.hapi = hapi
        self.port = port
        self.server = server
        self.duration = duration
        self.renew_every = renew_every
        self.event_types = event_types
        self.prefix = prefix
        self.sub_id: int | None = None
        self.last_ok: float = 0.0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="event-subscription", daemon=True)

    def renew(self) -> None:
        """One renewal step: refresh our ID, or register again if that fails."""
        server = self.server or local_ip_towards(self.hapi.host)
        try:
            if self.sub_id is not None:
                self.hapi.event_refresh(self.sub_id, self.duration)
            else:
                raise HapiError(-1, "not registered", "")
        except (HapiError, OSError, ValueError):
            info = self.hapi.event_register(server, self.port, self.duration, self.event_types, self.prefix)
            self.sub_id = int(info.get("ID")) if info.get("ID") is not None else None
        self.last_ok = time.time()
        self.last_error = None

    def _run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                self.renew()
                backoff = 2.0
                wait = self.renew_every
            except Exception as e:                      # camera offline etc.: retry soon
                self.last_error = str(e)
                self.sub_id = None
                log.warning("Alert subscription failed (%s), retrying in %.0f s", e, backoff)
                wait, backoff = backoff, min(backoff * 2, 60.0)
            self._stop.wait(wait)

    def start(self) -> "Subscription":
        self._thread.start()
        return self

    def stop(self, unsubscribe: bool = True) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=5)
        if unsubscribe and self.sub_id is not None:
            try:
                self.hapi.event_delete(self.sub_id)
            except Exception:
                pass
            self.sub_id = None

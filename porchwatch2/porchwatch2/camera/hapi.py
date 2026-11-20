"""Client for the camera's HTTP API ("HAPI", Anjvision firmware, doc ver. 1.7).

Everything here follows the vendor PDF "HTTP API Command Documentation". Notes on
the quirks of that document, which the code deliberately copes with:

* URLs: /HAPI/V1.0[/Channels/<ID>]/<service>/<resource>. A dual-lens camera is a
  multi-channel device; channel IDs start at 1. ``channel=None`` omits the part.
* Auth: username + password on every request. The password may be plain text or
  its 32-hex MD5; we always send the MD5 (the doc's own example
  e10adc3949ba59abbe56e057f20f883e is md5("123456"), the factory default).
  The MD5 works as a password by itself and travels in clear over HTTP, so the
  camera must live on a network without internet access.
* Replies: {"Response": {"ResponseCode": 0, "ResponseString": "Succeed",
  "Data": ...}}. 0 means success. "null" (a string) means no data. Some replies
  (system/video/get) put the payload next to Data instead of inside it.
* PTZ over HTTP is joystick-only: direction + speed 1-10 + autostop <= 1000 ms
  (the call blocks until it stops). Absolute moves come from ONVIF instead.
* Field names keep the firmware's own spelling ("intervalsecnods",
  "Threadhold_second", "DsplayWeek"); config dicts are passed through untouched.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Any, Callable

import requests

PTZ_DIRECTIONS = ("left", "right", "up", "down", "left_up", "right_up", "left_down", "right_down")

# /smart/objectdetect/capability target types (doc 2.14.2 / 4.7).
TARGET_TYPES = {0: "CAR", 1: "MOTO", 2: "ELECTRICBICYCLE", 3: "BICYCLE", 4: "HUMAN",
                5: "FACE", 6: "NONMOTO_VEHICLE", 7: "FIRE", 8: "FALLINGOBJECT"}

# Config pages that have a get + set pair: name -> path (without get/set).
CONFIG_PAGES = {
    "light": "system/light", "image": "system/image", "video": "system/video",
    "audio": "system/audio", "osd": "system/osd", "userosd": "system/userosd",
    "motiondetect": "smart/motiondetect", "objectdetect": "smart/objectdetect",
    "videocover": "smart/videocover", "facedetect": "smart/facedetect",
    "videogate": "smart/videogate", "regionai": "smart/regionai", "lpr": "smart/lpr",
    "flameflumes": "smart/flameflumes",
}

_STANDARD_KEYS = {"ResponseURL", "SessionID", " SessionID", "ResponseCode", "ResponseString", "Data"}
_MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_FORM = "application/x-www-form-urlencoded"


class HapiError(RuntimeError):
    def __init__(self, code: Any, message: str, path: str):
        super().__init__(f"{path}: {message} (code {code})")
        self.code = code
        self.message = message
        self.path = path


def md5_password(password: str) -> str:
    """The doc accepts plain or MD5; a 32-hex string is taken as already hashed."""
    if _MD5_RE.match(password):
        return password.lower()
    return hashlib.md5(password.encode("utf-8")).hexdigest()


def is_default_password(password: str) -> bool:
    return md5_password(password) == md5_password("123456")


class HapiClient:
    """Thin, typed wrapper around the HTTP API of one camera."""

    def __init__(self, host: str, username: str, password: str, *, port: int = 80,
                 scheme: str = "http", timeout: float = 5.0,
                 session: requests.Session | None = None):
        self.host = host
        self.base = f"{scheme}://{host}:{port}"
        self.username = username
        self._password = md5_password(password)
        self.timeout = timeout
        self.http = session or requests.Session()

    # ------------------------------------------------------------------ plumbing
    @staticmethod
    def path(resource: str, channel: int | None = None) -> str:
        p = "/HAPI/V1.0"
        if channel is not None:
            p += f"/Channels/{int(channel)}"
        return p + "/" + resource.strip("/")

    def _auth(self) -> dict[str, str]:
        return {"username": self.username, "password": self._password}

    def _parse(self, resp: requests.Response, path: str) -> Any:
        if resp.status_code in (401, 403):
            raise HapiError(resp.status_code, "not authorized", path)
        try:
            doc = resp.json()
        except ValueError:
            raise HapiError(resp.status_code, f"reply is not JSON: {resp.text[:120]!r}", path) from None
        r = doc.get("Response") if isinstance(doc, dict) else None
        if not isinstance(r, dict):
            raise HapiError(resp.status_code, "reply has no Response object", path)
        code = r.get("ResponseCode")
        if code != 0:
            raise HapiError(code, str(r.get("ResponseString") or "failed"), path)
        data = r.get("Data")
        if data == "null":
            data = None
        if data is None:
            extra = {k: v for k, v in r.items() if k not in _STANDARD_KEYS}
            if extra:
                return extra
        return data

    def get(self, resource: str, params: dict | None = None, channel: int | None = None,
            timeout: float | None = None) -> Any:
        path = self.path(resource, channel)
        q = {**(params or {}), **self._auth()}
        resp = self.http.get(self.base + path, params=q, timeout=timeout or self.timeout)
        return self._parse(resp, path)

    def put(self, resource: str, body: dict, channel: int | None = None) -> Any:
        """Set calls. The doc's PUT examples send JSON with credentials inside;
        the smart/* ones label it form-urlencoded, the system/* ones as JSON."""
        path = self.path(resource, channel)
        ctype = _FORM if resource.startswith("smart/") else "application/json"
        payload = json.dumps({**self._auth(), **body})
        resp = self.http.put(self.base + path, data=payload, headers={"Content-Type": ctype},
                             timeout=self.timeout)
        return self._parse(resp, path)

    def post(self, resource: str, body: dict, channel: int | None = None) -> Any:
        """Event subscription calls: credentials in the query, JSON body,
        Content-Type form-urlencoded (doc 1.2 and 3.2)."""
        path = self.path(resource, channel)
        resp = self.http.post(self.base + path, params=self._auth(), data=json.dumps(body),
                              headers={"Content-Type": _FORM}, timeout=self.timeout)
        return self._parse(resp, path)

    # ------------------------------------------------------------------ system info
    def device_info(self, channel: int | None = None) -> dict:
        return self.get("sysinfo/device_info", channel=channel) or {}

    def function_list(self) -> set[str]:
        """Supported APIs, normalised to 'ptz_ctrl/move' style lower-case paths."""
        out = set()
        for item in self.get("sysinfo/functionlist") or []:
            api = str(item.get("api", "")) if isinstance(item, dict) else str(item)
            api = re.sub(r"^/HAPI/V1\.0(\[/Channels/ID\]|/Channels/\d+)?/", "", api, flags=re.I)
            if api:
                out.add(api.strip("/").lower())
        return out

    def capability(self) -> set[str]:
        return {str(c.get("caps")) for c in self.get("sysinfo/capability") or [] if isinstance(c, dict)}

    def rtsp_urls(self) -> dict[str, str]:
        return dict(self.get("sysinfo/rtspurl") or {})

    def reboot(self) -> None:
        self.get("sysman/reboot")

    # ------------------------------------------------------------------ snapshot
    def snapshot(self, main_stream: bool = True, channel: int | None = None) -> bytes:
        """JPEG straight from the camera (no video compression). stream=1 is the
        main stream, stream=0 the sub stream (doc 2.5.1)."""
        path = self.path("snapshot.cgi", channel)
        q = {"stream": 1 if main_stream else 0, **self._auth()}
        resp = self.http.get(self.base + path, params=q, timeout=self.timeout)
        if resp.content[:2] == b"\xff\xd8":
            return resp.content
        self._parse(resp, path)          # raises the camera's error message
        raise HapiError(resp.status_code, "reply is not a JPEG", path)

    # ------------------------------------------------------------------ PTZ (joystick)
    def ptz_move(self, direction: str, speed: int = 5, autostop_ms: int = 0,
                 channel: int | None = None) -> Any:
        if direction not in PTZ_DIRECTIONS:
            raise ValueError(f"direction must be one of {PTZ_DIRECTIONS}")
        speed = int(min(max(speed, 1), 10))
        autostop_ms = int(min(max(autostop_ms, 0), 1000))
        return self.get("ptz_ctrl/move", {"direction": direction, "speed": speed, "autostop": autostop_ms},
                        channel, timeout=self.timeout + autostop_ms / 1000)

    def ptz_stop(self, channel: int | None = None) -> Any:
        return self.get("ptz_ctrl/stop", channel=channel)

    def _timed(self, resource: str, direction: str, allowed: tuple, autostop_ms: int,
               channel: int | None) -> Any:
        if direction not in allowed:
            raise ValueError(f"direction must be one of {allowed}")
        autostop_ms = int(min(max(autostop_ms, 0), 1000))
        return self.get(resource, {"direction": direction, "autostop": autostop_ms}, channel,
                        timeout=self.timeout + autostop_ms / 1000)

    def zoom(self, direction: str, autostop_ms: int = 0, channel: int | None = None) -> Any:
        return self._timed("ptz_ctrl/zoom", direction, ("in", "out"), autostop_ms, channel)

    def focus(self, direction: str, autostop_ms: int = 0, channel: int | None = None) -> Any:
        return self._timed("ptz_ctrl/focus", direction, ("near", "far"), autostop_ms, channel)

    def iris(self, direction: str, autostop_ms: int = 0, channel: int | None = None) -> Any:
        return self._timed("ptz_ctrl/iris", direction, ("open", "close"), autostop_ms, channel)

    def preset(self, method: str, number: int, channel: int | None = None) -> Any:
        if method not in ("set", "call", "delete"):
            raise ValueError("method must be set, call or delete")
        if not 1 <= int(number) <= 255:
            raise ValueError("preset number must be 1..255")
        return self.get("ptz_ctrl/preset", {"method": method, "presetno": int(number)}, channel)

    def adv_functions(self, channel: int | None = None) -> list[str]:
        return [str(f.get("functionname")) for f in self.get("ptz_ctrl/advfunction/get", channel=channel) or []
                if isinstance(f, dict)]

    def adv_exec(self, name: str, channel: int | None = None) -> Any:
        return self.get("ptz_ctrl/advfunction/exec", {"functionname": name}, channel)

    def serial_send(self, data: bytes, seq: int = 0, channel: int | None = None) -> bytes:
        """Raw bytes to the PTZ board's serial port (<= 58 bytes). Answers from the
        board, if any, arrive as UartDataTransparent events."""
        if len(data) > 58:
            raise ValueError("at most 58 bytes per call")
        r = self.get("ptz_ctrl/transparent", {"seq": int(seq), "data": base64.b64encode(data).decode()}, channel)
        return base64.b64decode((r or {}).get("data", "") or b"")

    # ------------------------------------------------------------------ configs
    def get_config(self, name: str, channel: int | None = None, **params) -> Any:
        return self.get(CONFIG_PAGES[name] + "/get", params or None, channel)

    def set_config(self, name: str, values: dict, channel: int | None = None) -> Any:
        return self.put(CONFIG_PAGES[name] + "/set", values, channel)

    def update_config(self, name: str, change: Callable[[dict], None], channel: int | None = None) -> dict:
        """Read a config page, let `change` edit it in place, then write back only
        the top-level keys that changed, each with its full sub-object (the doc
        requires every member of a structured item to be present)."""
        before = self.get_config(name, channel) or {}
        after = json.loads(json.dumps(before))
        change(after)
        diff = {k: v for k, v in after.items() if before.get(k) != v}
        if diff:
            self.set_config(name, diff, channel)
        return after

    def smart_capability(self) -> dict:
        return dict(self.get("smart/capability") or {})

    def detection_types(self) -> list[str]:
        """Target types the on-camera AI really supports (datasheet says humans,
        the vendor says humans and vehicles; this is the camera's own answer)."""
        out = []
        for item in self.get("smart/objectdetect/capability") or []:
            t = item.get("type") if isinstance(item, dict) else item
            out.append(TARGET_TYPES.get(int(t), str(t)))
        return out

    def video_streams(self, channel: int | None = None) -> list[dict]:
        cfg = self.get_config("video", channel) or {}
        streams = cfg.get("stream") if isinstance(cfg, dict) else cfg
        return list(streams or [])

    # ------------------------------------------------------------------ setup helpers
    def disable_builtin_tracking(self, channel: int | None = None) -> list[str]:
        """Stop the camera's own tracking so it never fights our software."""
        done = []
        try:
            if "TrackAutoOff" in self.adv_functions(channel):
                self.adv_exec("TrackAutoOff", channel)
                done.append("TrackAutoOff")
        except HapiError:
            pass

        def off(cfg: dict) -> None:
            action = cfg.get("AlarmAction")
            if isinstance(action, dict):
                for key in ("track_human_enable", "auto_zoom_enable", "gunball_track_mode"):
                    if action.get(key):
                        action[key] = 0
                        done.append(key)
        self.update_config("objectdetect", off, channel)
        return done

    def enable_event_reporting(self, types: tuple[str, ...] = ("HUMAN", "CAR"),
                               channel: int | None = None) -> dict:
        """Switch on detection for `types` and its 'report to alarm server' output,
        which subscriptions need (it is off in the doc's own example)."""
        def on(cfg: dict) -> None:
            cfg["enable"] = 1
            flag = cfg.get("ArmingFlag")
            if isinstance(flag, dict) and flag.get("ArmingMode", 0) == 0:
                flag["ArmingMode"] = 1
            tt = cfg.get("type")
            if isinstance(tt, dict):
                for t in types:
                    if t in tt:
                        tt[t] = 1
            action = cfg.get("AlarmAction")
            if isinstance(action, dict):
                server = action.get("AlarmServer")
                if isinstance(server, dict):
                    server["ArmingMode"] = 1
                else:
                    action["AlarmServer"] = {"ArmingMode": 1, "TimeSpanNum": 0, "DayPlan": []}
        return self.update_config("objectdetect", on, channel)

    def set_lights(self, *, work_mode: int | None = None, ctrl_mode: int | None = None,
                   brightness: int | None = None, channel: int | None = None) -> Any:
        """work_mode 0 IR / 1 white / 2 smart dual; ctrl_mode 0 auto, 6 always on,
        7 always off (doc 2.8)."""
        values = {}
        if work_mode is not None:
            values["led_work_mode"] = int(work_mode)
        if ctrl_mode is not None:
            values["led_ctrl_mode"] = int(ctrl_mode)
        if brightness is not None:
            values["led_brightness_mode"] = 1
            values["led_brightness_value"] = int(min(max(brightness, 0), 100))
        return self.set_config("light", values, channel)

    # ------------------------------------------------------------------ events
    def event_register(self, server: str, port: int, duration: int = 300, event_types: str = "all",
                       prefix: str = "") -> dict:
        duration = int(min(max(duration, 30), 3600))
        server_type = 0 if re.match(r"^\d+\.\d+\.\d+\.\d+$", server) else 1
        return dict(self.post("Event/subscription/regist", {
            "ServerType": server_type, "ServerName": server, "Port": int(port),
            "Duration": duration, "PostURLPrefix": prefix, "EventType": event_types}) or {})

    def event_refresh(self, sub_id: int, duration: int = 300) -> dict:
        duration = int(min(max(duration, 30), 3600))
        return dict(self.post("Event/subscription/refresh", {"ID": int(sub_id), "Duration": duration}) or {})

    def event_delete(self, sub_id: int) -> None:
        self.post("Event/subscription/delete", {"ID": int(sub_id)})

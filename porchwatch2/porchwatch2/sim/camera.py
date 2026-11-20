"""Virtual BC-IQ56YDW-8MQ: the same HTTP API, ONVIF PTZ and alert pushes as the
real camera (as far as the vendor documents describe them), over a simulated
street. Channel 1 = fixed wide lens, channel 2 = zoom ball.

Run it standalone:  python -m porchwatch2.sim.camera --port 8080
Things the documents leave open are switches here, so the software can be
tested against every answer the real camera might give:
  detect_vehicles   datasheet says humans only, the vendor says humans + vehicles
  report_position   whether ONVIF GetStatus returns the real position
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import itertools
import json
import logging
import math
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import requests

from ..camera.onvif import check_security_header, find
from .ptz import (PTZLimits, SimPTZ, norm_to_pan, norm_to_tilt, norm_to_zoom, pan_to_norm,
                  tilt_to_norm, zoom_to_norm)
from .scene import Scene, View

log = logging.getLogger(__name__)

WIDE_HFOV = 71.0          # 3.6 mm on a 1/3" sensor (2560 px x 2 um)
WIDE_TILT = 8.0
SENSOR_W_MM = 5.12

_ADV_FUNCTIONS = ["PTZReset", "GuardPositionOff", "GuardPositionOn", "TrackAutoOff", "TrackAutoOn"]
_FUNCTIONS = ["uid/getuid", "uid/keep_alive", "sysinfo/device_info", "sysinfo/functionlist",
              "sysinfo/capability", "sysinfo/rtspurl", "sysman/reboot", "systime/gettime",
              "ptz_ctrl/stop", "ptz_ctrl/move", "ptz_ctrl/preset", "ptz_ctrl/zoom", "ptz_ctrl/focus",
              "ptz_ctrl/iris", "ptz_ctrl/advfunction/exec", "ptz_ctrl/advfunction/get",
              "system/light/get", "system/light/set", "system/image/get", "system/image/set",
              "system/video/get", "system/video/set", "system/audio/get", "system/audio/set",
              "Smart/capability", "Smart/objectdetect/capability", "Smart/objectdetect/get",
              "Smart/objectdetect/set", "Event/subscription/regist", "Event/subscription/refresh",
              "Event/subscription/delete"]


def _arming():
    return {"ArmingMode": 1, "WeekPlan": {f"day{i}": 16777215 for i in range(7)}}


def _off_action():
    return {"ArmingMode": 0, "TimeSpanNum": 0, "DayPlan": []}


def default_configs() -> dict:
    return {
        "light": {"led_ctrl_mode": 0, "led_work_mode": 2, "light_open_brightness": 3, "light_off_sensitivity": 50,
                  "led_brightness_mode": 0, "led_brightness_value": 100, "night_starttime": "18:00:00",
                  "night_endtime": "06:00:00"},
        "image": {"brightness": 128, "contrast": 128, "saturation": 128, "sharpness": 128, "tvsystem": 0,
                  "hflip": 0, "vflip": 0, "rotate": 0},
        "audio": {"enable": 1, "sampleRate": 8000, "encodeFormat": "G.711", "bitRate": 64000, "volume_capture": 90,
                  "volume_play": 95, "amplify": 1, "aec_enable": 1, "mute_ptz_turn": 0},
        "video": {"stream": [
            {"enable": 1, "streamID": 1, "encodeFormat": "H265", "resolution": "2560X1440", "bitRateControl": "VBR",
             "gop": 40, "frameRate": 20, "bitRate": 4096, "bitRateQuality": 0, "qp_enable": 0, "qp_min": 30, "qp_max": 51},
            {"enable": 1, "streamID": 2, "encodeFormat": "H265", "resolution": "D1", "bitRateControl": "VBR",
             "gop": 60, "frameRate": 30, "bitRate": 512, "bitRateQuality": 0, "qp_enable": 0, "qp_min": 30, "qp_max": 51}]},
        "objectdetect": {
            "enable": 1, "type": {"CAR": 0, "MOTO": 0, "ELECTRICBICYCLE": 0, "BICYCLE": 0, "HUMAN": 1, "FACE": 0,
                                  "NONMOTO_VEHICLE": 0, "FIRE": 0, "FALLINGOBJECT": 0},
            "ArmingFlag": _arming(),
            "Polygon": {"PointCnt": 4, "Points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 100}, {"x": 0, "y": 100}]},
            "Sensitivity": 5, "AlarmThreshold": 75, "minTargetRate": 0, "nonMotionFilter": 1,
            "AlarmAction": {"LightAction": _off_action(), "BRAlarmAction": _off_action(), "AlarmServer": _off_action(),
                            "AlarmPush": _off_action(),
                            "AudioAction": {"times": 1, "intervalsecnods": 10, "filename": "/opt/ch/attention_en.mp3",
                                            "ArmingSetting": {"ArmingMode": 0, "TimeSpanNum": 1,
                                                              "DayPlan": [{"startTime": "00:00:00", "endTime": "23:59:59"}]}},
                            "IOOutputAction": {"channelCnt": 4, "channels": [{"portIndex": i, "enable": 0} for i in range(1, 5)]},
                            "draw_rect_enable": 1, "draw_human_enable": 1, "track_human_enable": 1,
                            "rect_twinkle_enable": 0, "auto_zoom_enable": 1, "gunball_track_mode": 1}},
    }


class VirtualCamera:
    def __init__(self, *, host: str = "127.0.0.1", port: int = 0, username: str = "admin",
                 password: str = "123456", detect_vehicles: bool = False, report_position: bool = True,
                 limits: PTZLimits | None = None, realtime: bool = True, fps: float = 20.0, seed: int = 1):
        self.username = username
        self.password = password
        self.detect_vehicles = detect_vehicles
        self.report_position = report_position
        self.lock = threading.RLock()
        self.scene = Scene(seed=seed)
        self.ptz = SimPTZ(limits=limits, pan=0.0, tilt=WIDE_TILT, zoom=1.0)
        self.configs = default_configs()
        self.exposure = {"mode": "AUTO", "us": 2000.0}
        self.builtin_tracking = True
        self.subscriptions: dict[int, dict] = {}
        self._sub_ids = itertools.count(1)
        self._uids: dict[str, float] = {}
        self._visible: dict[int, str] = {}          # id -> kind currently alarmed
        self.pushed: list[dict] = []                # alerts sent (for tests)
        self.requests: list[str] = []               # HAPI paths called (for tests)
        self.fps = fps
        self._stop = threading.Event()
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.server.daemon_threads = True
        self.host, self.port = self.server.server_address[:2]
        self._threads = [threading.Thread(target=self.server.serve_forever, name="vcam-http", daemon=True)]
        if realtime:
            self._threads.append(threading.Thread(target=self._run_world, name="vcam-world", daemon=True))

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "VirtualCamera":
        for t in self._threads:
            t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.server.shutdown()
        self.server.server_close()

    def _run_world(self) -> None:
        last = time.monotonic()
        while not self._stop.wait(1.0 / self.fps):
            now = time.monotonic()
            self.tick(now - last)
            last = now

    def tick(self, dt_s: float) -> None:
        with self.lock:
            self.scene.step(dt_s)
        self._check_alarms()

    # ------------------------------------------------------------------ views
    def view(self, channel: int, width: int = 1280, height: int = 720) -> View:
        if channel == 2:
            st = self.ptz.state()
            return View(st["pan"], st["tilt"], self.ptz.hfov(SENSOR_W_MM), width, height)
        return View(0.0, WIDE_TILT, WIDE_HFOV, width, height)

    def frame(self, channel: int, width: int = 1280, height: int = 720):
        v = self.view(channel, width, height)
        blur = defocus = 0.0
        if channel == 2:
            st = self.ptz.state()
            px_per_deg = width / v.hfov
            blur = st["pan_speed"] * px_per_deg * self.exposure["us"] / 1e6
            defocus = 1.0 - self.ptz.sharpness()
        with self.lock:
            return self.scene.render(v, blur_px=blur, defocus=defocus)

    def ground_truth(self, channel: int, width: int = 1280, height: int = 720) -> list[dict]:
        with self.lock:
            return self.scene.boxes(self.view(channel, width, height))

    # ------------------------------------------------------------------ alarms
    def _alarm_server_on(self) -> bool:
        od = self.configs["objectdetect"]
        return bool(od.get("enable")) and od["AlarmAction"]["AlarmServer"].get("ArmingMode", 0) != 0

    def _check_alarms(self) -> None:
        od = self.configs["objectdetect"]["type"]
        seen = {}
        for b in self.ground_truth(1, 640, 360):
            if b["kind"] == "person" and od.get("HUMAN"):
                seen[b["id"]] = "HumanShapeDetect"
            elif b["kind"] == "car" and self.detect_vehicles and od.get("CAR"):
                seen[b["id"]] = "VehicleDetect"
        started = {i: k for i, k in seen.items() if i not in self._visible}
        ended = {i: k for i, k in self._visible.items() if i not in seen}
        self._visible = seen
        if not self._alarm_server_on():
            return
        for active, group in ((True, started), (False, ended)):
            for sub_type in set(group.values()):
                self._push({"AlarmType": "ObjectDetect", "AlarmSubType": sub_type,
                            "TimeStamp": int(time.time()), "ChannelNo": 1, "NotificationType": 0,
                            "OccurFlag": "true" if active else "false", "DeviceID": "VCAM0000000000A1",
                            "Picture": self._picture() if active else None, "AlarmInfo": None})

    def _picture(self) -> dict:
        ok, jpg = cv2.imencode(".jpg", self.frame(1, 640, 360), [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = base64.b64encode(jpg.tobytes()).decode()
        return {"size": len(jpg), "data": data}

    def _push(self, msg: dict) -> None:
        now = time.time()
        for sub in list(self.subscriptions.values()):
            if sub["until"] < now:
                continue
            types = sub["types"]
            if types not in ("", "all") and msg["AlarmType"] not in [t.strip() for t in types.split(",")]:
                continue
            url = f"http://{sub['server']}:{sub['port']}{sub['prefix']}/HAPI/V1.0/Event/Notification"
            self.pushed.append(msg)
            threading.Thread(target=self._post, args=(url, json.dumps(msg)), daemon=True).start()

    @staticmethod
    def _post(url: str, body: str) -> None:
        try:
            requests.post(url, data=body, timeout=3, headers={
                "Content-Type": "application/x-www-form-urlencoded", "User-Agent": "ANJVISION HAPI NOTIFY"})
        except requests.RequestException as e:
            log.debug("alert push to %s failed: %s", url, e)

    # ------------------------------------------------------------------ HTTP
    def _handler(self):
        cam = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                log.debug("vcam: " + fmt, *args)

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> bytes:
                n = int(self.headers.get("Content-Length", "0") or 0)
                return self.rfile.read(n) if n > 0 else b""

            def do_GET(self):
                self._hapi("GET", b"")

            def do_PUT(self):
                self._hapi("PUT", self._body())

            def do_POST(self):
                body = self._body()
                if self.path.startswith("/onvif/"):
                    code, xml = cam.onvif(body)
                    return self._send(code, xml.encode(), "application/soap+xml; charset=utf-8")
                self._hapi("POST", body)

            def _hapi(self, method: str, body: bytes) -> None:
                u = urlparse(self.path)
                q = {k: v[-1] for k, v in parse_qs(u.query).items()}
                if body and method == "PUT":
                    try:
                        q.update(json.loads(body))
                    except ValueError:
                        return self._send(400, b"", "text/plain")
                payload = {}
                if body and method == "POST":
                    try:
                        payload = json.loads(body)
                    except ValueError:
                        return self._send(400, b"", "text/plain")
                res = cam.hapi(method, u.path, q, payload)
                if isinstance(res, bytes):
                    return self._send(200, res, "image/jpeg")
                self._send(200, json.dumps(res).encode(), "application/json")

        return Handler

    # ------------------------------------------------------------------ HAPI
    def _authorized(self, q: dict) -> bool:
        if "uid" in q:
            return self._uids.get(str(q["uid"]), 0) > time.time()
        user, pw = str(q.get("username", "")), str(q.get("password", ""))
        good = {self.password.lower(), hashlib.md5(self.password.encode()).hexdigest()}
        return user.lower() == self.username.lower() and pw.lower() in good

    @staticmethod
    def _resp(path: str, data=None, code: int = 0, text: str = "Succeed") -> dict:
        return {"Response": {"ResponseURL": path, "SessionID": "", "ResponseCode": code,
                             "ResponseString": text, "Data": "null" if data is None else data}}

    def hapi(self, method: str, path: str, q: dict, payload: dict):
        parts = path.split("/")
        if parts[:3] != ["", "HAPI", "V1.0"]:
            return self._resp(path, code=-1, text="unknown url")
        rest = parts[3:]
        channel = 1
        if len(rest) >= 2 and rest[0].lower() == "channels":
            channel, rest = int(rest[1]) or 1, rest[2:]
        res = "/".join(rest).lower()
        self.requests.append(res)
        if res == "uid/getuid":
            if not self._authorized(q):
                return self._resp(path, code=-2, text="auth failed")
            uid = format(int(time.time() * 1000) & 0xFFFFFFF, "X")
            self._uids[uid] = time.time() + 60
            r = self._resp(path)
            r["Response"]["SessionID"] = uid
            return r
        if not self._authorized(q):
            return self._resp(path, code=-2, text="auth failed")
        if "uid" in q:
            self._uids[str(q["uid"])] = time.time() + 60
        try:
            return self._route(method, path, res, channel, q, payload)
        except (ValueError, KeyError, TypeError):
            return self._resp(path, code=-1, text="params error")

    def _route(self, method, path, res, channel, q, payload):
        ok = lambda data=None: self._resp(path, data)              # noqa: E731
        bad = lambda: self._resp(path, code=-1, text="params error")  # noqa: E731
        if res == "uid/keep_alive":
            return ok()
        if res == "sysinfo/device_info":
            return ok({"SN": "VCAM0000000000A1", "device_type": "BC-IQ56YDW-8MQ", "model": "VIRTUAL",
                       "ether": "F0:00:00:00:00:A1", "fsversion": "VIRTUAL_RTMP-H5 V0.1"})
        if res == "sysinfo/functionlist":
            return ok([{"api": f"/HAPI/V1.0[/Channels/ID]/{f}"} for f in _FUNCTIONS])
        if res == "sysinfo/capability":
            caps = ["ptz_control", "audio_action", "double_led_panel", "ircut_setting", "VideoPD", "twolens_stitch"]
            if self.detect_vehicles:
                caps.append("VehicleCar")
            return ok([{"caps": c} for c in caps])
        if res == "sysinfo/rtspurl":
            return ok({"ch1_main": f"rtsp://{self.host}:554/stream0", "ch1_sub": f"rtsp://{self.host}:554/stream1",
                       "ch2_main": f"rtsp://{self.host}:554/stream2", "ch2_sub": f"rtsp://{self.host}:554/stream3"})
        if res == "snapshot.cgi":
            big = str(q.get("stream", "1")) == "1"
            w, h = (2560, 1440) if big else (704, 480)
            okk, jpg = cv2.imencode(".jpg", self.frame(channel, w, h), [cv2.IMWRITE_JPEG_QUALITY, 90])
            return jpg.tobytes()
        if res.startswith("ptz_ctrl/"):
            return self._ptz(res[len("ptz_ctrl/"):], q, ok, bad)
        if res == "smart/capability":
            return ok({"SupportTargetDetect": 1, "SupportFaceFd": 0, "SupportFaceFr": 0, "SupportLPR": 0,
                       "SupportVG": 0, "SupportRegionAI": 0, "SupportAudioDetect": 1, "SupportCoverDetect": 1})
        if res == "smart/objectdetect/capability":
            return ok([{"type": 4}] + ([{"type": 0}] if self.detect_vehicles else []))
        for name, page in (("light", "system/light"), ("image", "system/image"), ("audio", "system/audio"),
                           ("objectdetect", "smart/objectdetect")):
            if res == page + "/get":
                return ok(self.configs[name])
            if res == page + "/set":
                values = {k: v for k, v in q.items() if k not in ("username", "password", "uid")}
                if not values:
                    return bad()
                cfg = self.configs[name]
                for k, v in values.items():
                    if k not in cfg:
                        return bad()
                    cfg[k] = type(cfg[k])(v) if isinstance(cfg[k], (int, float)) and not isinstance(v, dict) else v
                if name == "objectdetect":
                    act = cfg["AlarmAction"]
                    self.builtin_tracking = bool(act.get("track_human_enable") or act.get("auto_zoom_enable")
                                                 or act.get("gunball_track_mode"))
                return ok(cfg)
        if res == "system/video/get":
            r = self._resp(path)
            r["Response"]["stream"] = self.configs["video"]["stream"]       # as in the doc: next to Data
            return r
        if res == "event/subscription/regist":
            for sub_id, sub in self.subscriptions.items():          # same server+port: no duplicate
                if sub["server"] == payload["ServerName"] and sub["port"] == int(payload["Port"]):
                    break
            else:
                sub_id = next(self._sub_ids)
            dur = int(payload.get("Duration", 3600))
            if not 30 <= dur <= 3600:
                return bad()
            now = int(time.time())
            self.subscriptions[sub_id] = {"server": payload["ServerName"], "port": int(payload["Port"]),
                                          "prefix": payload.get("PostURLPrefix", "") or "",
                                          "types": payload.get("EventType", "all") or "all", "until": now + dur}
            return ok({"ID": sub_id, "ServerType": int(payload["Port"]), "ServerName": payload["ServerName"],
                       "ServerPort": int(payload["Port"]), "EventType": payload.get("EventType", "all"),
                       "CurrentTime": now, "TerminationTime": now + dur})
        if res == "event/subscription/refresh":
            sub = self.subscriptions.get(int(payload["ID"]))
            if sub is None:
                return bad()
            now = int(time.time())
            sub["until"] = now + int(payload.get("Duration", 3600))
            return ok({"ID": int(payload["ID"]), "CurrentTime": now, "TerminationTime": sub["until"]})
        if res == "event/subscription/delete":
            self.subscriptions.pop(int(payload["ID"]), None)
            return ok({"ID": int(payload["ID"])})
        return self._resp(path, code=-1, text="unsupported")

    def _ptz(self, op, q, ok, bad):
        dirs = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1), "left_up": (-1, -1),
                "right_up": (1, -1), "left_down": (-1, 1), "right_down": (1, 1)}
        if op == "stop":
            self.ptz.stop()
            return ok()
        if op == "move":
            d = dirs.get(q.get("direction"))
            speed = int(q.get("speed", 5))
            if d is None or not 1 <= speed <= 10:
                return bad()
            stop_ms = min(int(q.get("autostop", 0) or 0), 1000)
            self.ptz.joystick(d[0], d[1], speed / 10, stop_ms / 1000 if stop_ms else None)
            if stop_ms:
                time.sleep(stop_ms / 1000)               # the real call blocks too
            return ok({"direction": q["direction"], "speed": speed, "autostop": stop_ms})
        if op == "zoom":
            if q.get("direction") not in ("in", "out"):
                return bad()
            stop_ms = min(int(q.get("autostop", 0) or 0), 1000)
            self.ptz.zoom_move(1 if q["direction"] == "in" else -1, stop_ms / 1000 if stop_ms else None)
            if stop_ms:
                time.sleep(stop_ms / 1000)
            return ok({"direction": q["direction"], "autostop": stop_ms})
        if op in ("focus", "iris"):
            return ok({"direction": q.get("direction"), "autostop": int(q.get("autostop", 0) or 0)})
        if op == "preset":
            n = int(q.get("presetno", 0))
            if not 1 <= n <= 255 or q.get("method") not in ("set", "call", "delete"):
                return bad()
            if q["method"] == "set":
                self.ptz.save_preset(n)
            elif q["method"] == "call":
                if not self.ptz.goto_preset(n):
                    return bad()
            else:
                self.ptz.presets.pop(n, None)
            return ok()
        if op == "advfunction/get":
            return ok([{"functionname": f} for f in _ADV_FUNCTIONS])
        if op == "advfunction/exec":
            name = q.get("functionname")
            if name not in _ADV_FUNCTIONS:
                return bad()
            if name == "TrackAutoOff":
                self.builtin_tracking = False
            elif name == "TrackAutoOn":
                self.builtin_tracking = True
            elif name == "PTZReset":
                self.ptz.absolute(0.0, WIDE_TILT, 1.0)
            return ok({"functionname": name})
        if op == "transparent":
            return ok({"seq": int(q.get("seq", 0)), "data": q.get("data", "")})
        return bad()

    # ------------------------------------------------------------------ ONVIF
    def onvif(self, body: bytes) -> tuple[int, str]:
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return 400, self._fault("Sender", "not XML")
        b = find(root, "Body")
        if b is None or not len(b):
            return 400, self._fault("Sender", "no body")
        req = b[0]
        op = req.tag.rsplit("}", 1)[-1]
        now = dt.datetime.now(dt.timezone.utc)
        if op != "GetSystemDateAndTime" and not check_security_header(root, {self.username: self.password}, now):
            return 400, self._fault("Sender", "NotAuthorized")
        handler = getattr(self, f"_ov_{op}", None)
        if handler is None:
            return 400, self._fault("Receiver", f"ActionNotSupported {op}")
        try:
            return 200, self._envelope(handler(req, now))
        except (ValueError, TypeError, AttributeError) as e:
            return 400, self._fault("Sender", f"InvalidArgVal {e}")

    @staticmethod
    def _envelope(inner: str) -> str:
        return ('<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
                'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
                'xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" xmlns:timg="http://www.onvif.org/ver20/imaging/wsdl" '
                f'xmlns:tt="http://www.onvif.org/ver10/schema"><s:Body>{inner}</s:Body></s:Envelope>')

    def _fault(self, code: str, reason: str) -> str:
        return self._envelope(f"<s:Fault><s:Code><s:Value>s:{code}</s:Value></s:Code>"
                              f"<s:Reason><s:Text xml:lang=\"en\">{reason}</s:Text></s:Reason></s:Fault>")

    def _ov_GetSystemDateAndTime(self, req, now):
        return (f"<tds:GetSystemDateAndTimeResponse><tds:SystemDateAndTime><tt:UTCDateTime>"
                f"<tt:Time><tt:Hour>{now.hour}</tt:Hour><tt:Minute>{now.minute}</tt:Minute><tt:Second>{now.second}</tt:Second></tt:Time>"
                f"<tt:Date><tt:Year>{now.year}</tt:Year><tt:Month>{now.month}</tt:Month><tt:Day>{now.day}</tt:Day></tt:Date>"
                f"</tt:UTCDateTime></tds:SystemDateAndTime></tds:GetSystemDateAndTimeResponse>")

    def _ov_GetCapabilities(self, req, now):
        base = f"http://{self.host}:{self.port}/onvif"
        return (f"<tds:GetCapabilitiesResponse><tds:Capabilities>"
                f"<tt:Device><tt:XAddr>{base}/device_service</tt:XAddr></tt:Device>"
                f"<tt:Imaging><tt:XAddr>{base}/imaging_service</tt:XAddr></tt:Imaging>"
                f"<tt:Media><tt:XAddr>{base}/media_service</tt:XAddr></tt:Media>"
                f"<tt:PTZ><tt:XAddr>{base}/ptz_service</tt:XAddr></tt:PTZ>"
                f"</tds:Capabilities></tds:GetCapabilitiesResponse>")

    def _ov_GetDeviceInformation(self, req, now):
        return ("<tds:GetDeviceInformationResponse><tds:Manufacturer>Virtual</tds:Manufacturer>"
                "<tds:Model>BC-IQ56YDW-8MQ</tds:Model><tds:FirmwareVersion>V0.1</tds:FirmwareVersion>"
                "<tds:SerialNumber>VCAM0000000000A1</tds:SerialNumber><tds:HardwareId>SIM</tds:HardwareId>"
                "</tds:GetDeviceInformationResponse>")

    def _ov_GetProfiles(self, req, now):
        def prof(token, name, src, w, h, ptz):
            ptz_xml = '<tt:PTZConfiguration token="ptzcfg"><tt:Name>ptz</tt:Name></tt:PTZConfiguration>' if ptz else ""
            return (f'<trt:Profiles token="{token}" fixed="true"><tt:Name>{name}</tt:Name>'
                    f'<tt:VideoSourceConfiguration token="vsc_{src}"><tt:SourceToken>{src}</tt:SourceToken></tt:VideoSourceConfiguration>'
                    f'<tt:VideoEncoderConfiguration token="enc_{token}"><tt:Encoding>H265</tt:Encoding>'
                    f'<tt:Resolution><tt:Width>{w}</tt:Width><tt:Height>{h}</tt:Height></tt:Resolution></tt:VideoEncoderConfiguration>'
                    f'{ptz_xml}</trt:Profiles>')
        return ("<trt:GetProfilesResponse>" + prof("wide_main", "wide main", "vs_wide", 2560, 1440, False)
                + prof("wide_sub", "wide sub", "vs_wide", 704, 480, False)
                + prof("ptz_main", "ptz main", "vs_ptz", 2560, 1440, True)
                + prof("ptz_sub", "ptz sub", "vs_ptz", 704, 480, True) + "</trt:GetProfilesResponse>")

    def _ov_GetStreamUri(self, req, now):
        token = find(req, "ProfileToken").text
        idx = {"wide_main": 0, "wide_sub": 1, "ptz_main": 2, "ptz_sub": 3}[token]
        return (f"<trt:GetStreamUriResponse><trt:MediaUri><tt:Uri>rtsp://{self.host}:554/stream{idx}</tt:Uri>"
                f"</trt:MediaUri></trt:GetStreamUriResponse>")

    def _ov_GetConfigurationOptions(self, req, now):
        rng = lambda a, b: f"<tt:Min>{a}</tt:Min><tt:Max>{b}</tt:Max>"      # noqa: E731
        return ("<tptz:GetConfigurationOptionsResponse><tptz:PTZConfigurationOptions><tt:Spaces>"
                "<tt:AbsolutePanTiltPositionSpace><tt:URI>http://www.onvif.org/ver10/tptz/PanTiltSpaces/PositionGenericSpace</tt:URI>"
                f"<tt:XRange>{rng(-1, 1)}</tt:XRange><tt:YRange>{rng(-1, 1)}</tt:YRange></tt:AbsolutePanTiltPositionSpace>"
                "<tt:AbsoluteZoomPositionSpace><tt:URI>http://www.onvif.org/ver10/tptz/ZoomSpaces/PositionGenericSpace</tt:URI>"
                f"<tt:XRange>{rng(0, 1)}</tt:XRange></tt:AbsoluteZoomPositionSpace>"
                "</tt:Spaces></tptz:PTZConfigurationOptions></tptz:GetConfigurationOptionsResponse>")

    def _ov_AbsoluteMove(self, req, now):
        pos = find(req, "Position")
        pt, zm = find(pos, "PanTilt"), find(pos, "Zoom")
        spd = find(req, "Speed")
        frac = float(find(spd, "PanTilt").get("x")) if spd is not None and find(spd, "PanTilt") is not None else 1.0
        pan = norm_to_pan(float(pt.get("x"))) if pt is not None else None
        tilt = norm_to_tilt(float(pt.get("y")), self.ptz.lim) if pt is not None else None
        zoom = norm_to_zoom(float(zm.get("x")), self.ptz.lim) if zm is not None else None
        self.ptz.absolute(pan, tilt, zoom, abs(frac) or 1.0)
        return "<tptz:AbsoluteMoveResponse/>"

    def _ov_ContinuousMove(self, req, now):
        vel = find(req, "Velocity")
        pt, zm = find(vel, "PanTilt"), find(vel, "Zoom")
        vx = float(pt.get("x")) if pt is not None else 0.0
        vy = float(pt.get("y")) if pt is not None else 0.0
        if pt is not None and (vx or vy):
            frac = max(abs(vx), abs(vy))
            self.ptz.joystick(vx / frac, -vy / frac, frac, None)
        if zm is not None and float(zm.get("x")):
            self.ptz.zoom_move(1 if float(zm.get("x")) > 0 else -1, None, abs(float(zm.get("x"))))
        return "<tptz:ContinuousMoveResponse/>"

    def _ov_Stop(self, req, now):
        self.ptz.stop()
        return "<tptz:StopResponse/>"

    def _ov_GetStatus(self, req, now):
        st = self.ptz.state()
        if self.report_position:
            x, y, z = pan_to_norm(st["pan"]), tilt_to_norm(st["tilt"], self.ptz.lim), zoom_to_norm(st["zoom"], self.ptz.lim)
        else:
            x = y = z = 0.0
        mv = "MOVING" if st["moving"] else "IDLE"
        return (f'<tptz:GetStatusResponse><tptz:PTZStatus><tt:Position><tt:PanTilt x="{x:.6f}" y="{y:.6f}"/>'
                f'<tt:Zoom x="{z:.6f}"/></tt:Position><tt:MoveStatus><tt:PanTilt>{mv}</tt:PanTilt>'
                f'<tt:Zoom>{mv}</tt:Zoom></tt:MoveStatus><tt:UtcTime>{now.strftime("%Y-%m-%dT%H:%M:%SZ")}</tt:UtcTime>'
                f'</tptz:PTZStatus></tptz:GetStatusResponse>')

    def _ov_GetImagingSettings(self, req, now):
        return (f"<timg:GetImagingSettingsResponse><timg:ImagingSettings><tt:Exposure><tt:Mode>{self.exposure['mode']}</tt:Mode>"
                f"<tt:ExposureTime>{self.exposure['us']:.0f}</tt:ExposureTime></tt:Exposure></timg:ImagingSettings>"
                f"</timg:GetImagingSettingsResponse>")

    def _ov_SetImagingSettings(self, req, now):
        exp = find(req, "Exposure")
        if exp is not None:
            mode = find(exp, "Mode")
            et = find(exp, "ExposureTime")
            if mode is not None and mode.text:
                self.exposure["mode"] = mode.text.strip().upper()
            if et is not None and et.text:
                us = float(et.text)
                if not 100 <= us <= 40000:                 # 1/10000 .. 1/25 s (datasheet)
                    raise ValueError("ExposureTime out of range")
                self.exposure["us"] = us
            if self.exposure["mode"] == "AUTO":
                self.exposure["us"] = 2000.0
        return "<timg:SetImagingSettingsResponse/>"


def main() -> None:
    ap = argparse.ArgumentParser(description="Virtual BC-IQ56YDW-8MQ camera")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--password", default="123456")
    ap.add_argument("--vehicles", action="store_true", help="on-camera AI also detects vehicles")
    ap.add_argument("--traffic", type=float, default=0.1, help="new cars/people per second")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    cam = VirtualCamera(host=args.host, port=args.port, password=args.password, detect_vehicles=args.vehicles).start()
    print(f"Virtual camera on http://{cam.host}:{cam.port}  (user admin, password {args.password})")
    try:
        while True:
            time.sleep(1.0)
            if cam.scene.rng.random() < args.traffic:
                with cam.lock:
                    if cam.scene.rng.random() < 0.6:
                        cam.scene.add_car(cam.scene.rng.choice((-1, 1)), cam.scene.rng.uniform(8, 16))
                    else:
                        cam.scene.add_person(cam.scene.rng.choice((-1, 1)))
    except KeyboardInterrupt:
        cam.stop()


if __name__ == "__main__":
    main()

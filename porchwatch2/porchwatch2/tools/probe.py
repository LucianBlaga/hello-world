"""First-contact check for the real camera. Answers the open questions itself:

  * which APIs, detections and channels it really has, and the RTSP addresses
  * whether ONVIF absolute moves work and whether GetStatus reports the position
  * how long a move takes and how long until the zoom picture is sharp again
  * whether alerts reach this PC

Usage (from the porchwatch2 folder):
    python -m porchwatch2.tools.probe --host 192.168.1.50 --password YOURPASS
Add --setup to also switch off the camera's own tracking and switch on alert
reporting. The zoom lens WILL move during the test. Results are printed and
saved to probe-report.json plus a few snapshots.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..camera.events import EventReceiver, Subscription
from ..camera.hapi import HapiClient, HapiError, is_default_password
from ..camera.onvif import OnvifClient, OnvifError

AF_WINDOW_S = 4.0          # how long to watch the picture sharpen after a move


def sharpness(jpeg: bytes) -> float:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    h, w = img.shape
    crop = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


class Report:
    def __init__(self):
        self.data: dict = {}

    def add(self, key: str, value, label: str | None = None) -> None:
        self.data[key] = value
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        print(f"  {label or key}: {text[:300]}")


def probe_hapi(cam: HapiClient, rep: Report, out: Path) -> None:
    print("\n== HTTP API")
    calls = [("device", cam.device_info), ("functions", lambda: sorted(cam.function_list())),
             ("capabilities", lambda: sorted(cam.capability())), ("rtsp", cam.rtsp_urls),
             ("smart", cam.smart_capability), ("detection_types", cam.detection_types),
             ("video_streams", cam.video_streams), ("ptz_functions", cam.adv_functions),
             ("lights", lambda: cam.get_config("light"))]
    for key, fn in calls:
        try:
            rep.add(key, fn())
        except (HapiError, OSError, ValueError) as e:
            rep.add(key, f"FAILED: {e}")
    for ch in (None, 1, 2):
        try:
            jpg = cam.snapshot(True, ch)
            name = out / f"snapshot-ch{ch or 'default'}.jpg"
            name.write_bytes(jpg)
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            rep.add(f"snapshot_ch{ch}", f"{img.shape[1]}x{img.shape[0]} saved to {name.name}")
        except (HapiError, OSError) as e:
            rep.add(f"snapshot_ch{ch}", f"FAILED: {e}")


def probe_onvif(ov: OnvifClient, cam: HapiClient, rep: Report, ptz_channel: int | None) -> None:
    print("\n== ONVIF")
    try:
        ov.connect()
        rep.add("onvif_services", ov.xaddr)
        rep.add("onvif_clock_offset_s", round(ov.clock_offset, 1))
        rep.add("onvif_device", ov.device_information())
        profiles = ov.profiles()
        rep.add("onvif_profiles", [p.__dict__ for p in profiles])
        rep.add("onvif_stream_uris", {p.token: ov.stream_uri(p.token) for p in profiles})
        prof = ov.ptz_profile()
    except (OnvifError, OSError, ValueError) as e:
        rep.add("onvif", f"FAILED: {e}")
        return
    try:
        spaces = ov.ptz_spaces(prof.ptz_config)
        rep.add("ptz_spaces", {"pan": spaces.pan, "tilt": spaces.tilt, "zoom": spaces.zoom, **spaces.raw})
    except (OnvifError, OSError, ValueError, AttributeError) as e:
        rep.add("ptz_spaces", f"FAILED: {e}")
    print("\n== PTZ timing (the zoom lens moves now)")
    targets = [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.25, 0.0, 0.5), (-0.25, 0.1, 0.5), (0.0, 0.0, 0.0)]
    results = []
    for pan, tilt, zoom in targets:
        t0 = time.monotonic()
        try:
            ov.absolute_move(prof.token, pan, tilt, zoom)
            st = ov.wait_idle(prof.token, timeout_s=15)
        except (OnvifError, OSError) as e:
            results.append({"target": (pan, tilt, zoom), "error": str(e)})
            continue
        moved_s = time.monotonic() - t0
        entry = {"target": (pan, tilt, zoom), "idle_after_s": round(moved_s, 2),
                 "reported": None if st is None else (st.pan, st.tilt, st.zoom)}
        # autofocus: snapshot sharpness until it stops improving
        samples = []
        t_af = time.monotonic()
        while time.monotonic() - t_af < AF_WINDOW_S:
            try:
                samples.append((round(time.monotonic() - t_af, 2), round(sharpness(cam.snapshot(True, ptz_channel)), 1)))
            except (HapiError, OSError):
                break
        if samples:
            best = max(s for _, s in samples)
            entry["sharp_after_s"] = next(t for t, s in samples if s >= 0.9 * best)
        results.append(entry)
    rep.add("ptz_moves", results)
    # judge only moves away from (0, 0, 0): a camera that always answers zeros
    # would otherwise look right on the moves back home
    checks = [(r["reported"], r["target"]) for r in results
              if r.get("reported") and any(r["target"]) and None not in r["reported"]]
    readback = bool(checks) and all(abs(rp[0] - t[0]) < 0.05 and abs(rp[1] - t[1]) < 0.05 and abs(rp[2] - t[2]) < 0.05
                                    for rp, t in checks)
    rep.add("ptz_position_readback", "yes" if readback else "no (GetStatus does not follow the moves)")


def probe_alerts(cam: HapiClient, rep: Report, seconds: float, port: int) -> None:
    print(f"\n== Alerts (walk in front of the camera within {seconds:.0f} s)")
    got: queue.Queue = queue.Queue()
    rx = EventReceiver(got.put, port=port, allowed_ips={cam.host}).start()
    sub = Subscription(cam, rx.port, duration=120)
    try:
        sub.renew()
        rep.add("alert_subscription", f"id {sub.sub_id}")
        end = time.monotonic() + seconds
        seen = []
        while time.monotonic() < end:
            try:
                ev = got.get(timeout=1.0)
            except queue.Empty:
                continue
            seen.append({"type": ev.alarm_type, "sub": ev.sub_type, "active": ev.active, "channel": ev.channel,
                         "picture": bool(ev.picture)})
            print(f"    alert: {seen[-1]}")
        rep.add("alerts", seen or "none received (is 'report to alarm server' on? run with --setup)")
    except (HapiError, OSError) as e:
        rep.add("alert_subscription", f"FAILED: {e}")
    finally:
        sub.stop(unsubscribe=True)
        rx.stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check what the camera really supports")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=80)
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--ptz-channel", type=int, default=2, help="HAPI channel of the zoom lens (guess: 2)")
    ap.add_argument("--alerts", type=float, default=30.0, help="seconds to wait for alerts (0 = skip)")
    ap.add_argument("--alert-port", type=int, default=9998)
    ap.add_argument("--setup", action="store_true", help="turn off built-in tracking, turn on alert reporting")
    ap.add_argument("--no-move", action="store_true", help="skip the PTZ timing test")
    ap.add_argument("--out", default="probe-report")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = Report()
    if is_default_password(args.password):
        print("WARNING: the camera still uses the factory password 123456. Change it in the camera's web page.")
        rep.data["default_password"] = True
    cam = HapiClient(args.host, args.user, args.password, port=args.port)
    probe_hapi(cam, rep, out)
    if args.setup:
        print("\n== Setup")
        for key, fn in (("builtin_tracking_off", cam.disable_builtin_tracking),
                        ("alert_reporting_on", lambda: (cam.enable_event_reporting(), "done")[1])):
            try:
                rep.add(key, fn())
            except (HapiError, OSError) as e:
                rep.add(key, f"FAILED: {e}")
    if not args.no_move:
        probe_onvif(OnvifClient(args.host, args.user, args.password, port=args.port), cam, rep, args.ptz_channel)
    if args.alerts > 0:
        probe_alerts(cam, rep, args.alerts, args.alert_port)
    (out / "probe-report.json").write_text(json.dumps(rep.data, indent=2, default=str))
    print(f"\nSaved {out / 'probe-report.json'}. Send me that file and the snapshots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

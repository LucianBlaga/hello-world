"""The auto-tracking brain: watch -> pick target -> follow & zoom -> capture -> home.

Only one PTZ camera exists, so while it is zoomed on someone the wide view is
gone. The controller therefore keeps each chase short (a few seconds), then
returns home to watch for the next one.
"""
from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .camera import PTZ, PTZState
from .config import Config
from .detectors import PERSON, VEHICLE, Detection, sharpness
from .storage import draw_label
from .tracker import CentroidTracker


class State(str, Enum):
    HOME = "watching"
    TRACK = "tracking"


def _expand(box, frac, w, h):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return (int(max(0, x1 - bw * frac)), int(max(0, y1 - bh * frac)),
            int(min(w, x2 + bw * frac)), int(min(h, y2 + bh * frac)))


def _crop(img, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    return img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]


@dataclass
class Target:
    kind: str
    det: Detection
    label: str
    started: float
    last_seen: float
    world: tuple                                    # (pan, tilt) degrees of aim point
    hist: deque = field(default_factory=lambda: deque(maxlen=12))  # (t, pan, tilt)
    zoomed_at: float | None = None
    feature_box: tuple | None = None                # face or plate box, frame coords
    best_face: tuple | None = None                  # (score, crop)
    best_body: tuple | None = None                  # (score, crop)
    best_frame: np.ndarray | None = None
    plate_votes: Counter = field(default_factory=Counter)
    plate_best: dict = field(default_factory=dict)  # text -> (conf, crop)
    accepted_plate: str | None = None

    def velocity(self) -> tuple[float, float]:
        """Angular velocity (deg/s) from the recent world positions."""
        if len(self.hist) < 3:
            return 0.0, 0.0
        t = np.array([h[0] for h in self.hist])
        if t[-1] - t[0] < 0.15:
            return 0.0, 0.0
        vp = np.polyfit(t - t[0], [h[1] for h in self.hist], 1)[0]
        vt = np.polyfit(t - t[0], [h[2] for h in self.hist], 1)[0]
        return float(vp), float(vt)


class Controller:
    def __init__(self, cfg: Config, ptz: PTZ, face_detector, plate_reader, storage,
                 ptz_enabled: bool = True):
        self.cfg = cfg
        self.ptz = ptz
        self.faces = face_detector
        self.plates = plate_reader
        self.storage = storage
        self._ptz_hw = ptz_enabled
        self.state = State.HOME
        self.target: Target | None = None
        self.tracker = CentroidTracker()
        self.cooldown_until = 0.0
        self.recent: list[tuple] = []   # (kind, pan, tilt, vpan, vtilt, t_end, until)
        self.last_event: dict | None = None
        self.active_until = 0.0         # something of interest in view until this time

    @property
    def ptz_enabled(self) -> bool:
        return self._ptz_hw and self.cfg.tracking_enabled

    def reset(self) -> None:
        """Drop the current target without saving (e.g. manual control took over)."""
        self.target = None
        self.state = State.HOME
        self.tracker.reset()

    # ------------------------------------------------------------ geometry
    def img_to_world(self, x, y, w, h):
        hf, vf = self.ptz.fov()
        st = self.ptz.state
        ax = math.degrees(math.atan((x - w / 2) / (w / 2) * math.tan(math.radians(hf / 2))))
        ay = math.degrees(math.atan((y - h / 2) / (h / 2) * math.tan(math.radians(vf / 2))))
        return st.pan + ax, st.tilt - ay

    def world_to_img(self, pan, tilt, w, h):
        hf, vf = self.ptz.fov()
        st = self.ptz.state
        dx = math.tan(math.radians(pan - st.pan)) / math.tan(math.radians(hf / 2))
        dy = math.tan(math.radians(st.tilt - tilt)) / math.tan(math.radians(vf / 2))
        return w / 2 + dx * w / 2, h / 2 + dy * h / 2

    # ------------------------------------------------------------ main step
    def step(self, frame: np.ndarray, dets: list[Detection], now: float | None = None) -> dict:
        now = time.time() if now is None else now
        if dets:
            self.active_until = now + 1.0
        if self.state == State.HOME:
            self._step_home(frame, dets, now)
        else:
            self._step_track(frame, dets, now)
        return self.status(now)

    def status(self, now: float) -> dict:
        t = self.target
        return {
            "state": self.state.value,
            "target": t.label if t else None,
            "zoom": round(self.ptz.state.zoom, 2),
            "pan": round(self.ptz.state.pan, 1),
            "tilt": round(self.ptz.state.tilt, 1),
            "active": self.state == State.TRACK or now < self.active_until,
            "last_event": self.last_event,
        }

    # ------------------------------------------------------------ HOME
    def _suppressed(self, kind, pan, tilt, now) -> bool:
        self.recent = [r for r in self.recent if r[6] > now]
        for k, p, t, vp, vt, t_end, _ in self.recent:
            if k != kind:
                continue
            dt = now - t_end
            if abs(pan - (p + vp * dt)) < 10 and abs(tilt - (t + vt * dt)) < 10:
                return True
        return False

    def _step_home(self, frame, dets, now):
        h, w = frame.shape[:2]
        if now < self.ptz.moving_until:
            return                      # blurred / shifting view: don't judge motion
        tracks = self.tracker.update(dets, now, w)
        if now < self.cooldown_until:
            return
        dcfg = self.cfg.detection
        candidates = []
        for tr in tracks:
            d = tr.det
            pan, tilt = self.img_to_world(*d.center, w, h)
            if self._suppressed(d.kind, pan, tilt, now):
                continue
            if d.kind == PERSON and tr.age(now) >= 0.3:
                candidates.append((1, d.height, tr))
            elif d.kind == VEHICLE and tr.travel(now, dcfg.motion_window_s) >= dcfg.motion_min_travel * w:
                candidates.append((2 if self.cfg.tracking.prefer_vehicles else 0, d.width, tr))
        if not candidates:
            return
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        tr = candidates[0][2]
        det = tr.det
        ax, ay = self._aim_point(det, None, h)
        world = self.img_to_world(ax, ay, w, h)
        self.target = Target(det.kind, det, det.label, now, now, world)
        # Seed the velocity estimate with what we saw from home, so moving
        # targets are led from the first command.
        off_x, off_y = ax - det.center[0], ay - det.center[1]
        for ts, cx, cy in list(tr.history)[-self.target.hist.maxlen:]:
            self.target.hist.append((ts, *self.img_to_world(cx + off_x, cy + off_y, w, h)))
        self.state = State.TRACK
        self.storage.log(f"Tracking {det.label}")

    # ------------------------------------------------------------ TRACK
    def _aim_point(self, det: Detection, feature_box, h):
        if feature_box is not None:
            x1, y1, x2, y2 = feature_box
            return (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1, x2, y2 = det.box
        frac = self.cfg.tracking.person_aim_y if det.kind == PERSON else self.cfg.tracking.vehicle_aim_y
        return (x1 + x2) / 2, y1 + (y2 - y1) * frac

    def _match(self, dets, w, h) -> Detection | None:
        t = self.target
        px, py = self.world_to_img(*t.world, w, h)
        best, best_d = None, 0.35 * w
        for d in dets:
            if d.kind != t.kind:
                continue
            ax, ay = self._aim_point(d, None, h)
            dist = math.hypot(ax - px, ay - py)
            if dist < best_d:
                best, best_d = d, dist
        return best

    def _step_track(self, frame, dets, now):
        t = self.target
        tc = self.cfg.tracking
        h, w = frame.shape[:2]

        det = self._match(dets, w, h)
        if det is not None:
            t.det, t.last_seen = det, now
            self._analyse(frame, det, now)
            ax, ay = self._aim_point(det, t.feature_box, h)
            t.world = self.img_to_world(ax, ay, w, h)
            t.hist.append((now, *t.world))
            if self.ptz_enabled:
                self._command(det, ax, ay, w, h, now)
            elif t.zoomed_at is None:
                t.zoomed_at = now       # fixed camera: start collecting right away

        if t.accepted_plate:
            return self._finish(now, "plate read")
        if t.best_face and t.best_face[0] >= tc.face_fill * h * 0.8:
            return self._finish(now, "face captured")
        if now - t.last_seen > tc.lost_timeout_s:
            return self._finish(now, "lost")
        if now - t.started > tc.max_track_s:
            return self._finish(now, "timeout")
        if t.zoomed_at is not None and now - t.zoomed_at > tc.capture_window_s:
            return self._finish(now, "done")

    def _command(self, det, ax, ay, w, h, now):
        t, tc, ptz = self.target, self.cfg.tracking, self.ptz
        st = ptz.state
        err_x, err_y = (ax - w / 2) / w, (ay - h / 2) / h
        goal_pan, goal_tilt = t.world
        vp, vt = t.velocity()
        new_pan, new_tilt = st.pan, st.tilt
        if abs(err_x) > tc.deadband or abs(vp) > 2:
            new_pan = st.pan + tc.gain * (goal_pan - st.pan) + vp * tc.lead_s
        if abs(err_y) > tc.deadband or abs(vt) > 2:
            new_tilt = st.tilt + tc.gain * (goal_tilt - st.tilt) + vt * tc.lead_s

        # --- zoom: how big is the thing we want (face / plate) vs. how big we want it
        zoom = st.zoom
        centred = abs(err_x) < 0.15 and abs(err_y) < 0.15
        fill, goal = self._fill(det, w, h)
        if fill is not None and centred:
            ratio = goal / max(fill, 1e-3)
            if ratio > 1.1:
                zoom = st.zoom * min(1 + tc.zoom_step, ratio)
            elif ratio < 0.8:
                zoom = st.zoom * max(1 - tc.zoom_step, ratio)
        # Never zoom so far that the whole vehicle can't be kept in frame.
        if det.kind == VEHICLE and det.width / w > tc.vehicle_fill:
            zoom = min(zoom, st.zoom * tc.vehicle_fill / (det.width / w))
        zoom = min(zoom, self.cfg.camera.max_zoom_ratio)
        at_goal = fill is not None and 0.8 <= goal / max(fill, 1e-3) <= 1.1
        at_max = zoom >= self.cfg.camera.max_zoom_ratio - 1e-3
        if t.zoomed_at is None and centred and (at_goal or at_max):
            t.zoomed_at = now
        ptz.move(PTZState(new_pan, new_tilt, zoom))

    def _fill(self, det, w, h):
        """Current and wanted size (fraction of frame) of the face or plate."""
        t, tc = self.target, self.cfg.tracking
        if det.kind == PERSON:
            if t.feature_box is not None:
                return (t.feature_box[3] - t.feature_box[1]) / h, tc.face_fill
            if det.box[3] < h * 0.97:           # whole body visible: face ~12% of height
                return 0.12 * det.height / h, tc.face_fill
            return None, tc.face_fill           # truncated & no face (walking away)
        if t.feature_box is not None:
            return (t.feature_box[2] - t.feature_box[0]) / w, tc.plate_fill
        return 0.22 * det.width / w, tc.plate_fill  # plate is ~22% of car width

    # ------------------------------------------------------------ capture
    def _analyse(self, frame, det, now):
        t, cc = self.target, self.cfg.capture
        h, w = frame.shape[:2]
        settled = now >= self.ptz.moving_until
        region = _expand(det.box, 0.1, w, h)
        crop = _crop(frame, region)
        if crop.size == 0:
            return
        ox, oy = region[0], region[1]
        body_score = det.height * det.width * (1.0 if settled else 0.5)
        if t.best_body is None or body_score > t.best_body[0]:
            t.best_body = (body_score, crop.copy())
            if cc.save_context_frame:
                t.best_frame = frame.copy()

        t.feature_box = None
        if det.kind == PERSON:
            head = crop[: max(1, int(crop.shape[0] * 0.5))]
            faces = self.faces(head) if self.faces else []
            if faces:
                (fx1, fy1, fx2, fy2), _ = max(faces, key=lambda f: f[0][3] - f[0][1])
                t.feature_box = (fx1 + ox, fy1 + oy, fx2 + ox, fy2 + oy)
                fh = fy2 - fy1
                fcrop = _crop(frame, _expand(t.feature_box, 0.4, w, h))
                sharp = sharpness(_crop(frame, t.feature_box))
                if fh >= cc.min_face_px and sharp >= cc.min_sharpness:
                    score = fh * min(sharp / cc.min_sharpness, 2.0) / 2.0
                    if t.best_face is None or score > t.best_face[0]:
                        t.best_face = (score, fcrop.copy())
        elif self.plates is not None and getattr(self.plates, "available", True):
            reads = self.plates(crop)
            if reads:
                r = max(reads, key=lambda r: r.conf)
                x1, y1, x2, y2 = r.box
                t.feature_box = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
                if r.conf >= cc.plate_min_conf and len(r.text) >= 4:
                    t.plate_votes[r.text] += 1
                    pcrop = _crop(frame, _expand(t.feature_box, 0.15, w, h))
                    if r.text not in t.plate_best or r.conf > t.plate_best[r.text][0]:
                        t.plate_best[r.text] = (r.conf, pcrop.copy())
                    if t.plate_votes[r.text] >= cc.plate_votes:
                        t.accepted_plate = r.text

    def _finish(self, now, reason):
        t = self.target
        images, meta = {}, {"kind": t.kind, "label": t.label, "reason": reason,
                            "duration_s": round(now - t.started, 1)}
        if t.kind == PERSON and t.best_face:
            images["face"] = t.best_face[1]
        if t.kind == VEHICLE and t.plate_votes:
            text, votes = t.plate_votes.most_common(1)[0]
            meta.update(plate=text, plate_votes=votes,
                        plate_confirmed=t.accepted_plate is not None,
                        plate_conf=round(t.plate_best[text][0], 3))
            images["plate"] = t.plate_best[text][1]
        if t.best_body:
            images[t.kind] = t.best_body[1]
        if t.best_frame is not None:
            images["scene"] = t.best_frame
        captured = bool(images)
        if captured:
            self.last_event = self.storage.save_event(meta, images)
        vp, vt = t.velocity()
        hold = self.cfg.tracking.recapture_after_s if (t.best_face or t.plate_votes) else 3.0
        self.recent.append((t.kind, *t.world, vp, vt, now, now + hold))
        self.target = None
        self.state = State.HOME
        self.tracker.reset()
        self.cooldown_until = now + self.cfg.tracking.cooldown_s
        if self.ptz_enabled:
            self.ptz.home()


# ------------------------------------------------------------ drawing

def annotate(frame: np.ndarray, dets: list[Detection], ctl: Controller, fps: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    s = max(1.0, w / 1280)
    for d in dets:
        color = (0, 200, 255) if d.kind == PERSON else (255, 160, 0)
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, int(2 * s))
        cv2.putText(out, f"{d.label} {d.conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, color, int(1 * s) + 1)
    t = ctl.target
    if t is not None:
        x1, y1, x2, y2 = (int(v) for v in t.det.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), int(3 * s))
        if t.feature_box is not None:
            fx1, fy1, fx2, fy2 = (int(v) for v in t.feature_box)
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (0, 255, 0), int(2 * s))
    cv2.drawMarker(out, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, int(20 * s), 1)
    st = ctl.ptz.state
    txt = f"{ctl.state.value.upper()}  pan {st.pan:+.1f}  tilt {st.tilt:+.1f}  zoom {st.zoom:.1f}x  {fps:.0f} fps"
    if t is not None and t.plate_votes:
        txt += f"  plate? {t.plate_votes.most_common(1)[0][0]}"
    draw_label(out, txt, (int(12 * s), int(30 * s)), 0.6 * s)
    return out

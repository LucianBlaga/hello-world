"""The auto-tracking brain: watch -> pick target -> follow & zoom -> capture -> home.

Only one PTZ camera exists, so while it is zoomed on someone the wide view is
gone. The controller therefore keeps each chase short (a few seconds), then
returns home to watch for the next one.
"""
from __future__ import annotations

import logging
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
from .imaging import downscale
from .storage import draw_label
from .tracker import CentroidTracker


# Detailed per-decision tracking log (written to logs/tracking.log by the app).
tlog = logging.getLogger("porchwatch.track")


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
    world_t: float = 0.0                            # when `world` was measured
    seen_zoom: float = 1.0                          # camera zoom when last seen (for size matching)
    at_max_zoom_since: float | None = None
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


class MotionMeter:
    """Tells from the picture itself whether the camera is still turning.

    Compares each (small, grey) frame with the previous one using phase
    correlation: while the gimbal turns, the whole scene slides sideways.
    Our timing model of the gimbal can be wrong; the picture can't.
    """

    def __init__(self, width: int = 192, threshold_px: float = 1.0):
        self.width = width
        self.threshold_px = threshold_px    # shift (px at `width`) that counts as "still moving"
        self.prev = None                    # previous small grey frame, unmasked
        self.prev_boxes: list = []
        self.shift_px = 0.0                 # last whole-scene shift, pixels at `width`
        self.response = 0.0                 # phase-correlation confidence

    @property
    def moving(self) -> bool:
        return self.shift_px > self.threshold_px

    def update(self, frame: np.ndarray, ignore_boxes=()) -> float:
        """`ignore_boxes`: people / vehicles, whose own movement must not count."""
        h, w = frame.shape[:2]
        small = downscale(frame, self.width)
        gray = np.float32(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)) if small.ndim == 3 else np.float32(small)
        sh = gray.shape[0]
        sx, sy = self.width / w, sh / h
        boxes = [(max(0, int(x1 * sx) - 1), max(0, int(y1 * sy) - 1), int(x2 * sx) + 2, int(y2 * sy) + 2)
                 for x1, y1, x2, y2 in ignore_boxes]
        prev, prev_boxes = self.prev, self.prev_boxes
        self.prev, self.prev_boxes = gray, boxes
        if prev is None or prev.shape != gray.shape:
            self.shift_px, self.response = 0.0, 0.0
            return 0.0
        # Blank the SAME areas (this frame's and the last frame's objects) in both
        # images; blanking only one frame makes the blank patch itself look like motion.
        a, b = prev.copy(), gray.copy()
        fill = float(gray.mean())
        for x1, y1, x2, y2 in boxes + prev_boxes:
            a[y1:y2, x1:x2] = fill
            b[y1:y2, x1:x2] = fill
        (dx, dy), response = cv2.phaseCorrelate(a, b)
        self.response = float(response)
        # Low response = no reliable texture (dark / flat picture): assume still.
        self.shift_px = float(np.hypot(dx, dy)) if response > 0.05 else 0.0
        return self.shift_px

    def reset(self) -> None:
        self.prev = None
        self.prev_boxes = []
        self.shift_px = 0.0


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
        self._moving_vehicle = False
        self._person_seen: deque = deque(maxlen=64)   # times of frames with a person in them
        self.motion = MotionMeter()
        self.patrol_idx = 0
        self.patrol_step = 1
        self.dwell_until = 0.0

    @property
    def ptz_enabled(self) -> bool:
        return self._ptz_hw and self.cfg.tracking_enabled

    def flush(self, now: float | None = None) -> None:
        """Pipeline stopping (settings change, shutdown, end of video): save what the
        current chase has collected instead of throwing it away."""
        if self.target is not None:
            self._finish(time.time() if now is None else now, "stopped")

    def reset(self) -> None:
        """Drop the current target without saving (e.g. manual control took over)."""
        self.target = None
        self.state = State.HOME
        self.tracker.reset()
        self.dwell_until = 0.0

    # ------------------------------------------------------------ patrol
    @property
    def patrolling(self) -> bool:
        return self.ptz_enabled and self.cfg.patrol.enabled

    def patrol_positions(self) -> list[float]:
        p = self.cfg.patrol
        lo, hi = sorted((p.left_pan, p.right_pan))
        n = max(1, int(p.stops))
        if n == 1 or hi - lo < 1e-3:
            return [(lo + hi) / 2]
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    def _goto_stop(self, now) -> None:
        stops = self.patrol_positions()
        self.patrol_idx = min(self.patrol_idx, len(stops) - 1)
        c = self.cfg.camera
        self.ptz.move(PTZState(stops[self.patrol_idx], c.home_tilt, c.home_zoom), force=True)
        self.ptz.moving_until += 0.3            # let the picture settle
        self.tracker.reset()
        self.dwell_until = self.ptz.moving_until + max(1.0, self.cfg.patrol.dwell_s)

    def _next_stop(self, now) -> None:
        n = len(self.patrol_positions())
        if n > 1:
            nxt = self.patrol_idx + self.patrol_step
            if not 0 <= nxt < n:                # bounce at the edges
                self.patrol_step = -self.patrol_step
                nxt = self.patrol_idx + self.patrol_step
            self.patrol_idx = nxt
        self._goto_stop(now)

    def _go_rest(self, now) -> None:
        """Where to go when not chasing anything: current patrol stop, or home."""
        if self.patrolling:
            self._goto_stop(now)
        else:
            self.ptz.home()

    # ------------------------------------------------------------ geometry
    def img_to_world(self, x, y, w, h):
        hf, vf = self.ptz.fov()
        st = self.ptz.est
        ax = math.degrees(math.atan((x - w / 2) / (w / 2) * math.tan(math.radians(hf / 2))))
        ay = math.degrees(math.atan((y - h / 2) / (h / 2) * math.tan(math.radians(vf / 2))))
        return st.pan + ax, st.tilt - ay

    def world_to_img(self, pan, tilt, w, h):
        hf, vf = self.ptz.fov()
        st = self.ptz.est
        dx = math.tan(math.radians(pan - st.pan)) / math.tan(math.radians(hf / 2))
        dy = math.tan(math.radians(st.tilt - tilt)) / math.tan(math.radians(vf / 2))
        return w / 2 + dx * w / 2, h / 2 + dy * h / 2

    # ------------------------------------------------------------ main step
    def step(self, frame: np.ndarray, dets: list[Detection], now: float | None = None) -> dict:
        now = time.time() if now is None else now
        self.ptz.update_estimate(now)
        if self.ptz_enabled:
            # If the picture still slides, the camera hasn't stopped, whatever the
            # timing model says: keep waiting. The camera only moves when we tell it
            # to, so sliding long after the model says it stopped is the scene
            # (trees, headlights), not the camera: then the picture is ignored.
            self.motion.update(frame, [d.box for d in dets])
            if self.motion.moving and now <= self.ptz.model_until + 1.5:
                self.ptz.moving_until = max(self.ptz.moving_until, now + 0.1)
        self._moving_vehicle = False
        if self.state == State.HOME:
            self._step_home(frame, dets, now)
        else:
            self._step_track(frame, dets, now)
        self.note_activity(dets, now)
        return self.status(now)

    def note_activity(self, dets: list[Detection], now: float) -> bool:
        """Decide whether this frame should (keep) trigger(ing) event recording.

        People and *moving* vehicles count, each only if enabled in the
        recording settings. Parked cars never do.
        """
        rc = self.cfg.recording
        t = self.target
        if any(d.kind == PERSON for d in dets):
            self._person_seen.append(now)
        person_confirmed = sum(1 for ts in self._person_seen if now - ts <= 1.0) >= rc.trigger_frames
        hit = ((rc.trigger_people and person_confirmed)
               or (rc.trigger_vehicles and self._moving_vehicle)
               or (self.state == State.TRACK and t is not None
                   and (rc.trigger_people if t.kind == PERSON else rc.trigger_vehicles)))
        if hit:
            self.active_until = now + 1.0
        return now < self.active_until

    def status(self, now: float) -> dict:
        t = self.target
        return {
            "state": (f"patrolling ({self.patrol_idx + 1}/{len(self.patrol_positions())})"
                      if self.state == State.HOME and self.patrolling else self.state.value),
            "target": t.label if t else None,
            "zoom": round(self.ptz.state.zoom, 2),
            "pan": round(self.ptz.state.pan, 1),
            "tilt": round(self.ptz.state.tilt, 1),
            "active": now < self.active_until,
            "last_event": self.last_event,
        }

    # ------------------------------------------------------------ HOME
    def _suppressed(self, kind, pan, tilt, now) -> bool:
        self.recent = [r for r in self.recent if r[6] > now]
        for k, p, t, vp, vt, t_end, _ in self.recent:
            if k != kind:
                continue
            dt = now - t_end
            # The speed estimate isn't perfect: allow more slack the longer ago it was.
            tol = 10 + 0.3 * math.hypot(vp, vt) * dt
            if abs(pan - (p + vp * dt)) < tol and abs(tilt - (t + vt * dt)) < tol:
                return True
        return False

    def _step_home(self, frame, dets, now):
        h, w = frame.shape[:2]
        if now < self.ptz.moving_until:
            return                      # blurred / shifting view: don't judge motion
        tracks = self.tracker.update(dets, now, w)
        dcfg = self.cfg.detection
        self._moving_vehicle = any(
            tr.det.kind == VEHICLE and tr.travel(now, dcfg.motion_window_s) >= dcfg.motion_min_travel * w
            for tr in tracks)
        if now < self.cooldown_until:
            return
        candidates = []
        for tr in tracks:
            d = tr.det
            pan, tilt = self.img_to_world(*d.center, w, h)
            if self._suppressed(d.kind, pan, tilt, now):
                continue
            tc = self.cfg.tracking
            # Time-based, so it means the same at 12 fps and at 30 fps: seen for at
            # least `min_age_s`, in a few frames, clearly enough.
            if (tr.age(now) < tc.min_age_s or tr.hits < tc.min_sightings
                    or tr.avg_conf < tc.min_avg_conf):
                continue                # not seen long / clearly enough yet: could be a ghost
            if d.kind == PERSON:
                candidates.append((1, d.height, tr))
            elif d.kind == VEHICLE and tr.travel(now, dcfg.motion_window_s) >= dcfg.motion_min_travel * w:
                candidates.append((2 if tc.prefer_vehicles else 0, d.width, tr))
        if not candidates:
            if self.patrolling and now >= self.dwell_until:
                self._next_stop(now)
            return
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        tr = candidates[0][2]
        det = tr.det
        ax, ay = self._aim_point(det, None, h)
        world = self.img_to_world(ax, ay, w, h)
        self.target = Target(det.kind, det, det.label, now, now, world, world_t=now,
                             seen_zoom=self.ptz.est.zoom)
        # Seed the velocity estimate with what we saw from home, so moving
        # targets are led from the first command.
        off_x, off_y = ax - det.center[0], ay - det.center[1]
        for ts, cx, cy in list(tr.history)[-self.target.hist.maxlen:]:
            self.target.hist.append((ts, *self.img_to_world(cx + off_x, cy + off_y, w, h)))
        self.state = State.TRACK
        self.storage.log(f"Tracking {det.label}")
        vp, vt = self.target.velocity()
        tlog.info("START %s at pan %.1f tilt %.1f, speed %.1f/%.1f deg/s, camera at pan %.1f zoom %.2f; "
                  "seen %d times over %.2fs, avg confidence %.2f",
                  det.label, world[0], world[1], vp, vt, self.ptz.est.pan, self.ptz.est.zoom,
                  tr.hits, tr.age(now), tr.avg_conf)

    # ------------------------------------------------------------ TRACK
    def _aim_point(self, det: Detection, feature_box, h):
        if feature_box is not None:
            x1, y1, x2, y2 = feature_box
            return (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1, x2, y2 = det.box
        frac = self.cfg.tracking.person_aim_y if det.kind == PERSON else self.cfg.tracking.vehicle_aim_y
        return (x1 + x2) / 2, y1 + (y2 - y1) * frac

    def _predict(self, now) -> tuple[float, float]:
        """Where the target should be now: last clean measurement + its velocity."""
        t = self.target
        vp, vt = t.velocity()
        dt = min(max(0.0, now - t.world_t), 2.0)
        return t.world[0] + vp * dt, t.world[1] + vt * dt

    def _match(self, dets, w, h, now) -> Detection | None:
        t = self.target
        px, py = self.world_to_img(*self._predict(now), w, h)
        best, best_d = None, 0.35 * w
        for d in dets:
            if d.kind != t.kind:
                continue
            ax, ay = self._aim_point(d, None, h)
            dist = math.hypot(ax - px, ay - py)
            if dist < best_d:
                best, best_d = d, dist
        if best is None:
            best = self._match_in_picture(dets, w, h)
            if best is not None:
                tlog.debug("MATCH by picture fallback (predicted x=%.2f y=%.2f was off)", px / w, py / h)
        return best

    def _match_in_picture(self, dets, w, h) -> Detection | None:
        """Fallback when the angle prediction is off (e.g. zoomed in, gimbal estimate
        a few degrees wrong): a similar-sized box near where the target was last
        seen, or near the centre, where the camera was just aimed at it."""
        t = self.target
        size_scale = self.ptz.est.zoom / max(t.seen_zoom, 1e-3)
        expected_h = max(1.0, t.det.height * size_scale)
        lx, ly = t.det.center
        best, best_score = None, 0.3
        for d in dets:
            if d.kind != t.kind or not 0.5 <= d.height / expected_h <= 2.0:
                continue
            cx, cy = d.center
            score = min(math.hypot(cx - lx, cy - ly), math.hypot(cx - w / 2, cy - h / 2)) / w
            if score < best_score:
                best, best_score = d, score
        return best

    def _step_track(self, frame, dets, now):
        t = self.target
        tc = self.cfg.tracking
        h, w = frame.shape[:2]

        det = self._match(dets, w, h, now)
        if det is None:
            tlog.debug("NO MATCH (%d detections), %.1fs since last seen, camera %s, scene shift %.2fpx (resp %.2f)",
                       len(dets), now - t.last_seen, "moving" if now < self.ptz.moving_until else "still",
                       self.motion.shift_px, self.motion.response)
        if det is not None:
            t.det, t.last_seen, t.seen_zoom = det, now, self.ptz.est.zoom
            self._analyse(frame, det, now)
            # Stop-and-measure: only trust where things are in the picture once the
            # camera has stopped. While it is still turning, the picture and our idea
            # of where the camera points disagree, and acting on that makes the
            # camera run away from fast targets.
            if not self.ptz_enabled or now >= self.ptz.moving_until:
                ax, ay = self._aim_point(det, t.feature_box, h)
                t.world, t.world_t = self.img_to_world(ax, ay, w, h), now
                t.hist.append((now, *t.world))
                tlog.debug("MEASURE target pan %.1f tilt %.1f (in picture x=%.2f y=%.2f), camera est pan %.1f "
                           "tilt %.1f zoom %.2f, scene shift %.2fpx, conf %.2f",
                           *t.world, ax / w, ay / h, self.ptz.est.pan, self.ptz.est.tilt, self.ptz.est.zoom,
                           self.motion.shift_px, det.conf)
                if self.ptz_enabled:
                    self._command(det, ax, ay, w, h, now)
                elif t.zoomed_at is None:
                    t.zoomed_at = now       # fixed camera: start collecting right away

        # Frames taken while the camera turns are smeared and usually detect nothing:
        # time spent waiting for the camera (up to 3 s) doesn't count as "lost".
        busy = 0.0
        if self.ptz_enabled:
            busy = min(max(0.0, self.ptz.moving_until - t.last_seen), 3.0)
        lost = now - t.last_seen > tc.lost_timeout_s + busy
        if self.ptz.est.zoom >= self.cfg.camera.max_zoom_ratio - 0.05:
            t.at_max_zoom_since = t.at_max_zoom_since or now
        else:
            t.at_max_zoom_since = None

        if tc.follow_until_gone:
            # Stay on it until it leaves the picture (or the safety limit), keeping
            # the best face / plate collected along the way.
            if lost:
                return self._finish(now, "left the view")
            if now - t.started > tc.follow_max_s:
                return self._finish(now, "follow time limit")
            return
        if t.accepted_plate:
            return self._finish(now, "plate read")
        if t.best_face and t.best_face[0] >= tc.face_fill * h * 0.8:
            return self._finish(now, "face captured")
        if t.best_face and t.at_max_zoom_since and now - t.at_max_zoom_since > 1.5:
            return self._finish(now, "face captured (max zoom)")   # can't get any bigger
        if lost:
            return self._finish(now, "lost")
        if now - t.started > tc.max_track_s:
            return self._finish(now, "timeout")
        if t.zoomed_at is not None and now - t.zoomed_at > tc.capture_window_s:
            return self._finish(now, "done")

    def _command(self, det, ax, ay, w, h, now):
        t, tc, ptz = self.target, self.cfg.tracking, self.ptz
        st, real = ptz.state, ptz.est      # last command / where the camera really is
        err_x, err_y = (ax - w / 2) / w, (ay - h / 2) / h
        goal_pan, goal_tilt = t.world      # the target's real direction (measured, camera still)
        vp, vt = t.velocity()
        cam = self.cfg.camera
        # Aim where the target will be when the camera has arrived and stopped.
        lead = tc.lead_s
        for _ in range(2):
            travel = max(abs(goal_pan + vp * lead - real.pan), abs(goal_tilt + vt * lead - real.tilt))
            lead = (tc.lead_s + cam.ptz_latency_s + cam.ptz_settle_margin_s
                    + travel / max(cam.pan_speed_dps, 1e-3))
        aim_pan, aim_tilt = goal_pan + vp * lead, goal_tilt + vt * lead
        moving_target = abs(vp) > 2 or abs(vt) > 2
        # Damp only small corrections on slow targets; big errors are corrected in one move.
        hf, _ = ptz.fov()
        big_error = max(abs(aim_pan - real.pan), abs(aim_tilt - real.tilt)) > 0.1 * hf
        gain = 1.0 if (moving_target or big_error) else tc.gain
        new_pan, new_tilt = st.pan, st.tilt
        if abs(err_x) > tc.deadband or moving_target:
            new_pan = real.pan + gain * (aim_pan - real.pan)
        if abs(err_y) > tc.deadband or moving_target:
            new_tilt = real.tilt + gain * (aim_tilt - real.tilt)

        # --- zoom: how big is the thing we want (face / plate) vs. how big we want it.
        # Sizes are measured at the REAL zoom, so scale from that, not the command.
        zoom = st.zoom
        centred = abs(err_x) < 0.15 and abs(err_y) < 0.15
        fill, goal = self._fill(det, w, h)
        if fill is not None and centred:
            ratio = goal / max(fill, 1e-3)
            # Well centred and slow: zoom up to 2x in one go (fewer wait-for-camera rounds).
            steady = abs(err_x) < 0.08 and abs(err_y) < 0.08 and math.hypot(vp, vt) <= 6
            max_step = 2.0 if steady else 1 + tc.zoom_step
            if ratio > 1.1:
                zoom = real.zoom * min(max_step, ratio)
            elif ratio < 0.8:
                zoom = real.zoom * max(1 - tc.zoom_step, ratio)
        # Never zoom so far that the whole vehicle can't be kept in frame.
        if det.kind == VEHICLE and det.width / w > tc.vehicle_fill:
            zoom = min(zoom, real.zoom * tc.vehicle_fill / (det.width / w))
        if moving_target:
            speed = math.hypot(vp, vt)
            if speed > 6:
                # 1) Fast targets (cars): zoom must not outlast the turn. The camera is
                #    aimed where the target will be when it ARRIVES; if zooming carries on
                #    after that, the car drives out of the narrowing picture while the
                #    camera sits blind.
                turn_time = cam.ptz_latency_s + travel / max(cam.pan_speed_dps, 1e-3)
                dz = cam.zoom_speed * max(turn_time, 0.3)
                zoom = min(max(zoom, real.zoom - dz), real.zoom + dz)
            # 2) Keep the view wide enough for the time between arriving and the next
            #    measurement, plus a margin for error in the speed estimate.
            drift = speed * (cam.ptz_settle_margin_s + 0.2 + 0.2 * lead)
            need = min(4.0 * drift, cam.hfov_deg)       # target within half of half the view
            if need > 0:
                zmax = math.tan(math.radians(cam.hfov_deg / 2)) / math.tan(math.radians(need / 2))
                zoom = min(zoom, max(1.0, zmax))
        zoom = min(zoom, self.cfg.camera.max_zoom_ratio)
        tlog.debug("COMMAND speed %.1f/%.1f deg/s, lead %.2fs -> pan %.1f tilt %.1f zoom %.2f",
                   vp, vt, lead, new_pan, new_tilt, zoom)
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
                t.best_frame = frame        # frames are never modified in place: no copy needed

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
        tlog.info("END %s after %.1fs: %s (plate votes %s)", t.label, now - t.started, reason, dict(t.plate_votes))
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
        # Same hold whether or not a capture succeeded: a short hold after a failed
        # chase made the camera swing back to the same person/car again and again.
        hold = self.cfg.tracking.recapture_after_s
        # Remember where it is NOW (the last clean measurement may be ~1 s old).
        self.recent.append((t.kind, *self._predict(now), vp, vt, now, now + hold))
        self.target = None
        self.state = State.HOME
        self.tracker.reset()
        self.cooldown_until = now + self.cfg.tracking.cooldown_s
        if self.ptz_enabled:
            self._go_rest(now)


# ------------------------------------------------------------ drawing

def annotate(img: np.ndarray, dets: list[Detection], ctl: Controller, fps: float,
             scale: float = 1.0) -> np.ndarray:
    """Draw detections and status on a copy of `img`. `scale` maps full-frame box
    coordinates onto `img` (pass a downscaled frame: drawing on 4K is costly)."""
    out = img.copy()
    h, w = out.shape[:2]
    s = max(0.6, w / 1280)

    def px(box):
        return tuple(int(v * scale) for v in box)

    for d in dets:
        color = (0, 200, 255) if d.kind == PERSON else (255, 160, 0)
        x1, y1, x2, y2 = px(d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(1, int(2 * s)))
        cv2.putText(out, f"{d.label} {d.conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, color, max(1, int(1 * s)))
    t = ctl.target
    if t is not None:
        x1, y1, x2, y2 = px(t.det.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), max(1, int(3 * s)))
        if t.feature_box is not None:
            fx1, fy1, fx2, fy2 = px(t.feature_box)
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (0, 255, 0), max(1, int(2 * s)))
    cv2.drawMarker(out, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, int(20 * s), 1)
    st = ctl.ptz.state
    label = "PATROLLING" if ctl.state == State.HOME and ctl.patrolling else ctl.state.value.upper()
    txt = f"{label}  pan {st.pan:+.1f}  tilt {st.tilt:+.1f}  zoom {st.zoom:.1f}x  {fps:.0f} fps"
    if t is not None and t.plate_votes:
        txt += f"  plate? {t.plate_votes.most_common(1)[0][0]}"
    draw_label(out, txt, (int(12 * s), int(30 * s)), 0.6 * s)
    return out

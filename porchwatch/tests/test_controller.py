"""Closed-loop simulation: virtual people/cars move through the world, the fake
camera's pan/tilt/zoom decides where they appear, and the controller has to
follow, zoom in, capture and return home."""
import json

import cv2
import numpy as np
import pytest

from porchwatch.camera import NullPTZ
from porchwatch.config import Config
from porchwatch.controller import Controller, State
from porchwatch.detectors import PERSON, VEHICLE, Detection, PlateRead
from porchwatch.storage import Storage

W, H = 640, 360
FPS = 20
RED = (0, 0, 255)       # face colour
WHITE = (255, 255, 255)  # plate colour


def texture(img, x1, y1, x2, y2, color):
    """Fill with a checkerboard so the sharpness filter sees detail."""
    x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    img[y1:y2, x1:x2] = color
    yy, xx = np.mgrid[y1:y2, x1:x2]
    img[y1:y2, x1:x2][((yy // 3 + xx // 3) % 2) == 0] = (np.array(color) * 0.6).astype(np.uint8)


def find_color(crop, color, tol=10):
    lo = np.clip(np.array(color) * 0.6 - tol, 0, 255).astype(np.uint8)
    hi = np.clip(np.array(color) + tol, 0, 255).astype(np.uint8)
    mask = cv2.inRange(crop, lo, hi)
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


class FakeFaces:
    def __call__(self, crop):
        box = find_color(crop, RED)
        return [(box, 0.9)] if box else []


class FakePlates:
    available = True

    def __call__(self, crop):
        box = find_color(crop, WHITE)
        if box is None or box[2] - box[0] < 12:
            return []          # too small to read, like real OCR
        # body colour tells us which car this is
        text = "MOVING1" if crop[..., 0].mean() > crop[..., 1].mean() else "PARKED1"
        return [PlateRead(text, 0.93, box)]


class Sim:
    def __init__(self, tmp_path, **tracking):
        self.t = 1000.0
        cfg = Config()
        cfg.camera.command_interval_s = 0.0
        cfg.capture.output_dir = str(tmp_path / "captures")
        cfg.capture.min_face_px = 20
        cfg.capture.min_sharpness = 10
        for k, v in tracking.items():
            setattr(cfg.tracking, k, v)
        self.cfg = cfg
        self.ptz = NullPTZ(cfg.camera, clock=lambda: self.t)
        self.storage = Storage(cfg.capture)
        self.ctl = Controller(cfg, self.ptz, FakeFaces(), FakePlates(), self.storage, ptz_enabled=True)
        self.objects = []

    def project(self, pan1, tilt_top, pan2, tilt_bottom):
        x1, y1 = self.ctl.world_to_img(pan1, tilt_top, W, H)
        x2, y2 = self.ctl.world_to_img(pan2, tilt_bottom, W, H)
        return x1, y1, x2, y2

    def render(self):
        img = np.full((H, W, 3), 40, np.uint8)
        dets = []
        for obj in self.objects:
            pan = obj["pan0"] + obj["speed"] * (self.t - obj["t0"])
            half = obj["width"] / 2
            box = self.project(pan - half, obj["top"], pan + half, obj["bottom"])
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            texture(img, *box, obj["color"])
            bw, bh = box[2] - box[0], box[3] - box[1]
            if obj["kind"] == PERSON:   # face: top 12 % of body, 40 % of width
                texture(img, cx - bw * 0.2, box[1], cx + bw * 0.2, box[1] + bh * 0.12, RED)
            else:                       # plate: lower middle
                texture(img, cx - bw * 0.11, box[1] + bh * 0.65, cx + bw * 0.11, box[1] + bh * 0.8, WHITE)
            clipped = (max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3]))
            dets.append(Detection(obj["kind"], clipped, 0.9, obj["label"]))
        return img, dets

    def run(self, seconds, on_frame=None):
        for _ in range(int(seconds * FPS)):
            img, dets = self.render()
            self.ctl.step(img, dets, self.t)
            if on_frame:
                on_frame(self)
            self.t += 1.0 / FPS

    def events(self):
        f = self.storage.events_file
        return [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []


def person(sim, pan0, speed, dist_deg=7.0):
    return {"kind": PERSON, "label": "person", "pan0": pan0, "speed": speed, "t0": sim.t,
            "width": dist_deg * 0.3, "top": 2.0, "bottom": 2.0 - dist_deg, "color": (60, 170, 60)}


def car(sim, pan0, speed, label, color):
    return {"kind": VEHICLE, "label": label, "pan0": pan0, "speed": speed, "t0": sim.t,
            "width": 12.0, "top": -2.0, "bottom": -7.0, "color": color}


def test_follows_person_zooms_and_saves_face(tmp_path):
    sim = Sim(tmp_path)
    sim.objects.append(person(sim, pan0=-20, speed=1.5))
    max_zoom, min_err = [1.0], [1.0]

    def watch(s):
        if s.ctl.state == State.TRACK:
            max_zoom[0] = max(max_zoom[0], s.ptz.state.zoom)
            fb = s.ctl.target.feature_box
            if fb is not None:
                err = abs((fb[0] + fb[2]) / 2 - W / 2) / W
                min_err[0] = min(min_err[0], err)

    sim.run(12, watch)
    events = sim.events()
    assert events, "no capture saved"
    ev = events[0]
    assert ev["kind"] == "person"
    assert "face" in ev["files"]
    assert (tmp_path / "captures" / ev["files"]["face"]).exists()
    assert max_zoom[0] > 2.0, "never zoomed in"
    assert min_err[0] < 0.05, "never centred the face"
    # after capturing it returned home and does not immediately re-chase the same person
    assert len([e for e in events if e["kind"] == "person"]) == 1
    assert sim.ptz.state.pan == pytest.approx(sim.cfg.camera.home_pan)


def test_ignores_parked_car_follows_moving_car_and_reads_plate(tmp_path):
    sim = Sim(tmp_path)
    sim.objects.append(car(sim, pan0=15, speed=0.0, label="car", color=(40, 160, 40)))    # parked (green)
    sim.objects.append(car(sim, pan0=-30, speed=8.0, label="car", color=(200, 90, 40)))   # moving (blue)
    tracked_parked = []

    def watch(s):
        t = s.ctl.target
        if t is not None and t.plate_votes and "PARKED1" in t.plate_votes:
            tracked_parked.append(s.t)

    sim.run(8, watch)
    events = sim.events()
    assert events, "no capture saved"
    plates = [e.get("plate") for e in events]
    assert plates.count("MOVING1") == 1, f"captured the same car {plates.count('MOVING1')} times"
    assert "PARKED1" not in plates and not tracked_parked
    ev = next(e for e in events if e.get("plate") == "MOVING1")
    assert ev["plate_confirmed"] and "plate" in ev["files"]


def test_gives_up_when_target_disappears(tmp_path):
    sim = Sim(tmp_path)
    p = person(sim, pan0=0, speed=0.0)
    sim.objects.append(p)
    sim.run(0.6)
    assert sim.ctl.state == State.TRACK
    sim.objects.clear()
    sim.run(sim.cfg.tracking.lost_timeout_s + 0.3)
    assert sim.ctl.state == State.HOME


def test_tracking_disabled_never_moves_camera(tmp_path):
    sim = Sim(tmp_path)
    sim.cfg.tracking_enabled = False
    sim.objects.append(person(sim, pan0=-5, speed=0.5, dist_deg=25))
    sim.run(6)
    assert sim.ptz.state.zoom == 1.0 and sim.ptz.state.pan == 0.0
    assert sim.events(), "fixed-camera mode should still save captures"


class StrictPlates(FakePlates):
    """Only readable once the camera has zoomed in on the plate."""

    def __call__(self, crop):
        box = find_color(crop, WHITE)
        if box is None or box[2] - box[0] < 45:
            return []
        return super().__call__(crop)


@pytest.mark.parametrize("speed", [4, 8, 15])
def test_zooms_to_read_plate_of_passing_car(tmp_path, speed):
    sim = Sim(tmp_path)
    sim.ctl.plates = StrictPlates()
    sim.objects.append(car(sim, pan0=-35, speed=speed, label="car", color=(200, 90, 40)))
    zooms = []
    sim.run(10, lambda s: zooms.append(s.ptz.state.zoom))
    assert max(zooms) > 2.0
    assert [e.get("plate") for e in sim.events()] == ["MOVING1"]


def _enable_patrol(sim, left=-60, right=60, stops=3, dwell=2.0):
    p = sim.cfg.patrol
    p.enabled, p.left_pan, p.right_pan, p.stops, p.dwell_s = True, left, right, stops, dwell


def test_patrol_sweeps_back_and_forth_when_nothing_happens(tmp_path):
    sim = Sim(tmp_path)
    _enable_patrol(sim)
    visited = []

    def watch(s):
        pan = round(s.ptz.state.pan)
        if not visited or visited[-1] != pan:
            visited.append(pan)

    sim.run(20, watch)
    assert set(visited) == {-60, 0, 60}, visited
    # ping-pong: every step moves to a neighbouring stop, direction flips only at the edges
    for a, b, c in zip(visited, visited[1:], visited[2:]):
        assert abs(b - a) == 60 and (b == c - (b - a) or b in (-60, 60)), visited
    assert not sim.events()


def test_patrol_finds_person_outside_home_view_then_resumes(tmp_path):
    sim = Sim(tmp_path)
    _enable_patrol(sim)
    # standing far right: invisible from home (pan 0, ~76 deg view) but seen at the +60 stop
    sim.objects.append(person(sim, pan0=62, speed=0.0))
    tracked = []
    sim.run(25, lambda s: tracked.append(s.ctl.state == State.TRACK))
    events = sim.events()
    assert len(events) == 1 and events[0]["kind"] == "person" and "face" in events[0]["files"]
    # after the capture it kept patrolling (the camera still moves between stops)
    last_track = max(i for i, t in enumerate(tracked) if t)
    pans_after = {round(sim.ptz.state.pan)}
    sim.objects.clear()
    sim.run(10, lambda s: pans_after.add(round(s.ptz.state.pan)))
    assert len(pans_after) >= 2 and last_track < len(tracked) - 1


def test_patrol_positions():
    cfg = Config()
    cfg.patrol.left_pan, cfg.patrol.right_pan, cfg.patrol.stops = 40, -40, 5
    ctl = Controller(cfg, NullPTZ(cfg.camera), None, None, None)
    assert ctl.patrol_positions() == [-40, -20, 0, 20, 40]
    cfg.patrol.stops = 1
    assert ctl.patrol_positions() == [0]

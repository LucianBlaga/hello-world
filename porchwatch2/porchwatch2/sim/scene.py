"""A simple street in front of the house, seen by both lenses of the virtual camera.

World coordinates in metres: camera at (0, 0) at `cam_height` above ground,
x to the right along the street, y away from the house, z up. Pan 0 deg looks
straight at the street (+y), positive pan turns right (+x).

Objects are boxes; cars have a plate on the front and back, readable only when
the car faces the camera enough (plates face along the street). Rendering is
flat-shaded on purpose: the point is geometry, timing and blur, not realism.
It also returns ground-truth boxes, which stand in for a detector in tests.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Thing:
    kind: str                         # "car" | "person"
    x: float
    y: float
    vx: float                         # m/s along the street
    length: float
    width: float
    height: float
    color: tuple[int, int, int]
    plate: str = ""
    id: int = 0

    def corners(self) -> np.ndarray:
        """8 box corners (x, y, z)."""
        hl, hw = self.length / 2, self.width / 2
        xs, ys, zs = (self.x - hl, self.x + hl), (self.y - hw, self.y + hw), (0.0, self.height)
        return np.array([(a, b, c) for a in xs for b in ys for c in zs], dtype=np.float64)


@dataclass
class View:
    pan: float
    tilt: float
    hfov: float
    width: int
    height: int


@dataclass
class Scene:
    cam_height: float = 3.0
    street_y: float = 14.0            # street centre line
    lane_offset: float = 1.8
    sidewalk_y: float = 8.5
    things: list[Thing] = field(default_factory=list)
    t: float = 0.0
    seed: int = 1
    _next_id: int = 1

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def add_car(self, direction: int = 1, speed: float = 11.0, x: float | None = None, plate: str | None = None) -> Thing:
        letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
        plate = plate or f"{self.rng.randint(1, 9)}{''.join(self.rng.choice(letters) for _ in range(3))}{self.rng.randint(100, 999)}"
        y = self.street_y + (-self.lane_offset if direction > 0 else self.lane_offset)
        car = Thing("car", x if x is not None else -45.0 * direction, y, speed * direction, 4.6, 1.8, 1.45,
                    tuple(self.rng.randint(40, 220) for _ in range(3)), plate, self._next_id)
        self._next_id += 1
        self.things.append(car)
        return car

    def add_person(self, direction: int = 1, speed: float = 1.4, x: float | None = None) -> Thing:
        p = Thing("person", x if x is not None else -25.0 * direction, self.sidewalk_y, speed * direction,
                  0.5, 0.4, 1.75, tuple(self.rng.randint(30, 200) for _ in range(3)), "", self._next_id)
        self._next_id += 1
        self.things.append(p)
        return p

    def step(self, dt: float) -> None:
        self.t += dt
        for th in self.things:
            th.x += th.vx * dt
        self.things = [th for th in self.things if abs(th.x) < 60]

    # ------------------------------------------------------------------ geometry
    def project(self, pts: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray]:
        """World points -> pixel coords and depth (depth <= 0 means behind)."""
        f = (view.width / 2) / math.tan(math.radians(view.hfov) / 2)
        p, t = math.radians(view.pan), math.radians(view.tilt)
        rel = pts - np.array([0.0, 0.0, self.cam_height])
        # rotate world into camera frame: undo pan (about z), then tilt.
        # PTZ convention: tilt 0 = horizon, positive = looking down (90 = straight down).
        x1 = rel[:, 0] * math.cos(p) - rel[:, 1] * math.sin(p)
        y1 = rel[:, 0] * math.sin(p) + rel[:, 1] * math.cos(p)
        z1 = rel[:, 2]
        fwd = y1 * math.cos(t) - z1 * math.sin(t)
        up = y1 * math.sin(t) + z1 * math.cos(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = view.width / 2 + f * x1 / fwd
            v = view.height / 2 - f * up / fwd
        return np.stack([u, v], axis=1), fwd

    def boxes(self, view: View) -> list[dict]:
        """Ground-truth boxes of visible things: {id, kind, box, plate_box, plate}."""
        out = []
        for th in self.things:
            uv, depth = self.project(th.corners(), view)
            if (depth <= 0.5).any():
                continue
            x0, y0 = uv.min(axis=0)
            x1, y1 = uv.max(axis=0)
            if x1 < 0 or y1 < 0 or x0 >= view.width or y0 >= view.height:
                continue
            item = {"id": th.id, "kind": th.kind,
                    "box": (max(0.0, x0), max(0.0, y0), min(view.width - 1.0, x1), min(view.height - 1.0, y1)),
                    "plate": th.plate, "plate_box": None, "distance": float(depth.mean())}
            if th.kind == "car":
                item["plate_box"] = self._plate_box(th, view)
            out.append(item)
        return out

    def _plate_box(self, car: Thing, view: View):
        """Visible plate (front or back, whichever faces the camera), else None."""
        front = 1 if car.vx >= 0 else -1
        for side in (front, -front):
            px = car.x + side * car.length / 2
            to_cam = np.array([-px, -car.y])
            normal = np.array([side, 0.0])
            cos_a = float(normal @ to_cam / (np.linalg.norm(to_cam) + 1e-9))
            if cos_a < 0.35:                  # seen too edge-on to read
                continue
            pts = np.array([(px, car.y - 0.153, 0.45), (px, car.y + 0.153, 0.45),
                            (px, car.y - 0.153, 0.60), (px, car.y + 0.153, 0.60)])
            uv, depth = self.project(pts, view)
            if (depth <= 0.5).any():
                continue
            x0, y0 = uv.min(axis=0)
            x1, y1 = uv.max(axis=0)
            if x0 >= 0 and y0 >= 0 and x1 < view.width and y1 < view.height:
                return (float(x0), float(y0), float(x1), float(y1))
        return None

    # ------------------------------------------------------------------ drawing
    def render(self, view: View, blur_px: float = 0.0, defocus: float = 0.0, noise: float = 2.0) -> np.ndarray:
        img = np.empty((view.height, view.width, 3), np.uint8)
        img[:] = (170, 190, 200)                               # sky
        self._ground(img, view)
        items = sorted(self.things, key=lambda th: -th.y)      # far first
        for th in items:
            self._draw_thing(img, th, view)
        if blur_px >= 1.5:
            k = int(min(blur_px, 61)) | 1
            kernel = np.zeros((1, k), np.float32)
            kernel[0, :] = 1.0 / k
            img = cv2.filter2D(img, -1, kernel)
        if defocus > 0.05:
            s = 1 + 2 * int(defocus * 6)
            img = cv2.GaussianBlur(img, (s, s), 0)
        if noise > 0:
            grain = np.random.default_rng().normal(0, noise, img.shape)
            img = np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)
        return img

    def _poly(self, img, view, pts, color):
        uv, depth = self.project(np.asarray(pts, dtype=np.float64), view)
        if (depth <= 0.2).any():
            return
        cv2.fillPoly(img, [np.round(uv).astype(np.int32)], color, lineType=cv2.LINE_AA)

    def _ground(self, img, view):
        far = 120.0
        self._poly(img, view, [(-far, 1, 0), (far, 1, 0), (far, far, 0), (-far, far, 0)], (95, 140, 110))
        self._poly(img, view, [(-far, self.sidewalk_y - 1, 0), (far, self.sidewalk_y - 1, 0),
                               (far, self.sidewalk_y + 1, 0), (-far, self.sidewalk_y + 1, 0)], (180, 180, 175))
        self._poly(img, view, [(-far, self.street_y - 4, 0), (far, self.street_y - 4, 0),
                               (far, self.street_y + 4, 0), (-far, self.street_y + 4, 0)], (70, 70, 75))

    def _draw_thing(self, img, th: Thing, view: View):
        hl, hw, h = th.length / 2, th.width / 2, th.height
        x, y = th.x, th.y
        dark = tuple(int(c * 0.7) for c in th.color)
        self._poly(img, view, [(x - hl, y - hw, 0), (x + hl, y - hw, 0), (x + hl, y - hw, h), (x - hl, y - hw, h)], th.color)
        side = 1 if x < 0 else -1                                # end facing the camera
        ex = x + side * hl
        self._poly(img, view, [(ex, y - hw, 0), (ex, y + hw, 0), (ex, y + hw, h), (ex, y - hw, h)], dark)
        if th.kind == "car":
            pb = self._plate_box(th, view)
            if pb:
                x0, y0, x1, y1 = (int(round(v)) for v in pb)
                cv2.rectangle(img, (x0, y0), (x1, y1), (235, 235, 235), -1)
                w = max(1, x1 - x0)
                scale = w / 160.0
                if scale > 0.15:
                    cv2.putText(img, th.plate, (x0 + int(w * 0.06), (y0 + y1) // 2 + int((y1 - y0) * 0.2)),
                                cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), max(1, int(scale * 2)), cv2.LINE_AA)

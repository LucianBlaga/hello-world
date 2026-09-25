"""Tiny centroid tracker used while the camera is parked at home.

Gives detections a stable id across frames and lets us tell moving vehicles
from parked ones. Only fed frames taken while the gimbal is still, so camera
motion never looks like object motion.
"""
from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .detectors import Detection


@dataclass
class Track:
    id: int
    det: Detection
    history: deque = field(default_factory=lambda: deque(maxlen=60))   # (t, cx, cy)
    last_seen: float = 0.0
    first_seen: float = 0.0

    def travel(self, now: float, window: float) -> float:
        """Distance (pixels) the centre moved over the last `window` seconds."""
        pts = [(t, x, y) for t, x, y in self.history if now - t <= window]
        if len(pts) < 4:
            return 0.0                  # too few sightings to tell (noise would look like motion)
        # Fit a straight line through the centres: box jitter averages out,
        # steady movement doesn't.
        ts = np.array([p[0] for p in pts]) - pts[0][0]
        if ts[-1] <= 0:
            return 0.0
        vx = np.polyfit(ts, [p[1] for p in pts], 1)[0]
        vy = np.polyfit(ts, [p[2] for p in pts], 1)[0]
        return float(np.hypot(vx, vy) * ts[-1])

    def age(self, now: float) -> float:
        return now - self.first_seen


class CentroidTracker:
    def __init__(self, max_missing_s: float = 0.8, max_jump: float = 0.2):
        self.max_missing_s = max_missing_s
        self.max_jump = max_jump            # fraction of frame width
        self.tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, dets: list[Detection], now: float, frame_w: int) -> list[Track]:
        gate = self.max_jump * frame_w
        pairs = []
        for tid, tr in self.tracks.items():
            for i, d in enumerate(dets):
                if d.kind != tr.det.kind:
                    continue
                dist = ((d.center[0] - tr.det.center[0]) ** 2 +
                        (d.center[1] - tr.det.center[1]) ** 2) ** 0.5
                if dist <= gate:
                    pairs.append((dist, tid, i))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, tid, i in pairs:
            if tid in used_t or i in used_d:
                continue
            used_t.add(tid)
            used_d.add(i)
            tr = self.tracks[tid]
            tr.det, tr.last_seen = dets[i], now
            tr.history.append((now, *dets[i].center))
        for i, d in enumerate(dets):
            if i not in used_d:
                tid = next(self._ids)
                tr = Track(tid, d, last_seen=now, first_seen=now)
                tr.history.append((now, *d.center))
                self.tracks[tid] = tr
        for tid in [t for t, tr in self.tracks.items() if now - tr.last_seen > self.max_missing_s]:
            del self.tracks[tid]
        return [tr for tr in self.tracks.values() if tr.last_seen == now]

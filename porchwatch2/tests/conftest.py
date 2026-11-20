import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from porchwatch2.sim.camera import VirtualCamera  # noqa: E402
from porchwatch2.sim.ptz import PTZLimits  # noqa: E402

FAST = PTZLimits(pan_speed=300.0, tilt_speed=300.0, zoom_full_s=0.4, latency_s=0.02, af_base_s=0.05, af_per_zoom_s=0.0)


@pytest.fixture
def make_vcam():
    cams = []

    def make(**kw):
        kw.setdefault("realtime", False)
        kw.setdefault("limits", FAST)
        cam = VirtualCamera(**kw).start()
        cams.append(cam)
        return cam
    yield make
    for c in cams:
        c.stop()


@pytest.fixture
def vcam(make_vcam):
    return make_vcam()


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def clock():
    return FakeClock()

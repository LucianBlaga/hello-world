import math

import pytest

from porchwatch2.sim.ptz import (PTZLimits, SimPTZ, norm_to_pan, norm_to_tilt, norm_to_zoom, pan_to_norm,
                                 tilt_to_norm, zoom_to_norm)
from porchwatch2.sim.scene import Scene, View


def test_absolute_move_takes_latency_plus_travel(clock):
    ptz = SimPTZ(clock=clock, pan=0.0, tilt=0.0, zoom=1.0)
    ptz.absolute(60.0, 0.0, 1.0)
    clock.advance(0.1)                       # still in command latency (0.15 s)
    assert ptz.state()["pan"] == 0.0 and ptz.state()["moving"]
    clock.advance(1.0)                       # 0.95 s of travel at 30 deg/s
    assert ptz.state()["pan"] == pytest.approx(28.5, abs=0.01)
    clock.advance(1.2)
    st = ptz.state()
    assert st["pan"] == pytest.approx(60.0) and not st["moving"]


def test_pan_takes_the_short_way_round(clock):
    ptz = SimPTZ(clock=clock, pan=350.0)
    ptz.absolute(10.0, None, None)
    clock.advance(0.15 + 0.5)                # 20 deg at 30 deg/s needs 0.67 s
    assert ptz.state()["pan"] == pytest.approx(5.0, abs=0.01)


def test_joystick_autostop_and_tilt_limits(clock):
    ptz = SimPTZ(clock=clock, tilt=0.0)
    ptz.joystick(0, -1, 1.0, 2.0)            # up for 2 s: would reach -60, limit is -15
    clock.advance(3.0)
    st = ptz.state()
    assert st["tilt"] == -15.0 and not st["moving"]


def test_zoom_full_range_time(clock):
    ptz = SimPTZ(clock=clock)
    ptz.absolute(None, None, 20.0)
    clock.advance(0.15 + 2.0)
    assert ptz.state()["zoom"] == pytest.approx(math.sqrt(20.0), rel=1e-3)   # halfway in log space
    clock.advance(2.1)
    assert ptz.state()["zoom"] == 20.0


def test_focus_recovers_after_move(clock):
    ptz = SimPTZ(clock=clock, limits=PTZLimits(af_base_s=0.3, af_per_zoom_s=0.0))
    ptz.absolute(30.0, None, None)
    clock.advance(0.5)
    assert ptz.sharpness() < 0.5
    clock.advance(1.0 + 0.3)
    assert ptz.sharpness() == 1.0


def test_presets_round_trip(clock):
    ptz = SimPTZ(clock=clock, pan=40.0, tilt=10.0, zoom=5.0)
    ptz.save_preset(3)
    ptz.absolute(0.0, 0.0, 1.0)
    clock.advance(5)
    assert ptz.goto_preset(3)
    clock.advance(5)
    st = ptz.state()
    assert (st["pan"], st["tilt"], st["zoom"]) == pytest.approx((40.0, 10.0, 5.0))
    assert not ptz.goto_preset(9)


@pytest.mark.parametrize("deg", [0.0, 90.0, 180.0, 359.0])
def test_onvif_mapping_round_trips(deg):
    assert norm_to_pan(pan_to_norm(deg)) == pytest.approx(deg)
    assert norm_to_tilt(tilt_to_norm(30.0)) == pytest.approx(30.0)
    assert norm_to_zoom(zoom_to_norm(7.0)) == pytest.approx(7.0)


def test_projection_puts_straight_ahead_in_the_middle():
    s = Scene()
    v = View(0.0, 0.0, 70.0, 1000, 500)
    uv, depth = s.project([[0.0, 20.0, s.cam_height]], v)
    assert uv[0] == pytest.approx([500.0, 250.0]) and depth[0] == pytest.approx(20.0)
    uv, _ = s.project([[5.0, 20.0, s.cam_height]], View(math.degrees(math.atan2(5, 20)), 0.0, 70.0, 1000, 500))
    assert uv[0][0] == pytest.approx(500.0)          # panning right brings it to the centre


def test_plate_visible_only_when_the_car_faces_the_camera():
    s = Scene()
    car = s.add_car(1, x=-20.0)
    ang = math.degrees(math.atan2(car.x + 2.3, car.y))
    tilt = math.degrees(math.atan2(s.cam_height - 0.5, math.hypot(car.x + 2.3, car.y)))
    view = View(ang, tilt, 8.0, 1280, 720)
    assert s.boxes(view)[0]["plate_box"] is not None          # approaching at an angle
    car.x = 0.0                                               # broadside, right in front
    assert s.boxes(View(0.0, 10.0, 30.0, 1280, 720))[0]["plate_box"] is None


def test_render_shapes():
    s = Scene()
    s.add_person(1, x=0.0)
    img = s.render(View(0.0, 8.0, 71.0, 640, 360), blur_px=9, defocus=0.5)
    assert img.shape == (360, 640, 3)

import base64
import datetime as dt
import hashlib
import xml.etree.ElementTree as ET

import pytest

from porchwatch2.camera.onvif import OnvifClient, OnvifError, check_security_header, make_security_header


def ov(cam, password="123456"):
    c = OnvifClient(cam.host, "admin", password, port=cam.port, timeout=3)
    c.connect()
    return c


def test_digest_matches_the_ws_security_formula():
    nonce, created = b"0123456789abcdef", "2026-11-20T10:00:00Z"
    xml = make_security_header("admin", "pw", created, nonce)
    expect = base64.b64encode(hashlib.sha1(nonce + created.encode() + b"pw").digest()).decode()
    assert expect in xml


def test_server_side_check_rejects_old_or_wrong_tokens():
    nonce, created = b"n" * 16, "2026-11-20T10:00:00Z"
    env = f'<e xmlns:s="http://www.w3.org/2003/05/soap-envelope">{make_security_header("admin", "pw", created, nonce)}</e>'
    root = ET.fromstring(env)
    now = dt.datetime(2026, 11, 20, 10, 1, tzinfo=dt.timezone.utc)
    assert check_security_header(root, {"admin": "pw"}, now)
    assert not check_security_header(root, {"admin": "other"}, now)
    assert not check_security_header(root, {"admin": "pw"}, now + dt.timedelta(hours=1))


def test_discovery_and_profiles(vcam):
    c = ov(vcam)
    assert {"media", "ptz", "imaging"} <= set(c.xaddr)
    profs = c.profiles()
    assert [p.token for p in profs] == ["wide_main", "wide_sub", "ptz_main", "ptz_sub"]
    assert c.ptz_profile().token == "ptz_main"
    assert c.stream_uri("ptz_main").endswith("/stream2")
    sp = c.ptz_spaces("ptzcfg")
    assert sp.pan == (-1.0, 1.0) and sp.zoom == (0.0, 1.0) and sp.raw["absolute_pan_tilt"]


def test_wrong_password_faults(vcam):
    c = OnvifClient(vcam.host, "admin", "bad", port=vcam.port)
    with pytest.raises(OnvifError, match="NotAuthorized"):
        c.connect()


def test_absolute_move_and_status(vcam):
    c = ov(vcam)
    c.absolute_move("ptz_main", 0.5, 0.0, 0.25)
    st = c.wait_idle("ptz_main", timeout_s=3)
    assert st is not None and not st.pan_tilt_moving
    assert (st.pan, st.tilt, st.zoom) == pytest.approx((0.5, 0.0, 0.25), abs=1e-3)
    s = vcam.ptz.state()
    assert s["pan"] == pytest.approx(270.0) and s["zoom"] == pytest.approx(5.75)


def test_camera_without_position_readback(make_vcam):
    cam = make_vcam(report_position=False)
    c = ov(cam)
    c.absolute_move("ptz_main", 0.5, 0.0, 0.25)
    st = c.wait_idle("ptz_main", timeout_s=3)
    assert (st.pan, st.zoom) == (0.0, 0.0)               # moved, but status says nothing useful
    assert cam.ptz.state()["pan"] == pytest.approx(270.0)


def test_fast_shutter_for_plates(vcam):
    c = ov(vcam)
    c.set_exposure("vs_ptz", manual=True, exposure_us=1000)
    assert vcam.exposure == {"mode": "MANUAL", "us": 1000.0}
    with pytest.raises(OnvifError):
        c.set_exposure("vs_ptz", manual=True, exposure_us=50)     # faster than 1/10000 s
    c.set_exposure("vs_ptz", manual=False)
    assert vcam.exposure["mode"] == "AUTO"


def test_continuous_move_and_stop(vcam):
    import time
    c = ov(vcam)
    c.continuous_move("ptz_main", 0.5, 0.0)
    time.sleep(0.2)
    c.stop("ptz_main")
    time.sleep(0.1)
    st = vcam.ptz.state()
    assert not st["moving"] and 10 < st["pan"] < 60

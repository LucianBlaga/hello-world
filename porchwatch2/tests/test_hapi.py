import hashlib
import time

import cv2
import numpy as np
import pytest

from porchwatch2.camera.hapi import HapiClient, HapiError, is_default_password, md5_password


def client(cam, password="123456"):
    return HapiClient(cam.host, "admin", password, port=cam.port, timeout=3)


def test_md5_helpers():
    assert md5_password("123456") == "e10adc3949ba59abbe56e057f20f883e"     # the doc's example hash
    assert md5_password("E10ADC3949BA59ABBE56E057F20F883E") == "e10adc3949ba59abbe56e057f20f883e"
    assert is_default_password("123456") and not is_default_password("Str0ng!pass")


def test_wrong_password_is_refused(vcam):
    with pytest.raises(HapiError) as e:
        client(vcam, "wrong").device_info()
    assert e.value.code == -2


def test_md5_password_is_sent_not_plain(vcam):
    c = client(vcam, "123456")
    assert c._auth()["password"] == hashlib.md5(b"123456").hexdigest()
    assert c.device_info()["device_type"] == "BC-IQ56YDW-8MQ"


def test_info_calls(make_vcam):
    c = client(make_vcam(detect_vehicles=True))
    assert {"ptz_ctrl/move", "event/subscription/regist", "smart/objectdetect/get"} <= c.function_list()
    assert "ptz_control" in c.capability()
    assert set(c.rtsp_urls()) == {"ch1_main", "ch1_sub", "ch2_main", "ch2_sub"}
    assert c.detection_types() == ["HUMAN", "CAR"]


def test_datasheet_variant_detects_humans_only(vcam):
    assert client(vcam).detection_types() == ["HUMAN"]


def test_video_streams_found_next_to_data(vcam):
    streams = client(vcam).video_streams()
    assert [s["streamID"] for s in streams] == [1, 2]


@pytest.mark.parametrize("channel", [1, 2])
def test_snapshot_is_full_size_jpeg(vcam, channel):
    jpg = client(vcam).snapshot(True, channel)
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (1440, 2560, 3)


def test_joystick_move_blocks_and_turns(vcam):
    c = client(vcam)
    t0 = time.monotonic()
    c.ptz_move("right", speed=5, autostop_ms=200)       # 150 deg/s (fast test head) for 0.2 s
    assert time.monotonic() - t0 >= 0.2
    time.sleep(0.1)
    assert vcam.ptz.state()["pan"] == pytest.approx(30.0, abs=0.5)
    c.ptz_move("left", speed=10, autostop_ms=100)
    time.sleep(0.1)
    assert vcam.ptz.state()["pan"] == pytest.approx(0.0, abs=0.5)


def test_input_validation_before_sending(vcam):
    c = client(vcam)
    with pytest.raises(ValueError):
        c.ptz_move("sideways")
    with pytest.raises(ValueError):
        c.preset("call", 0)
    with pytest.raises(ValueError):
        c.zoom("closer")
    with pytest.raises(ValueError):
        c.serial_send(b"x" * 59)


def test_presets_and_zoom(vcam):
    c = client(vcam)
    c.zoom("in", autostop_ms=200)                    # half of the 0.4 s full range
    time.sleep(0.1)
    z = vcam.ptz.state()["zoom"]
    assert 3.0 < z < 6.0
    c.preset("set", 7)
    c.zoom("out", autostop_ms=500)
    time.sleep(0.1)
    assert vcam.ptz.state()["zoom"] == 1.0
    c.preset("call", 7)
    time.sleep(0.4)
    assert vcam.ptz.state()["zoom"] == pytest.approx(z)
    with pytest.raises(HapiError):
        c.preset("call", 99)                          # never saved


def test_setup_turns_off_builtin_tracking(vcam):
    c = client(vcam)
    done = c.disable_builtin_tracking()
    assert "TrackAutoOff" in done
    act = vcam.configs["objectdetect"]["AlarmAction"]
    assert act["track_human_enable"] == 0 and act["auto_zoom_enable"] == 0 and act["gunball_track_mode"] == 0
    assert not vcam.builtin_tracking
    assert act["AudioAction"]["intervalsecnods"] == 10       # firmware spelling kept, rest untouched


def test_enable_event_reporting(vcam):
    c = client(vcam)
    assert vcam.configs["objectdetect"]["AlarmAction"]["AlarmServer"]["ArmingMode"] == 0
    c.enable_event_reporting(("HUMAN", "CAR"))
    od = vcam.configs["objectdetect"]
    assert od["AlarmAction"]["AlarmServer"]["ArmingMode"] == 1
    assert od["type"]["HUMAN"] == 1 and od["type"]["CAR"] == 1


def test_lights(vcam):
    client(vcam).set_lights(work_mode=1, ctrl_mode=6, brightness=40)
    light = vcam.configs["light"]
    assert (light["led_work_mode"], light["led_ctrl_mode"], light["led_brightness_value"]) == (1, 6, 40)


def test_camera_error_message_is_raised(vcam):
    with pytest.raises(HapiError, match="params error"):
        client(vcam).set_config("light", {"no_such_field": 1})

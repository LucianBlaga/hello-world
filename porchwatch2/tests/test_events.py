import base64
import json
import queue
import time

import pytest
import requests

from porchwatch2.camera.events import EventReceiver, Subscription, fix_name, parse_event
from porchwatch2.camera.hapi import HapiClient

DOC_EXAMPLE = ('{"AlarmType":"MotionDetect","AlarmSubType":"MotionAlarm","TimeStamp":1705310512,"ChannelNo":0,'
               '"NotificationType":0,"OccurFlag":"false","DeviceID":"EF000000000000A3","AlarmInfo":null}')


def test_parse_the_docs_own_example():
    ev = parse_event(DOC_EXAMPLE.encode(), "10.0.0.5")
    assert (ev.alarm_type, ev.sub_type, ev.active, ev.channel) == ("MotionDetect", "MotionAlarm", False, 0)


def test_translation_damaged_names_are_restored():
    assert fix_name("MotiDetect") == "MotionDetect"
    assert fix_name("IPCflict") == "IPConflict"
    assert fix_name("HumanShapeDetect") == "HumanShapeDetect"


def test_picture_is_decoded():
    jpg = b"\xff\xd8" + b"\x00" * 200
    msg = {"AlarmType": "ObjectDetect", "AlarmSubType": "HumanShapeDetect", "OccurFlag": "true", "ChannelNo": 1,
           "Picture": {"size": len(jpg), "data": base64.b64encode(jpg).decode()}}
    ev = parse_event(json.dumps(msg).encode(), "x")
    assert ev.picture == jpg and ev.is_person and ev.active


@pytest.fixture
def receiver():
    got = queue.Queue()
    rx = EventReceiver(got.put, port=0, bind="127.0.0.1", allowed_ips={"127.0.0.1"}).start()
    yield rx, got
    rx.stop()


def _url(rx):
    return f"http://127.0.0.1:{rx.port}/HAPI/V1.0/Event/Notification"


def test_receiver_accepts_form_labelled_json(receiver):
    rx, got = receiver
    r = requests.post(_url(rx), data=DOC_EXAMPLE, headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 200 and got.get(timeout=2).alarm_type == "MotionDetect"


def test_receiver_rejects_strangers_and_junk():
    rx = EventReceiver(lambda e: None, port=0, bind="127.0.0.1", allowed_ips={"10.9.9.9"}).start()
    try:
        assert requests.post(_url(rx), data=DOC_EXAMPLE).status_code == 403
    finally:
        rx.stop()
    rx = EventReceiver(lambda e: None, port=0, bind="127.0.0.1").start()
    try:
        assert requests.post(f"http://127.0.0.1:{rx.port}/other", data=DOC_EXAMPLE).status_code == 404
        assert requests.post(_url(rx), data="{not json").status_code == 400
        import socket
        with socket.create_connection(("127.0.0.1", rx.port)) as sock:   # claims 50 MB, sends nothing
            sock.sendall(b"POST /HAPI/V1.0/Event/Notification HTTP/1.1\r\nHost: x\r\nContent-Length: 52428800\r\n\r\n")
            assert sock.recv(64).startswith(b"HTTP/1.0 413")
    finally:
        rx.stop()


def _wait(q, timeout=3):
    return q.get(timeout=timeout)


def test_end_to_end_person_alerts(vcam, receiver):
    rx, got = receiver
    cam = HapiClient(vcam.host, "admin", "123456", port=vcam.port)
    sub = Subscription(cam, rx.port, server="127.0.0.1", duration=60)
    sub.renew()
    with vcam.lock:
        person = vcam.scene.add_person(1, x=0.0)
    vcam.tick(0.05)
    time.sleep(0.3)
    assert got.empty()                       # reporting to the alarm server is off by default
    with vcam.lock:
        person.x = -60                       # leave and come back once reporting is on
    vcam.tick(0.05)
    cam.enable_event_reporting()
    with vcam.lock:
        person.x = 0.0
    vcam.tick(0.05)
    ev = _wait(got)
    assert ev.is_person and ev.active and ev.channel == 1 and ev.picture[:2] == b"\xff\xd8"
    with vcam.lock:
        person.x = 40.0                      # out of the wide view
    vcam.tick(0.05)
    ev = _wait(got)
    assert ev.is_person and not ev.active


def test_subscription_survives_camera_forgetting_it(vcam):
    cam = HapiClient(vcam.host, "admin", "123456", port=vcam.port)
    sub = Subscription(cam, 9998, server="127.0.0.1", duration=60)
    sub.renew()
    first = sub.sub_id
    sub.renew()                              # plain refresh keeps the ID
    assert sub.sub_id == first and len(vcam.subscriptions) == 1
    vcam.subscriptions.clear()               # e.g. the camera rebooted
    sub.renew()
    assert sub.sub_id is not None and len(vcam.subscriptions) == 1
    sub.stop(unsubscribe=True)
    assert vcam.subscriptions == {}


def test_same_server_and_port_is_not_duplicated(vcam):
    cam = HapiClient(vcam.host, "admin", "123456", port=vcam.port)
    a = cam.event_register("127.0.0.1", 9998, 60)
    b = cam.event_register("127.0.0.1", 9998, 60)
    assert a["ID"] == b["ID"]

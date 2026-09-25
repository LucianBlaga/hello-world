import time
from types import SimpleNamespace

import numpy as np
import pytest

from porchwatch.config import Config, config_to_dict, load_config, save_config, update_config
from porchwatch.detectors import PERSON, Detection, filter_zones
from porchwatch.storage import Recorder, Storage, enforce_retention


def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.recording.width, cfg.recording.height = 1920, 1080
    cfg.detection.ignore_zones = [[0.0, 0.0, 0.2, 0.3]]
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    again = load_config(path)
    assert config_to_dict(again) == config_to_dict(cfg)
    assert isinstance(again.detection.watch_zone, tuple)


def test_unknown_setting_rejected():
    with pytest.raises(ValueError):
        update_config(Config(), {"recording": {"nonsense": 1}})


def test_zone_filter():
    cfg = Config().detection
    cfg.ignore_zones = [(0.0, 0.0, 0.5, 1.0)]
    dets = [Detection(PERSON, (10, 10, 50, 90), 0.9), Detection(PERSON, (500, 10, 560, 90), 0.9)]
    kept = filter_zones(dets, 640, 360, cfg)
    assert len(kept) == 1 and kept[0].box[0] == 500


@pytest.mark.parametrize("mode", ["events", "continuous"])
def test_recorder_writes_video(tmp_path, mode):
    cfg = Config().recording
    cfg.output_dir, cfg.mode, cfg.codec = str(tmp_path / "rec"), mode, "MJPG"
    cfg.width, cfg.height, cfg.fps, cfg.pre_record_s, cfg.post_record_s = 320, 180, 10, 1, 0.5
    rec = Recorder(cfg)
    frame = np.zeros((360, 640, 3), np.uint8)
    t = time.time()
    for i in range(40):     # 4 s of camera at 10 fps; active from 2 s to 3 s
        rec.feed(frame, t + i * 0.1, active=20 <= i < 30)
    rec.close()
    files = list((tmp_path / "rec").rglob("*.avi"))
    assert len(files) == 1 and files[0].stat().st_size > 1000
    import cv2
    n = int(cv2.VideoCapture(str(files[0])).get(cv2.CAP_PROP_FRAME_COUNT))
    if mode == "continuous":
        assert n == 40
    else:                   # 1 s pre-roll + 1 s active + ~0.5 s post-roll
        assert 20 <= n <= 28, n


def test_retention_removes_old_files(tmp_path):
    cfg = Config()
    cfg.recording.output_dir = str(tmp_path / "rec")
    cfg.capture.output_dir = str(tmp_path / "cap")
    old = tmp_path / "rec" / "2020-01-01" / "old.mp4"
    new = tmp_path / "rec" / "2026-09-25" / "new.mp4"
    for p in (old, new):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"x" * 10)
    past = time.time() - 40 * 86400
    import os
    os.utime(old, (past, past))
    assert enforce_retention(cfg.capture, cfg.recording) == 1
    assert not old.exists() and new.exists() and not old.parent.exists()


def _client(tmp_path, password=""):
    from porchwatch.web import create_app

    cfg = Config()
    cfg.capture.output_dir = str(tmp_path / "cap")
    cfg.recording.output_dir = str(tmp_path / "rec")
    cfg.web.password = password
    applied = {}
    ctx = SimpleNamespace(
        cfg=cfg, config_path=str(tmp_path / "config.yaml"), latest_jpeg=None, latest_raw_jpeg=None,
        status={"state": "watching"}, fps=12.0, recorder=None, paused=False, error=None,
        storage=Storage(cfg.capture),
        apply_config=lambda new, restart: applied.update(cfg=new, restart=restart),
        manual_ptz=lambda a: a == "home",
        manual_rec=False,
    )
    ctx.set_manual_record = lambda on: setattr(ctx, "manual_rec", on)
    return create_app(ctx).test_client(), ctx, applied


def test_web_settings_save(tmp_path):
    client, ctx, applied = _client(tmp_path)
    r = client.get("/api/settings").get_json()
    assert any(s["title"] == "Recording" for s in r["schema"])
    r = client.post("/api/settings", json={"recording": {"fps": 10, "width": 1920, "height": 1080},
                                           "tracking": {"gain": 0.3}}).get_json()
    assert r["ok"] and r["restarted"]
    assert set(r["changed"]) == {"recording.fps", "recording.width", "recording.height", "tracking.gain"}
    assert applied["cfg"].recording.fps == 10
    assert load_config(ctx.config_path).recording.width == 1920
    # live-only change does not restart the camera
    r = client.post("/api/settings", json={"tracking": {"gain": 0.5}}).get_json()
    assert r["ok"] and not r["restarted"]
    bad = client.post("/api/settings", json={"camera": {"bogus": 1}})
    assert bad.status_code == 400


def test_web_status_events_and_auth(tmp_path):
    client, ctx, _ = _client(tmp_path)
    ctx.storage.save_event({"kind": "vehicle", "label": "car", "plate": "B123ABC", "reason": "plate read"},
                           {"plate": np.zeros((20, 60, 3), np.uint8)})
    assert client.get("/api/status").get_json()["state"] == "watching"
    ev = client.get("/api/events").get_json()
    assert ev[0]["plate"] == "B123ABC"
    assert client.get("/captures/" + ev[0]["files"]["plate"]).status_code == 200
    assert client.get("/").status_code == 200

    locked, _, _ = _client(tmp_path, password="s3cret")
    assert locked.get("/api/status").status_code == 401
    import base64
    hdr = {"Authorization": "Basic " + base64.b64encode(b"admin:s3cret").decode()}
    assert locked.get("/api/status", headers=hdr).status_code == 200


def test_recording_plays_in_real_time_when_frames_arrive_slowly(tmp_path):
    import cv2
    cfg = Config().recording
    cfg.output_dir, cfg.mode, cfg.codec = str(tmp_path / "rec"), "continuous", "MJPG"
    cfg.width, cfg.height, cfg.fps = 320, 180, 30
    rec = Recorder(cfg)
    frame = np.zeros((180, 320, 3), np.uint8)
    t = time.time()
    for i in range(51):             # frames arrive at 17 fps for 3 s
        rec.feed(frame, t + i / 17, active=True)
    rec.close()
    f = next((tmp_path / "rec").rglob("*.avi"))
    cap = cv2.VideoCapture(str(f))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    assert 2.8 <= duration <= 3.2, duration


def test_web_rejects_unwritable_folder(tmp_path):
    client, ctx, applied = _client(tmp_path)
    blocker = tmp_path / "afile"
    blocker.write_text("x")                 # a file where a folder is expected
    r = client.post("/api/settings", json={"recording": {"output_dir": str(blocker / "rec")}})
    assert r.status_code == 400 and "Recordings folder" in r.get_json()["error"]
    assert not applied
    good = tmp_path / "other_drive" / "recordings"
    r = client.post("/api/settings", json={"recording": {"output_dir": str(good)}}).get_json()
    assert r["ok"] and r["restarted"] and good.is_dir()


@pytest.mark.parametrize("mode", ["off", "events"])
def test_manual_rec_button_records_and_stops_immediately(tmp_path, mode):
    import cv2
    cfg = Config().recording
    cfg.output_dir, cfg.mode, cfg.codec = str(tmp_path / "rec"), mode, "MJPG"
    cfg.width, cfg.height, cfg.fps, cfg.pre_record_s, cfg.post_record_s = 320, 180, 10, 0, 10
    rec = Recorder(cfg)
    frame = np.zeros((360, 640, 3), np.uint8)
    t = time.time()
    for i in range(60):             # 6 s; REC pressed from 1 s to 3 s, nothing detected
        rec.feed(frame, t + i * 0.1, active=False, manual=10 <= i < 30)
    rec.close()
    files = list((tmp_path / "rec").rglob("*.avi"))
    assert len(files) == 1 and "_manual" in files[0].name, files
    n = int(cv2.VideoCapture(str(files[0])).get(cv2.CAP_PROP_FRAME_COUNT))
    assert 19 <= n <= 22, n        # ~2 s: stopped right away, no 10 s post-record


def test_web_record_button(tmp_path):
    client, ctx, _ = _client(tmp_path)
    assert client.post("/api/record", json={"on": True}).get_json() == {"manual_rec": True}
    assert client.get("/api/status").get_json()["manual_rec"] is True
    assert client.post("/api/record", json={"on": False}).get_json() == {"manual_rec": False}

"""Tests for the field-audit fixes: web security, validation, robustness, and
direct tests for the gimbal model, motion meter, re-capture suppression and the
whole App wiring."""
import threading
import time

import cv2
import numpy as np
import pytest

from porchwatch.camera import NullPTZ, PTZState
from porchwatch.config import Config, save_config
from porchwatch.controller import Controller, MotionMeter
from porchwatch.detectors import PERSON, VEHICLE, Detection
from porchwatch.storage import Recorder
from porchwatch.tracker import Track
from porchwatch.web import sanitize_config
from test_config_storage_web import _client


# ------------------------------------------------------------------ web security

def test_post_requires_json_so_plain_forms_from_other_sites_cant_act(tmp_path):
    client, ctx, _ = _client(tmp_path)
    r = client.post("/api/record", data="on=true", content_type="application/x-www-form-urlencoded")
    assert r.status_code == 415
    assert ctx.manual_rec is False


def test_cross_site_origin_refused(tmp_path):
    client, ctx, _ = _client(tmp_path)
    r = client.post("/api/record", json={"on": True}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and ctx.manual_rec is False
    ok = client.post("/api/record", json={"on": True}, headers={"Origin": "http://localhost"})
    assert ok.status_code == 200 and ctx.manual_rec is True


def test_dns_rebinding_host_refused(tmp_path):
    client, ctx, _ = _client(tmp_path)
    assert client.get("/api/status", headers={"Host": "evil.example:8080"}).status_code == 403
    assert client.get("/api/status", headers={"Host": "192.168.1.20:8080"}).status_code == 200
    assert client.get("/api/status", headers={"Host": "[::1]:8080"}).status_code == 200
    ctx.cfg.web.allowed_hosts = ["mypc.local"]
    assert client.get("/api/status", headers={"Host": "mypc.local:8080"}).status_code == 200


def test_password_never_sent_to_browser_and_kept_when_blank(tmp_path):
    import base64
    client, ctx, applied = _client(tmp_path, password="s3cret")
    hdr = {"Authorization": "Basic " + base64.b64encode(b"admin:s3cret").decode()}
    r = client.get("/api/settings", headers=hdr).get_json()
    assert r["values"]["web"]["password"] == "" and r["password_set"] is True
    assert "s3cret" not in client.get("/api/settings", headers=hdr).get_data(as_text=True)
    # saving the page (blank password field) keeps the password
    r = client.post("/api/settings", json={"web": {"password": ""}, "tracking": {"gain": 0.5}}, headers=hdr)
    assert r.get_json()["ok"] and applied["cfg"].web.password == "s3cret"
    # NONE removes it
    r = client.post("/api/settings", json={"web": {"password": "NONE"}}, headers=hdr)
    assert r.get_json()["ok"] and applied["cfg"].web.password == ""


@pytest.mark.parametrize("payload,fragment", [
    ({"recording": {"fps": 0}}, "Recording frame rate"),
    ({"tracking": {"gain": 5}}, "Follow speed"),
    ({"recording": {"width": 123, "height": 45}}, "Recording resolution"),
    ({"camera": {"home_zoom": "big"}}, "Home zoom"),
    ({"recording": {"trigger_people": "yes"}}, "Record on people"),
    ({"detection": {"watch_zone": [0.5, 0, 0.2, 1]}}, "Watch zone"),
    ({"detection": {"ignore_zones": [[0, 0, 2, 1]]}}, "Ignore zones"),
])
def test_out_of_range_settings_rejected(tmp_path, payload, fragment):
    client, _, applied = _client(tmp_path)
    r = client.post("/api/settings", json=payload)
    assert r.status_code == 400 and fragment in r.get_json()["error"]
    assert not applied


def test_bad_values_in_config_file_are_reset_on_start():
    cfg = Config()
    cfg.recording.fps = 0
    cfg.tracking.gain = -3
    cfg.detection.model = "my-custom-model.pt"     # free choice in the file: left alone
    fixes = sanitize_config(cfg)
    assert cfg.recording.fps == Config().recording.fps and cfg.tracking.gain == Config().tracking.gain
    assert cfg.detection.model == "my-custom-model.pt"
    assert len(fixes) == 2


def test_concurrent_config_saves_dont_collide(tmp_path):
    path = tmp_path / "config.yaml"
    errors = []

    def saver():
        try:
            for _ in range(20):
                save_config(Config(), path)
        except Exception as exc:        # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=saver) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors and path.exists()
    assert not list(tmp_path.glob("*.tmp"))


# ------------------------------------------------------------------ recorder / encoder

def test_recording_never_overwrites_an_existing_file(tmp_path):
    from datetime import datetime
    cfg = Config().recording
    cfg.output_dir, cfg.mode, cfg.codec, cfg.width, cfg.height, cfg.fps = str(tmp_path), "off", "MJPG", 160, 90, 10
    t = time.time()
    dt = datetime.fromtimestamp(t)
    day = tmp_path / dt.strftime("%Y-%m-%d")
    day.mkdir()
    existing = day / (dt.strftime("%H%M%S") + "_manual.avi")
    existing.write_bytes(b"earlier recording")
    rec = Recorder(cfg)
    frame = np.zeros((90, 160, 3), np.uint8)
    for i in range(10):
        rec.feed(frame, t + i * 0.1, active=False, manual=True)
        time.sleep(0.01)
    rec.close()
    assert existing.read_bytes() == b"earlier recording"
    assert any("_manual_2" in p.name for p in day.iterdir())


def test_recorder_memory_is_bounded_when_encoder_stalls(tmp_path):
    cfg = Config().recording
    cfg.output_dir, cfg.mode, cfg.codec, cfg.fps = str(tmp_path), "continuous", "MJPG", 30
    rec = Recorder(cfg)
    rec._running = False                # recorder thread stops taking frames
    rec._thread.join(timeout=5)
    big = np.zeros((2160, 3840, 3), np.uint8)
    t = time.time()
    for i in range(400):
        rec.feed(big, t + i / 30, active=True)
    held = sum(item[0].nbytes for item in list(rec.q.queue))
    assert held <= 620e6                # bounded (~600 MB), not 400 x 25 MB of 4K frames
    assert rec.q.qsize() <= cfg.fps * 3 # at most ~3 s of frames


def test_recorder_shrinks_frames_before_queueing(tmp_path):
    cfg = Config().recording
    cfg.output_dir, cfg.mode, cfg.codec, cfg.fps = str(tmp_path), "continuous", "MJPG", 30
    cfg.width, cfg.height = 1920, 1080
    rec = Recorder(cfg)
    rec._running = False
    rec._thread.join(timeout=5)
    t = time.time()
    for i in range(60):                 # 2 s of 4K at 30 fps while the recorder is stuck
        rec.feed(np.zeros((2160, 3840, 3), np.uint8), t + i / 30, active=True)
    assert rec.q.qsize() == 60          # nothing dropped: a 2 s stall is absorbed
    assert rec.q.queue[0][0].shape == (1080, 1920, 3)


def test_failing_encoder_reports_and_does_not_hang(tmp_path):
    from porchwatch.video import FFmpegWriter, ffmpeg_exe
    if ffmpeg_exe() is None:
        pytest.skip("imageio-ffmpeg not installed")
    w = FFmpegWriter(tmp_path / "x.mp4", 15, (160, 90), "no_such_encoder", 28)
    t0 = time.time()
    for _ in range(50):
        w.write(np.zeros((90, 160, 3), np.uint8))
    w.release()
    assert time.time() - t0 < 20 and not w.isOpened()


def test_audio_thread_not_leaked_when_microphone_fails():
    from porchwatch.audio import AudioSource

    class BrokenMic:
        def __init__(self, *a):
            pass

        def start(self):
            raise RuntimeError("device busy")

        def close(self):
            pass

    before = sum(1 for t in threading.enumerate() if t.name == "audio" and t.is_alive())
    for _ in range(3):
        with pytest.raises(RuntimeError):
            AudioSource(Config().audio, keep_s=1, stream_factory=BrokenMic)
    time.sleep(0.8)
    after = sum(1 for t in threading.enumerate() if t.name == "audio" and t.is_alive())
    assert after == before


# ------------------------------------------------------------------ gimbal model, motion meter

def test_gimbal_estimate_follows_latency_and_speed():
    now = [100.0]
    cfg = Config().camera
    cfg.ptz_latency_s, cfg.pan_speed_dps, cfg.command_interval_s = 0.25, 60.0, 0
    ptz = NullPTZ(cfg, clock=lambda: now[0])
    ptz.move(PTZState(30, 0, 1))
    assert ptz.est.pan == 0
    now[0] += 0.25
    assert ptz.update_estimate(now[0]).pan == pytest.approx(0, abs=1e-6)   # still in its delay
    now[0] += 0.25
    assert ptz.update_estimate(now[0]).pan == pytest.approx(15, abs=0.5)   # 60 deg/s
    now[0] += 1.0
    assert ptz.update_estimate(now[0]).pan == pytest.approx(30)
    assert ptz.model_until == pytest.approx(100 + 0.25 + 0.5 + cfg.ptz_settle_margin_s)


def _textured(w=640, h=360, seed=0):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, (h, w + 100, 3), dtype=np.uint8), (0, 0), 2)


def test_motion_meter_sees_camera_turn_but_not_a_walking_person():
    world = _textured()
    m = MotionMeter()
    m.update(world[:, 0:640])
    m.update(world[:, 20:660])                  # the camera turned: whole scene slid 20 px
    assert m.moving and m.shift_px == pytest.approx(20 * 192 / 640, abs=1.0)

    m = MotionMeter()
    for x in range(0, 200, 10):                 # camera still, a person walks across
        frame = world[:, 0:640].copy()
        cv2.rectangle(frame, (x, 100), (x + 60, 300), (0, 180, 0), -1)
        m.update(frame, [(x, 100, x + 60, 300)])
        assert not m.moving, m.shift_px         # the blanked person must not look like camera motion


# ------------------------------------------------------------------ re-capture suppression, low fps

def _ctl():
    cfg = Config()
    return Controller(cfg, NullPTZ(cfg.camera), None, None, None)


def test_suppression_follows_the_captured_car():
    ctl = _ctl()
    ctl.recent.append((VEHICLE, 10.0, 0.0, 8.0, 0.0, 100.0, 130.0))   # captured at pan 10, 8 deg/s
    assert ctl._suppressed(VEHICLE, 10 + 8 * 2, 0, 102.0)              # where it should be 2 s later
    assert not ctl._suppressed(VEHICLE, -30, 0, 102.0)                 # a different car
    assert not ctl._suppressed(PERSON, 26, 0, 102.0)                   # a person there: not the car
    assert not ctl._suppressed(VEHICLE, 26, 0, 131.0)                  # after the hold time


def test_moving_car_recognised_at_low_frame_rate():
    tr = Track(1, Detection(VEHICLE, (0, 0, 10, 10), 0.9))
    tr.history.clear()
    for i in range(4):                          # 4 fps, 120 px/s
        tr.history.append((100 + i * 0.25, 100 + 30 * i, 200))
    assert tr.travel(100.75, 0.6) == pytest.approx(72, rel=0.05)


def test_picture_fallback_matches_when_angle_prediction_is_off():
    ctl = _ctl()
    from porchwatch.controller import Target, State
    ctl.ptz.est = PTZState(0, 0, 4.0)
    det = Detection(PERSON, (300, 80, 340, 280), 0.9)
    ctl.target = Target(PERSON, det, "person", 0, 0, (25.0, 0.0), world_t=0, seen_zoom=4.0)  # wrong by ~25 deg
    ctl.state = State.TRACK
    moved = Detection(PERSON, (310, 80, 350, 280), 0.9)
    assert ctl._match([moved], 640, 360, 0.1) is moved


# ------------------------------------------------------------------ whole app, end to end

def test_app_runs_end_to_end_on_a_video(tmp_path, monkeypatch):
    """Frames -> fake detector -> controller -> recorder -> events, all wired up."""
    from porchwatch.app import App
    from test_controller import FakeFaces, FakePlates, RED, texture

    monkeypatch.chdir(tmp_path)
    video = str(tmp_path / "street.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 360))
    for i in range(75):                                         # 3 s: a person walks in
        img = np.full((360, 640, 3), 90, np.uint8)
        x = 100 + i * 3
        texture(img, x, 60, x + 50, 300, (60, 170, 60))
        texture(img, x + 15, 60, x + 35, 90, RED)
        vw.write(img)
    vw.release()

    class BlobDetector:
        def __init__(self, cfg):
            self.cfg = cfg

        def __call__(self, f):
            m = cv2.inRange(f, np.array([26, 92, 26], np.uint8), np.array([70, 180, 70], np.uint8))
            ys, xs = np.nonzero(m)
            if len(xs) < 200:
                return []
            return [Detection(PERSON, (xs.min(), ys.min(), xs.max(), ys.max()), 0.9, "person")]

    cfg = Config()
    cfg.web.enabled = False
    cfg.capture.output_dir, cfg.recording.output_dir = str(tmp_path / "cap"), str(tmp_path / "rec")
    cfg.recording.mode, cfg.recording.codec = "continuous", "MJPG"
    cfg.capture.min_face_px, cfg.capture.min_sharpness = 15, 10
    app = App(cfg, str(tmp_path / "config.yaml"), video=video, detector_factory=BlobDetector,
              face_factory=FakeFaces, plate_factory=FakePlates, no_preview=True)
    th = threading.Thread(target=app.run, daemon=True)
    th.start()
    th.join(timeout=60)
    assert not th.is_alive() and app.error is None
    assert app.fps > 0
    assert list((tmp_path / "rec").rglob("*.avi")), "no recording written"
    events = app.storage.recent_events()
    assert events and events[0]["kind"] == "person" and "face" in events[0]["files"]
    assert (tmp_path / "logs" / "tracking.log").exists()


# ------------------------------------------------------------------ second field audit

def test_failed_chase_is_not_repeated_after_3_seconds(tmp_path):
    """A chase that captured nothing used to be blocked for only 3 s: the camera kept
    swinging back to the same person."""
    from test_controller import Sim, person
    sim = Sim(tmp_path)
    sim.ctl.faces = lambda crop: []                 # never finds a face -> chase ends without capture
    sim.objects.append(person(sim, pan0=-10, speed=0.0))
    starts = []
    sim.run(20, lambda s: starts.append(s.ctl.state.value))
    chases = "".join(x[0] for x in starts).count("wt")
    assert chases == 1, f"{chases} chases of the same standing person in 20 s"


def test_ghost_filter_is_time_based_at_high_frame_rate(tmp_path):
    """At 30 fps a 0.3 s flicker has 9 sightings - more than a frame-count rule needs."""
    from test_controller import Sim, person, REAL_GIMBAL
    from porchwatch.controller import State
    sim = Sim(tmp_path, **REAL_GIMBAL, fps=30)
    ghost = person(sim, pan0=-20, speed=0.0)
    ghost["life"] = 0.3
    sim.objects.append(ghost)
    states = []
    sim.run(2, lambda s: states.append(s.ctl.state))
    assert State.TRACK not in states


def test_settings_not_applied_when_file_cannot_be_saved(tmp_path, monkeypatch):
    import porchwatch.web as web
    client, ctx, applied = _client(tmp_path)

    def boom(*a, **k):
        raise PermissionError("config.yaml is locked")
    monkeypatch.setattr(web, "save_config", boom)
    r = client.post("/api/settings", json={"tracking": {"gain": 0.3}})
    assert r.status_code == 500 and "locked" in r.get_json()["error"]
    assert not applied


def test_status_does_not_walk_the_disk_every_request(tmp_path, monkeypatch):
    import porchwatch.storage as st
    calls = []
    monkeypatch.setattr(st, "_disk_usage", lambda p: calls.append(p) or {"used_gb": 0, "free_gb": 1, "disk_gb": 1})
    st._usage_cache.clear()
    for _ in range(5):
        st.disk_usage(str(tmp_path))
    assert len(calls) == 1


def test_leftover_model_files_are_tidied(tmp_path, monkeypatch):
    import os
    import porchwatch.detectors as det
    monkeypatch.setattr(det, "MODELS_DIR", tmp_path / "models")
    (tmp_path / "yolo26m.pt").write_bytes(b"weights")
    stale = tmp_path / "yolo11n.pt.part"
    stale.write_bytes(b"partial")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    det.tidy_model_files(tmp_path)
    assert (tmp_path / "models" / "yolo26m.pt").exists()
    assert not (tmp_path / "yolo26m.pt").exists() and not stale.exists()


# ------------------------------------------------------------------ TensorRT option

def test_tensorrt_engine_size_and_name():
    from porchwatch.detectors import engine_path, engine_size
    assert engine_size(1280, (2160, 3840, 3)) == (736, 1280)      # 16:9, not a padded square
    assert engine_size(3840, (1080, 1920, 3)) == (1088, 1920)     # capped at the frame
    assert engine_size(1280, (1920, 1080, 3)) == (1280, 736)      # portrait camera
    a = engine_path("yolo26m.pt", (736, 1280), True, "10.1")
    assert a.name == "yolo26m_1280x736_fp16_trt10.1.engine"
    # anything that makes an engine invalid gives a different file
    assert len({a, engine_path("yolo26l.pt", (736, 1280), True, "10.1"),
                engine_path("yolo26m.pt", (1088, 1920), True, "10.1"),
                engine_path("yolo26m.pt", (736, 1280), False, "10.1"),
                engine_path("yolo26m.pt", (736, 1280), True, "10.2")}) == 5


def test_tensorrt_missing_falls_back_to_pytorch(monkeypatch):
    import builtins
    from porchwatch.detectors import ObjectDetector
    real_import = builtins.__import__

    def no_trt(name, *a, **k):
        if name in ("tensorrt", "onnx", "onnxslim"):
            raise ImportError("missing")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_trt)
    det = ObjectDetector.__new__(ObjectDetector)          # skip loading a real YOLO model
    det.cfg = Config().detection
    det.cfg.tensorrt = True
    det.cuda, det.precision, det.engine, det._engine_thread, det.engine_status = True, {}, None, None, "off"
    det._start_engine_build((2160, 3840, 3))
    assert det.engine is None and det._engine_thread is None
    assert "pip install tensorrt onnx onnxslim" in det.engine_status

    det.cuda, det.engine_status = False, "off"
    det._start_engine_build((2160, 3840, 3))
    assert det.engine_status == "needs an NVIDIA GPU"

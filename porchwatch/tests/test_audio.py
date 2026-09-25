import subprocess
import threading
import time
import wave

import cv2
import numpy as np
import pytest

from porchwatch.audio import AudioProcessor, AudioSource, ffmpeg_exe, wav_stream_header
from porchwatch.config import Config
from porchwatch.storage import Recorder

SR = 16000


def tone(n, freq=440, amp=0.05, sr=SR, t0=0):
    t = (np.arange(n) + t0) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_boost_filter_and_limiter():
    cfg = Config().audio
    cfg.gain_db, cfg.high_pass_hz = 12, 0
    p = AudioProcessor(cfg, SR)
    out = p.process(tone(1600))
    assert np.max(np.abs(out)) == pytest.approx(0.05 * 10 ** (12 / 20), rel=0.05)   # ~4x louder

    cfg.gain_db = 36                                   # would clip without the limiter
    out = p.process(tone(1600, amp=0.5))
    assert np.max(np.abs(out)) <= 1.0

    cfg.gain_db, cfg.high_pass_hz = 0, 200            # rumble (40 Hz) removed, voice (1 kHz) kept
    p = AudioProcessor(cfg, SR)
    for _ in range(5):
        low = p.process(tone(1600, freq=40, amp=0.3))
    p2 = AudioProcessor(cfg, SR)
    for _ in range(5):
        high = p2.process(tone(1600, freq=1000, amp=0.3))
    assert np.max(np.abs(low)) < 0.05 and np.max(np.abs(high)) > 0.25


def test_noise_gate_quiets_hiss_only():
    cfg = Config().audio
    cfg.gain_db, cfg.high_pass_hz, cfg.noise_gate, cfg.gate_threshold_db = 0, 0, True, -40
    p = AudioProcessor(cfg, SR)
    for _ in range(20):
        hiss = p.process(np.random.default_rng(0).normal(0, 0.001, 800).astype(np.float32))
    for _ in range(5):
        voice = p.process(tone(800, amp=0.2))
    assert np.max(np.abs(hiss)) < 0.0005 and np.max(np.abs(voice)) > 0.15


class FakeMic:
    """Calls the audio callback in real time with a continuous tone."""

    def __init__(self, callback, sr, block):
        self.cb, self.sr, self.block = callback, sr, block
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        n = 0
        while self.running:
            self.cb(tone(self.block, t0=n)[:, None], self.block, None, None)
            n += self.block
            time.sleep(self.block / self.sr)

    def stop(self):
        self.running = False

    def close(self):
        pass


def _source(pre_s=1.0):
    cfg = Config().audio
    cfg.sample_rate, cfg.gain_db, cfg.high_pass_hz = SR, 0, 0
    return AudioSource(cfg, keep_s=pre_s, stream_factory=FakeMic)


def test_wav_is_time_aligned_and_includes_pre_record(tmp_path):
    src = _source()
    time.sleep(1.5)                           # history builds up
    start = time.time() - 1.0                 # recording starts 1 s in the past (pre-record)
    src.start_recording(tmp_path / "a.wav", start)
    time.sleep(1.0)
    path = src.stop_recording()
    src.close()
    with wave.open(str(path)) as w:
        seconds = w.getnframes() / w.getframerate()
    assert 1.8 <= seconds <= 2.2, seconds     # 1 s pre-record + 1 s live, no gaps or doubles


def test_listen_stream_starts_with_wav_header():
    src = _source()
    gen = src.listen()
    assert next(gen) == wav_stream_header(SR)
    chunk = next(gen)
    assert len(chunk) > 0 and len(chunk) % 2 == 0
    gen.close()
    src.close()


@pytest.mark.skipif(ffmpeg_exe() is None, reason="imageio-ffmpeg not installed")
def test_recording_gets_audio_track(tmp_path):
    rc = Config().recording
    rc.output_dir, rc.mode, rc.codec = str(tmp_path / "rec"), "continuous", "mp4v"
    rc.width, rc.height, rc.fps = 320, 180, 15
    src = _source()
    rec = Recorder(rc, audio=src)
    frame = np.zeros((180, 320, 3), np.uint8)
    t_end = time.time() + 2.0
    while time.time() < t_end:
        rec.feed(frame, time.time(), active=True)
        time.sleep(1 / 30)
    rec.close()
    src.close()
    files = list((tmp_path / "rec").rglob("*"))
    videos = [f for f in files if f.suffix == ".mp4"]
    assert len(videos) == 1 and not [f for f in files if f.suffix == ".wav"], files
    info = subprocess.run([ffmpeg_exe(), "-i", str(videos[0])], capture_output=True, text=True).stderr
    assert "Audio: aac" in info and "Video:" in info


def _street_frames(n, w=640, h=360):
    """A textured scene with a moving object, closer to real footage than flat colour."""
    rng = np.random.default_rng(1)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 3)
    for i in range(n):
        f = bg.copy()
        x = (i * 7) % (w - 60)
        cv2.rectangle(f, (x, 150), (x + 60, 300), (40, 160, 40), -1)
        f = cv2.add(f, rng.integers(0, 6, (h, w, 3), dtype=np.uint8))      # sensor noise
        yield f


def _record(tmp_path, codec, crf=28, audio=None):
    rc = Config().recording
    rc.output_dir, rc.mode, rc.codec, rc.crf, rc.encoder = str(tmp_path / codec), "continuous", codec, crf, "cpu"
    rc.width, rc.height, rc.fps = 640, 360, 15
    rec = Recorder(rc, audio=audio)
    t = time.time()
    for i, f in enumerate(_street_frames(60)):         # 4 s of video
        while rec.q.qsize() >= 4:                        # real-time pace, like a camera
            time.sleep(0.002)
        rec.feed(f, t + i / 15, active=True)
    rec.close()
    return [p for p in (tmp_path / codec).rglob("*") if p.is_file()]


@pytest.mark.skipif(ffmpeg_exe() is None, reason="imageio-ffmpeg not installed")
def test_h264_is_much_smaller_than_mp4v_and_crf_controls_size(tmp_path):
    import cv2 as _cv2
    old = _record(tmp_path, "mp4v")[0].stat().st_size
    h264_files = _record(tmp_path, "h264")
    assert h264_files[0].suffix == ".mp4"
    new = h264_files[0].stat().st_size
    cap = _cv2.VideoCapture(str(h264_files[0]))
    assert int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) in range(58, 62)          # all frames, real time
    small = _record(tmp_path / "hi", "h264", crf=36)[0].stat().st_size
    print(f"mp4v {old/1e3:.0f} kB, h264@28 {new/1e3:.0f} kB, h264@36 {small/1e3:.0f} kB")
    assert new < old / 3
    assert small < new / 1.8


@pytest.mark.skipif(ffmpeg_exe() is None, reason="imageio-ffmpeg not installed")
def test_h264_recording_gets_audio(tmp_path):
    src = _source()
    time.sleep(0.5)
    files = _record(tmp_path, "h264", audio=src)
    src.close()
    assert len(files) == 1 and files[0].suffix == ".mp4", files
    info = subprocess.run([ffmpeg_exe(), "-i", str(files[0])], capture_output=True, text=True).stderr
    assert "Video: h264" in info and "Audio: aac" in info

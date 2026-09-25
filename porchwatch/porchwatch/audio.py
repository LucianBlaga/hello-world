"""Microphone capture with boost / filtering, time-aligned WAV recording,
live listening, and muxing audio into the finished video files."""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from .config import AudioConfig

log = logging.getLogger(__name__)


class AudioProcessor:
    """high-pass -> gain (boost) -> noise gate -> limiter, block by block."""

    def __init__(self, cfg: AudioConfig, sample_rate: int):
        self.cfg = cfg
        self.sr = sample_rate
        self._hp_hz = None
        self._sos = None
        self._zi = None
        self._gate = 1.0
        self.level_db = -120.0          # peak level after processing (dBFS)

    def _high_pass(self, x: np.ndarray) -> np.ndarray:
        hz = int(self.cfg.high_pass_hz or 0)
        if hz <= 0:
            self._hp_hz = None
            return x
        from scipy.signal import butter, sosfilt

        if hz != self._hp_hz:
            self._sos = butter(2, hz, "highpass", fs=self.sr, output="sos")
            self._zi = np.zeros((self._sos.shape[0], 2))
            self._hp_hz = hz
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        return y

    def process(self, block: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        x = self._high_pass(block.astype(np.float64))
        x = x * 10 ** (cfg.gain_db / 20.0)
        if cfg.noise_gate:
            rms_db = 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-12)
            target = 1.0 if rms_db > cfg.gate_threshold_db else 0.05     # -26 dB when closed
            speed = 0.6 if target > self._gate else 0.15                  # fast open, slow close
            new = self._gate + (target - self._gate) * speed
            x = x * np.linspace(self._gate, new, len(x))                  # no clicks
            self._gate = new
        # tanh is ~transparent for quiet sound and rounds off loud peaks: a soft limiter
        x = np.tanh(x) if cfg.limiter else np.clip(x, -1.0, 1.0)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        self.level_db = 20 * np.log10(peak + 1e-9)
        return x.astype(np.float32)


def list_input_devices() -> list[tuple[int, str, str]]:
    import sounddevice as sd

    apis = sd.query_hostapis()
    return [(i, d["name"], apis[d["hostapi"]]["name"])
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def resolve_input_device(device):
    """Index, "" (system default) or part of a name such as "OBSBOT"."""
    if device in ("", None):
        return None
    if isinstance(device, int) or str(device).strip().isdigit():
        return int(device)
    import sounddevice as sd

    default_api = sd.default.hostapi
    matches = [(i, n) for i, n, _ in list_input_devices() if str(device).lower() in n.lower()]
    if not matches:
        names = sorted({n for _, n, _ in list_input_devices()})
        raise RuntimeError(f"No microphone named like {device!r}. Found: {names}")
    # Windows lists each mic once per audio API; prefer the default one (MME),
    # which resamples to any rate.
    preferred = [m for m in matches if sd.query_devices(m[0])["hostapi"] == default_api]
    idx, name = (preferred or matches)[0]
    log.info("Using microphone %d: %s", idx, name)
    return idx


class AudioSource:
    """Captures the microphone continuously.

    Keeps a few seconds of history (for pre-record), writes time-aligned WAV
    files for the recorder, and feeds live listeners.
    """

    def __init__(self, cfg: AudioConfig, keep_s: float, stream_factory=None):
        self.cfg = cfg
        self.sr = int(cfg.sample_rate)
        self.proc = AudioProcessor(cfg, self.sr)
        self.keep_s = keep_s + 3.0
        self.ring: deque = deque()              # (t_start, int16 block)
        self.lock = threading.Lock()
        self.listeners: set[queue.Queue] = set()
        self._wav: wave.Wave_write | None = None
        self._wav_path: Path | None = None
        self._rec_t0 = 0.0
        self._rec_samples = 0
        self._q: queue.Queue = queue.Queue(maxsize=400)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="audio", daemon=True)
        self._thread.start()
        block = int(self.sr * 0.05)
        if stream_factory is None:
            import sounddevice as sd

            self.stream = sd.InputStream(device=resolve_input_device(cfg.device), channels=1,
                                         samplerate=self.sr, blocksize=block, dtype="float32",
                                         callback=self._callback)
        else:
            self.stream = stream_factory(self._callback, self.sr, block)
        self.stream.start()

    # -- capture ----------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        t_start = time.time() - frames / self.sr + self.cfg.sync_offset_ms / 1000.0
        try:
            self._q.put_nowait((t_start, np.array(indata[:, 0], dtype=np.float32)))
        except queue.Full:
            pass

    def _loop(self):
        while self._running:
            try:
                t0, block = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            pcm = (self.proc.process(block) * 32767).astype(np.int16)
            with self.lock:
                self.ring.append((t0, pcm))
                while self.ring and self.ring[0][0] < t0 - self.keep_s:
                    self.ring.popleft()
                if self._wav is not None:
                    self._write_aligned(t0, pcm)
                for q in list(self.listeners):
                    try:
                        q.put_nowait(pcm.tobytes())
                    except queue.Full:
                        pass

    @property
    def level_db(self) -> float:
        return self.proc.level_db

    # -- recording ----------------------------------------------------------
    def _write_aligned(self, t0: float, pcm: np.ndarray) -> None:
        """Place `pcm` at its timestamp in the WAV: pad gaps with silence, trim overlaps."""
        pos = int(round((t0 - self._rec_t0) * self.sr))
        if pos > self._rec_samples:
            gap = pos - self._rec_samples
            if gap > int(0.02 * self.sr):       # ignore tiny timestamp jitter
                self._wav.writeframes(np.zeros(gap, np.int16).tobytes())
                self._rec_samples += gap
        elif pos < self._rec_samples:
            pcm = pcm[self._rec_samples - pos:]
        if len(pcm):
            self._wav.writeframes(pcm.tobytes())
            self._rec_samples += len(pcm)

    def start_recording(self, path: Path, t_start: float) -> None:
        with self.lock:
            self._close_wav()
            wav = wave.open(str(path), "wb")
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sr)
            self._wav, self._wav_path = wav, path
            self._rec_t0, self._rec_samples = t_start, 0
            for t0, pcm in self.ring:           # pre-record from history
                if t0 + len(pcm) / self.sr > t_start:
                    self._write_aligned(t0, pcm)

    def _close_wav(self) -> Path | None:
        path = None
        if self._wav is not None:
            self._wav.close()
            path = self._wav_path
        self._wav = None
        self._wav_path = None
        return path

    def stop_recording(self) -> Path | None:
        with self.lock:
            return self._close_wav()

    # -- live listening ----------------------------------------------------
    def listen(self):
        """Yields a never-ending 16-bit mono WAV stream."""
        q: queue.Queue = queue.Queue(maxsize=100)
        with self.lock:
            self.listeners.add(q)
        try:
            yield wav_stream_header(self.sr)
            while self._running:
                try:
                    yield q.get(timeout=1.0)
                except queue.Empty:
                    continue
        finally:
            with self.lock:
                self.listeners.discard(q)

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self._running = False
        self._thread.join(timeout=2)
        self.stop_recording()


def wav_stream_header(sr: int) -> bytes:
    big = 0x7FFFFFF0                            # "unknown" length for streaming
    import struct

    return (b"RIFF" + struct.pack("<I", big) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
            b"data" + struct.pack("<I", big))


# ------------------------------------------------------------------ muxing

def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def mux_audio(video: Path, wav: Path, bitrate_kbps: int = 128) -> Path | None:
    """Combine a silent video and a WAV into one file (video is copied, not re-encoded).

    .mp4 stays .mp4; .avi (MJPG/XVID) becomes .mkv. Returns the final path.
    """
    exe = ffmpeg_exe()
    if exe is None:
        log.error("Can't add audio to %s: pip install imageio-ffmpeg", video.name)
        return None
    ext = ".mp4" if video.suffix.lower() == ".mp4" else ".mkv"
    final = video.with_suffix(ext)
    tmp = video.with_name(video.stem + ".muxing" + ext)
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav),
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", f"{bitrate_kbps}k", str(tmp)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    res = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
    if res.returncode != 0 or not tmp.exists():
        log.error("Adding audio to %s failed: %s", video.name, res.stderr.strip()[-300:])
        tmp.unlink(missing_ok=True)
        return None
    os.replace(tmp, final)
    if final != video:
        video.unlink(missing_ok=True)
    wav.unlink(missing_ok=True)
    return final

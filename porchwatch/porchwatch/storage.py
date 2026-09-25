"""Snapshots, event log, video recording and retention clean-up."""
from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import CaptureConfig, RecordingConfig

log = logging.getLogger(__name__)


class Storage:
    """Saves capture events (face / plate / scene images + JSON line log)."""

    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self.root = Path(cfg.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_file = self.root / "events.jsonl"
        self.messages: deque = deque(maxlen=200)
        self._lock = threading.Lock()

    def log(self, msg: str) -> None:
        log.info(msg)
        self.messages.append({"time": datetime.now().isoformat(timespec="seconds"), "msg": msg})

    def save_event(self, meta: dict, images: dict[str, np.ndarray]) -> dict:
        now = datetime.now()
        day = self.root / now.strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        stem = now.strftime("%H%M%S_%f")[:-3] + f"_{meta.get('kind', 'event')}"
        files = {}
        for name, img in images.items():
            if img is None or img.size == 0:
                continue
            if name == "scene" and img.shape[1] > 1920:
                scale = 1920 / img.shape[1]
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            path = day / f"{stem}_{name}.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            files[name] = path.relative_to(self.root).as_posix()
        event = {"time": now.isoformat(timespec="seconds"), **meta, "files": files}
        with self._lock, open(self.events_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        summary = meta.get("plate") or meta.get("label")
        self.log(f"Saved {meta.get('kind')} event ({summary}, {meta.get('reason')})")
        return event

    def recent_events(self, limit: int = 50) -> list[dict]:
        if not self.events_file.exists():
            return []
        with self._lock, open(self.events_file, "r", encoding="utf-8") as fh:
            lines = deque(fh, maxlen=limit)
        out = []
        for line in reversed(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out


def draw_label(img: np.ndarray, text: str, org: tuple, scale: float) -> None:
    """White text on a dark box (a thick outline drifts: Hershey glyphs widen with thickness)."""
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    pad = max(3, int(6 * scale))
    x1, y1, x2, y2 = max(0, x - pad), max(0, y - th - pad), min(img.shape[1], x + tw + pad), min(img.shape[0], y + base + pad)
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * 0.35).astype(np.uint8)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _codec_ext(codec: str) -> str:
    return ".avi" if codec.upper() in ("MJPG", "XVID", "DIVX") else ".mp4"


def _uses_ffmpeg(codec: str) -> bool:
    from .video import FFMPEG_CODECS
    return codec.lower() in FFMPEG_CODECS


class Recorder:
    """Writes video in the background at its own resolution / frame rate.

    mode "continuous": always recording, new file every `segment_minutes`.
    mode "events"    : starts when `active` goes true, keeps `pre_record_s`
                       seconds from before the trigger, stops `post_record_s`
                       after activity ends.
    """

    def __init__(self, cfg: RecordingConfig, audio=None, audio_bitrate_kbps: int = 128):
        self.cfg = cfg
        self.audio = audio                    # porchwatch.audio.AudioSource or None
        self.audio_bitrate_kbps = audio_bitrate_kbps
        self._mux_threads: list[threading.Thread] = []
        self.encoder_name: str | None = None      # ffmpeg encoder, e.g. h264_nvenc / libx264
        self._manual = False
        if _uses_ffmpeg(cfg.codec):
            from .video import pick_encoder
            self.encoder_name = pick_encoder(cfg.codec.lower(), cfg.encoder)
            if self.encoder_name:
                log.info("Recording with %s (compression %d)", self.encoder_name, cfg.crf)
            else:
                log.warning("No %s encoder available (pip install imageio-ffmpeg); using mp4v", cfg.codec)
        self.root = Path(cfg.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.writer: cv2.VideoWriter | None = None
        self.current_file: Path | None = None
        self.segment_started = 0.0
        self.last_active = 0.0
        self.next_frame_t = 0.0
        self.video_t: float | None = None     # timeline position of the next video frame
        # Pre-roll is kept JPEG-compressed to save memory.
        self.preroll: deque = deque(maxlen=max(1, int(cfg.pre_record_s * cfg.fps) + 1))
        # Raw frames are up to 25 MB each at 4K: the queue is bounded in feed() to
        # ~1 s of frames and ~300 MB, so a stalled encoder can't eat the memory.
        self.q: queue.Queue = queue.Queue()
        self._dropped = 0
        self._drop_logged = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="recorder", daemon=True)
        self._thread.start()

    @property
    def recording(self) -> bool:
        return self.writer is not None

    def feed(self, frame: np.ndarray, now: float, active: bool, manual: bool = False) -> None:
        """`manual`: the REC button is on - record regardless of mode or detections."""
        cfg = self.cfg
        if cfg.mode == "off" and not manual and not self.recording:
            return
        if now < self.next_frame_t:
            return                      # down-sample to the recording frame rate
        self.next_frame_t = max(self.next_frame_t + 1.0 / cfg.fps, now - 0.5 / cfg.fps)
        limit = max(4, min(int(cfg.fps), int(300e6 // max(1, frame.nbytes))))
        if self.q.qsize() < limit:
            self.q.put_nowait((frame, now, active or manual, manual))
        else:
            self._dropped += 1
            if now - self._drop_logged > 10:
                log.warning("Recorder falling behind: %d frame(s) dropped", self._dropped)
                self._drop_logged, self._dropped = now, 0

    def _prepare(self, frame, now):
        cfg = self.cfg
        if frame.shape[1] != cfg.width or frame.shape[0] != cfg.height:
            if frame.shape[1] >= 4 * cfg.width:     # cheap pre-shrink (see imaging.downscale)
                frame = cv2.resize(frame, (cfg.width * 2, cfg.height * 2), interpolation=cv2.INTER_NEAREST)
            frame = cv2.resize(frame, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
        else:
            frame = frame.copy()
        if cfg.timestamp_overlay:
            txt = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
            s = cfg.width / 1280
            draw_label(frame, txt, (int(12 * s), cfg.height - int(14 * s)), 0.8 * s)
        return frame

    def _open(self, now):
        cfg = self.cfg
        dt = datetime.fromtimestamp(now)
        day = self.root / dt.strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        kind = "_manual" if self._manual else ("" if cfg.mode == "continuous" else "_event")
        stem = dt.strftime("%H%M%S") + kind
        n = 1
        # Never overwrite: e.g. the hour repeated when clocks go back in autumn.
        while any(day.glob(f"{stem}.*")):
            n += 1
            stem = f"{dt.strftime('%H%M%S')}{kind}_{n}"
        path = day / (stem + _codec_ext(cfg.codec))
        if self.encoder_name:
            from .video import FFmpegWriter
            writer = FFmpegWriter(path, cfg.fps, (cfg.width, cfg.height), self.encoder_name, cfg.crf)
        else:
            codec = "mp4v" if _uses_ffmpeg(cfg.codec) else cfg.codec
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec[:4]), cfg.fps,
                                     (cfg.width, cfg.height))
        if not writer.isOpened():
            log.error("Could not open video writer (codec %s). Try mp4v or MJPG.", cfg.codec)
            return
        self.writer, self.current_file, self.segment_started = writer, path, now
        self.video_t = None
        log.info("Recording to %s", path)

    def _write(self, frame, t):
        """Write `frame` for as many frame slots as have passed, so the video
        plays in real time even when fewer frames arrive than the recording
        frame rate (e.g. detection running at 17 fps, recording set to 30)."""
        fps = self.cfg.fps
        if self.video_t is None:
            self.video_t = t
            if self.audio is not None:          # audio file starts at the first video frame
                self.audio.start_recording(self.current_file.with_suffix(".wav"), t)
        if t < self.video_t - 0.5 / fps:
            return                      # ahead of the timeline: drop
        count = 1 + int((t - self.video_t) * fps)
        count = min(count, 2 * fps)     # cap freezes after long stalls
        for _ in range(count):
            self.writer.write(frame)
        self.video_t += count / fps

    def _close(self):
        if self.writer is not None:
            self.writer.release()
            log.info("Finished %s", self.current_file)
            wav = self.audio.stop_recording() if self.audio is not None else None
            if wav is not None and wav.exists():
                from .audio import mux_audio

                th = threading.Thread(target=mux_audio, name="mux",
                                      args=(self.current_file, wav, self.audio_bitrate_kbps))
                th.start()
                self._mux_threads.append(th)
        self.writer = None
        self.current_file = None

    def _loop(self):
        while self._running or not self.q.empty():
            try:
                frame, now, active, manual = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            cfg = self.cfg
            mode = "events" if cfg.mode == "off" else cfg.mode    # "off" still honours the REC button
            frame = self._prepare(frame, now)
            if self._manual and not manual and mode == "events" and self.writer is not None:
                self._close()                   # REC button released: stop now, no post-record
                self.last_active = 0.0
                active = False
            self._manual = manual
            if active:
                self.last_active = now
            if mode == "continuous":
                if self.writer and now - self.segment_started > cfg.segment_minutes * 60:
                    self._close()
                if not self.writer:
                    self._open(now)
                if self.writer:
                    self._write(frame, now)
                continue
            # events mode
            if self.writer is None:
                if active:
                    self._open(now)
                    if self.writer:
                        for t_pre, jpg in self.preroll:
                            self._write(cv2.imdecode(jpg, cv2.IMREAD_COLOR), t_pre)
                    self.preroll.clear()
                else:
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        self.preroll.append((now, jpg))
                    continue
            if self.writer:
                self._write(frame, now)
                too_long = now - self.segment_started > cfg.segment_minutes * 60
                if now - self.last_active > cfg.post_record_s or too_long:
                    self._close()
        self._close()

    def close(self):
        self._running = False
        self._thread.join(timeout=10)
        for th in self._mux_threads:            # let the last file get its audio
            th.join(timeout=60)


# ------------------------------------------------------------------ retention

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv"}


def _dir_files(root: Path, suffixes) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]


def enforce_retention(capture: CaptureConfig, rec: RecordingConfig, protect: Path | None = None) -> int:
    """Delete old snapshots/recordings. Returns number of files removed."""
    removed = 0
    now = time.time()
    for root, days, suffixes in ((Path(capture.output_dir), capture.retention_days, {".jpg"}),
                                 (Path(rec.output_dir), rec.retention_days, VIDEO_SUFFIXES | {".wav"})):
        if not root.exists() or days <= 0:
            continue
        for p in _dir_files(root, suffixes):
            if p != protect and now - p.stat().st_mtime > days * 86400:
                p.unlink(missing_ok=True)
                removed += 1
    root = Path(rec.output_dir)
    if rec.max_storage_gb > 0 and root.exists():
        files = sorted(_dir_files(root, VIDEO_SUFFIXES | {".wav"}), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        limit = rec.max_storage_gb * 1024 ** 3
        for p in files:
            if total <= limit:
                break
            if p == protect:
                continue
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
            removed += 1
    # Remove now-empty day folders.
    for root in (Path(capture.output_dir), root):
        if root.exists():
            for d in sorted(root.iterdir(), reverse=True):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
    if removed:
        log.info("Retention: removed %d old files", removed)
    return removed


_usage_cache: dict = {}


def disk_usage(path: str, max_age_s: float = 60.0) -> dict:
    """Space used by `path` and free on its drive. Walking thousands of recordings
    takes a while, and the dashboard asks every second: cached for a minute."""
    hit = _usage_cache.get(str(path))
    if hit and time.time() - hit[0] < max_age_s:
        return hit[1]
    result = _disk_usage(path)
    _usage_cache[str(path)] = (time.time(), result)
    return result


def _disk_usage(path: str) -> dict:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    used = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    total, _, free = shutil.disk_usage(p)
    return {"used_gb": round(used / 1024 ** 3, 2), "free_gb": round(free / 1024 ** 3, 1),
            "disk_gb": round(total / 1024 ** 3, 1)}

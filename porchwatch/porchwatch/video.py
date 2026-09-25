"""Efficient video encoding (H.264 / H.265) by piping frames into ffmpeg.

Much smaller files than OpenCV's mp4v/MJPG at the same quality, adjustable
with a single "compression" (CRF / CQ) value, and optionally encoded on an
NVIDIA GPU (NVENC) so it costs almost no CPU.
"""
from __future__ import annotations

import functools
import logging
import os
import subprocess
import threading
from collections import deque
from pathlib import Path

import numpy as np

from .audio import ffmpeg_exe

log = logging.getLogger(__name__)

FFMPEG_CODECS = ("h264", "h265")
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


@functools.lru_cache(maxsize=None)
def encoder_works(name: str) -> bool:
    """True if ffmpeg can actually encode with `name` here (NVENC needs a GPU + driver)."""
    exe = ffmpeg_exe()
    if exe is None:
        return False
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x144:d=0.2",
           "-frames:v", "3", "-c:v", name, "-f", "null", "-"]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=30, creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return res.returncode == 0


def pick_encoder(codec: str, encoder: str) -> str | None:
    """ffmpeg encoder name for codec h264/h265 and preference auto/nvidia/cpu."""
    gpu = {"h264": "h264_nvenc", "h265": "hevc_nvenc"}[codec]
    cpu = {"h264": "libx264", "h265": "libx265"}[codec]
    order = {"nvidia": [gpu, cpu], "cpu": [cpu], "auto": [gpu, cpu]}.get(encoder, [gpu, cpu])
    for name in order:
        if encoder_works(name):
            return name
    return None


def encoder_args(name: str, crf: int) -> list[str]:
    crf = int(min(max(crf, 0), 51))
    if name.endswith("_nvenc"):
        # Constant-quality VBR; p4 = balanced speed/quality preset.
        args = ["-c:v", name, "-preset", "p4", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    elif name == "libx265":
        args = ["-c:v", name, "-preset", "fast", "-crf", str(crf), "-x265-params", "log-level=error"]
    else:
        args = ["-c:v", name, "-preset", "veryfast", "-crf", str(crf)]
    if name in ("libx265", "hevc_nvenc"):
        args += ["-tag:v", "hvc1"]              # plays in Apple/Windows players
    return args + ["-pix_fmt", "yuv420p"]


class FFmpegWriter:
    """Drop-in for cv2.VideoWriter: write(bgr_frame), release(), isOpened()."""

    def __init__(self, path: Path, fps: int, size: tuple[int, int], encoder: str, crf: int):
        w, h = size
        self.path = Path(path)
        self.size = (w, h)
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               *encoder_args(encoder, crf),
               # Fragmented MP4: the file stays playable even if the PC crashes mid-recording.
               "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
               str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=_NO_WINDOW)
        self._ok = True
        # Always drain ffmpeg's messages: if nobody reads them the pipe fills up,
        # ffmpeg blocks, and so would the recorder.
        self._errors: deque = deque(maxlen=30)
        self._err_thread = threading.Thread(target=self._drain, name="ffmpeg-stderr", daemon=True)
        self._err_thread.start()

    def _drain(self) -> None:
        for line in iter(self.proc.stderr.readline, b""):
            self._errors.append(line.decode(errors="replace").rstrip())

    def isOpened(self) -> bool:
        return self._ok and self.proc.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if not self._ok:
            return
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            self._ok = False
            self._err_thread.join(timeout=2)
            log.error("Video encoder stopped: %s", " | ".join(self._errors)[-300:])

    def release(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._err_thread.join(timeout=2)
        if self.proc.returncode not in (0, None) and self._errors:
            log.warning("Video encoder: %s", " | ".join(self._errors)[-300:])

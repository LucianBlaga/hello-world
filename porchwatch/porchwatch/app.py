"""Main loop: frames -> detection -> tracking controller -> recording / dashboard."""
from __future__ import annotations

import logging
import queue
import threading
import time

import cv2

from .camera import FrameSource, NullPTZ, PTZState, make_ptz
from .config import Config, config_to_dict, update_config
from .controller import Controller, annotate
from .storage import Recorder, Storage, enforce_retention

log = logging.getLogger(__name__)


class App:
    def __init__(self, cfg: Config, config_path: str, video: str | None = None,
                 detector_factory=None, face_factory=None, plate_factory=None,
                 no_preview: bool = False, no_ptz: bool = False):
        self.cfg = cfg
        self.no_preview = no_preview
        self.no_ptz = no_ptz
        self.config_path = config_path
        self.video = video
        self._detector_factory = detector_factory
        self._face_factory = face_factory
        self._plate_factory = plate_factory
        self._detector = None
        self._detector_key = None
        self._faces = None
        self._plates = None

        self.cfg_lock = threading.Lock()
        self.latest_jpeg: bytes | None = None       # annotated, for the live view
        self.latest_raw_jpeg: bytes | None = None   # clean, for drawing zones
        self.status: dict = {"state": "starting"}
        self.fps = 0.0
        self.error: str | None = None
        self.paused = False
        self.storage: Storage | None = None
        self.recorder: Recorder | None = None
        self.controller: Controller | None = None
        self.ptz = None
        self._restart = False
        self._quit = False
        self._ptz_cmds: queue.Queue = queue.Queue()

    # ------------------------------------------------------------ control API (web thread)
    def apply_config(self, new_cfg: Config, restart: bool) -> None:
        with self.cfg_lock:
            # Update in place so every component sees the new values immediately.
            update_config(self.cfg, config_to_dict(new_cfg))
        if restart:
            log.info("Settings changed that need the camera/recorder reopened - restarting pipeline")
            self._restart = True

    def manual_ptz(self, action: str) -> bool:
        valid = {"left", "right", "up", "down", "zoom_in", "zoom_out", "home", "set_home"}
        if action not in valid:
            return False
        self._ptz_cmds.put(action)
        return True

    def stop(self) -> None:
        self._quit = True

    # ------------------------------------------------------------ lazy model loading
    def _get_detector(self):
        d = self.cfg.detection
        key = (d.model, d.device, d.imgsz)
        if self._detector is None or key != self._detector_key:
            if self._detector_factory:
                self._detector = self._detector_factory(d)
            else:
                from .detectors import ObjectDetector
                log.info("Loading detection model %s", d.model)
                self._detector = ObjectDetector(d)
            self._detector_key = key
        else:
            self._detector.cfg = d
        return self._detector

    def _get_faces(self):
        if self._faces is None:
            from .detectors import FaceDetector
            self._faces = (self._face_factory or FaceDetector)()
        return self._faces

    def _get_plates(self):
        if self._plates is None:
            from .detectors import PlateReader
            self._plates = (self._plate_factory or PlateReader)()
        return self._plates

    # ------------------------------------------------------------ main loop
    def run(self) -> None:
        from .web import start_web

        start_web(self)
        threading.Thread(target=self._retention_loop, name="retention", daemon=True).start()
        try:
            while not self._quit:
                self._restart = False
                try:
                    self._run_pipeline()
                    self.error = None
                except Exception as exc:
                    log.exception("Pipeline error")
                    self.error = str(exc)
                    self.status = {"state": "error"}
                    if self.video:
                        raise
                    time.sleep(3)       # e.g. camera unplugged: retry
                if self.video and not self._restart:
                    break               # finished the test video
        finally:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    def _retention_loop(self):
        while not self._quit:
            try:
                rec = self.recorder
                enforce_retention(self.cfg.capture, self.cfg.recording,
                                  protect=rec.current_file if rec else None)
            except Exception:
                log.exception("Retention clean-up failed")
            time.sleep(600)

    def _handle_manual(self, ptz, ctl):
        while not self._ptz_cmds.empty():
            action = self._ptz_cmds.get_nowait()
            st = ptz.state
            hfov, vfov = ptz.fov()
            if action == "home":
                ctl.reset()
                ptz.home()
                self.paused = False
                continue
            if action == "set_home":
                with self.cfg_lock:
                    c = self.cfg.camera
                    c.home_pan, c.home_tilt, c.home_zoom = round(st.pan, 1), round(st.tilt, 1), round(st.zoom, 2)
                from .config import save_config
                save_config(self.cfg, self.config_path)
                self.storage.log(f"Home set to pan {st.pan:.1f}, tilt {st.tilt:.1f}, zoom {st.zoom:.1f}x")
                continue
            # Any manual move pauses auto-tracking until "home" is pressed.
            self.paused = True
            ctl.reset()
            new = PTZState(st.pan, st.tilt, st.zoom)
            if action == "left":
                new.pan -= hfov * 0.15
            elif action == "right":
                new.pan += hfov * 0.15
            elif action == "up":
                new.tilt += vfov * 0.15
            elif action == "down":
                new.tilt -= vfov * 0.15
            elif action == "zoom_in":
                new.zoom *= 1.25
            elif action == "zoom_out":
                new.zoom /= 1.25
            ptz.move(new, force=True)

    def _run_pipeline(self) -> None:
        cfg = self.cfg
        self.storage = self.storage or Storage(cfg.capture)
        if self.storage.root.as_posix() != cfg.capture.output_dir:
            self.storage = Storage(cfg.capture)
        self.storage.cfg = cfg.capture
        detector = self._get_detector()
        faces, plates = self._get_faces(), self._get_plates()

        source = FrameSource(cfg.camera, self.video)
        recorder = None
        try:
            ptz = NullPTZ(cfg.camera) if self.no_ptz else make_ptz(cfg.camera, source)
            self.ptz = ptz
            ctl = Controller(cfg, ptz, faces, plates, self.storage,
                             ptz_enabled=not isinstance(ptz, NullPTZ))
            self.controller = ctl
            if ctl.ptz_enabled:
                ptz.home()
            recorder = Recorder(cfg.recording)
            self.recorder = recorder
            self.storage.log(f"Started ({'video file' if self.video else 'camera'}, "
                             f"PTZ {'on' if ctl.ptz_enabled else 'off'}, recording {cfg.recording.mode})")

            seq, t_prev, fps = 0, time.time(), 0.0
            next_stream = 0.0
            while not self._quit and not self._restart:
                seq_new, stamp, frame = source.next(seq)
                if frame is None or seq_new == seq:
                    if not source.running:
                        if not self.video:
                            raise RuntimeError("Camera disconnected")
                        break
                    continue
                seq = seq_new
                now = time.time()
                self._handle_manual(ptz, ctl)

                with self.cfg_lock:
                    detector = self._get_detector()
                    dets = detector(frame)
                    if self.paused:
                        status = ctl.status(now)
                        status["active"] = bool(dets)
                        status["state"] = "paused (manual)"
                    else:
                        status = ctl.step(frame, dets, now)
                self.status = status
                recorder.feed(frame, now, status["active"])

                dt = now - t_prev
                t_prev = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
                self.fps = fps

                show = cfg.show_preview and not self.no_preview
                if now >= next_stream or show:
                    vis = annotate(frame, dets, ctl, fps)
                    if now >= next_stream:
                        next_stream = now + 1.0 / max(1, cfg.web.stream_fps)
                        sw = cfg.web.stream_width
                        scale = sw / frame.shape[1]
                        small = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                        raw = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                        self.latest_jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                        self.latest_raw_jpeg = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                    if show:
                        if not self._show(vis, ptz, ctl):
                            self._quit = True
        finally:
            if recorder:
                recorder.close()
            source.close()

    def _show(self, vis, ptz, ctl) -> bool:
        try:
            h, w = vis.shape[:2]
            if w > 1280:
                vis = cv2.resize(vis, (1280, int(h * 1280 / w)))
            cv2.imshow("PorchWatch", vis)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error:
            log.warning("No GUI available; disabling local preview window (use the web dashboard)")
            self.no_preview = True
            return True
        if key in (ord("q"), 27):
            return False
        keymap = {ord("h"): "home", ord("a"): "left", ord("d"): "right", ord("w"): "up",
                  ord("s"): "down", ord("+"): "zoom_in", ord("="): "zoom_in", ord("-"): "zoom_out"}
        if key in keymap:
            self.manual_ptz(keymap[key])
        elif key == ord("p"):
            self.paused = not self.paused
        return True

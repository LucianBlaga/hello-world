"""Command line entry point.

    python -m porchwatch                 # run (camera + dashboard)
    python -m porchwatch --video street.mp4 --no-ptz
    python -m porchwatch probe           # show camera controls / formats
    python -m porchwatch test-ptz        # move the gimbal through a short routine
"""
from __future__ import annotations

import argparse
import logging
import time

from .config import load_config, save_config
from pathlib import Path


def _test_ptz(cfg) -> None:
    from .camera import FrameSource, PTZState, make_ptz

    src = FrameSource(cfg.camera)
    try:
        ptz = make_ptz(cfg.camera, src)
        steps = [
            ("home", PTZState(cfg.camera.home_pan, cfg.camera.home_tilt, 1.0)),
            ("pan right 30", PTZState(30, 0, 1.0)),
            ("pan left 30", PTZState(-30, 0, 1.0)),
            ("tilt up 20", PTZState(0, 20, 1.0)),
            ("tilt down 20", PTZState(0, -20, 1.0)),
            ("zoom 2x", PTZState(0, 0, 2.0)),
            ("zoom 4x", PTZState(0, 0, 4.0)),
            ("home", PTZState(cfg.camera.home_pan, cfg.camera.home_tilt, 1.0)),
        ]
        for name, st in steps:
            print(f"{name:14s} raw={ptz.to_raw(ptz.clamp(st))}")
            ptz.move(st, force=True)
            time.sleep(2.0)
        print("If it moved the wrong way, set camera.invert_pan / invert_tilt in Settings.")
    finally:
        src.close()


def main() -> None:
    ap = argparse.ArgumentParser(prog="porchwatch", description="OBSBOT Tiny 2 auto-tracking security camera")
    ap.add_argument("command", nargs="?", default="run", choices=["run", "probe", "test-ptz"])
    ap.add_argument("-c", "--config", default="config.yaml", help="settings file (created if missing)")
    ap.add_argument("--video", help="run on a video file instead of the camera (testing)")
    ap.add_argument("--no-preview", action="store_true", help="no local window (use the web dashboard)")
    ap.add_argument("--no-ptz", action="store_true", help="don't move the camera (this run only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    cfg = load_config(args.config)
    if not Path(args.config).exists():
        save_config(cfg, args.config)
        logging.info("Created default settings file %s", args.config)

    if args.command == "probe":
        from .camera import probe
        print(probe(cfg.camera))
        return
    if args.command == "test-ptz":
        _test_ptz(cfg)
        return

    from .app import App

    app = App(cfg, args.config, video=args.video, no_preview=args.no_preview, no_ptz=args.no_ptz)
    try:
        app.run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()

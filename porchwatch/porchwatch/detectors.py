"""Object, face and licence-plate detectors.

* People & vehicles : Ultralytics YOLO (COCO classes)
* Faces             : OpenCV YuNet (auto-downloaded), Haar cascade fallback
* Licence plates    : fast-alpr (plate detector + OCR, ONNX, auto-downloaded)
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DetectionConfig

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

PERSON = "person"
VEHICLE = "vehicle"
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    kind: str                   # PERSON or VEHICLE
    box: tuple                  # x1, y1, x2, y2 in pixels
    conf: float
    label: str = ""

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def _in_zone(cx: float, cy: float, zone) -> bool:
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2


def filter_zones(dets: list[Detection], w: int, h: int, cfg: DetectionConfig) -> list[Detection]:
    out = []
    for d in dets:
        cx, cy = d.center[0] / w, d.center[1] / h
        if not _in_zone(cx, cy, cfg.watch_zone):
            continue
        if any(_in_zone(cx, cy, z) for z in cfg.ignore_zones):
            continue
        out.append(d)
    return out


def model_path(name: str) -> Path:
    """Where a detection model lives: models/<name>, so downloads don't land next to
    the code. A bare name already downloaded to the working folder (older versions)
    is moved there instead of being downloaded again."""
    p = Path(name)
    if p.is_absolute() or p.parent != Path("."):
        return p                        # an explicit path chosen by the user
    target = MODELS_DIR / p.name
    if not target.exists() and p.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(target))
        log.info("Moved %s into %s", p.name, MODELS_DIR)
    return target


def tidy_model_files(folder: Path | None = None) -> None:
    """Older versions let Ultralytics download YOLO weights into the working folder:
    move them into models/ and remove stale partial downloads."""
    folder = folder or Path.cwd()
    for f in folder.glob("yolo*.pt"):
        target = MODELS_DIR / f.name
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            if target.exists():
                f.unlink()
            else:
                shutil.move(str(f), str(target))
        except OSError as exc:
            log.warning("Couldn't tidy %s: %s", f.name, exc)
    for f in folder.glob("yolo*.part"):
        try:
            if time.time() - f.stat().st_mtime > 600:       # not a download in progress
                f.unlink()
        except OSError:
            pass


def engine_size(imgsz: int, frame_shape) -> tuple[int, int]:
    """(h, w) for a TensorRT engine: the long side = imgsz (capped at the frame),
    the short side following the frame's shape, both multiples of 32. A 16:9
    engine skips the black bars a square one would process."""
    fh, fw = frame_shape[:2]
    long_side = min(int(imgsz), (max(fh, fw) + 31) // 32 * 32)
    short = int(np.ceil(long_side * min(fh, fw) / max(fh, fw) / 32) * 32)
    return (short, long_side) if fw >= fh else (long_side, short)


def engine_path(model: str, size: tuple, fp16: bool, trt_version: str) -> Path:
    """Engines only work for the model, input size, precision and TensorRT version
    they were built with: all of that goes in the name, so a change rebuilds."""
    stem = Path(model).stem
    return MODELS_DIR / f"{stem}_{size[1]}x{size[0]}_{'fp16' if fp16 else 'fp32'}_trt{trt_version}.engine"


class ObjectDetector:
    def __init__(self, cfg: DetectionConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        tidy_model_files()
        # Ultralytics downloads a missing model to exactly this path.
        self.model = YOLO(str(model_path(cfg.model)))
        self.precision: dict = {}
        self.cuda = False
        if cfg.device not in ("cpu", "mps"):
            try:
                import torch
                self.cuda = torch.cuda.is_available()
            except ImportError:
                pass
        if cfg.fp16 and self.cuda:
            self.precision = {"quantize": 16}           # FP16 (older ultralytics: half=True)
        # TensorRT: built once in the background; PyTorch is used until it's ready.
        self.engine = None                              # YOLO model running a TensorRT engine
        self.engine_imgsz = None                        # (h, w) the engine was built for
        self._engine_thread: threading.Thread | None = None
        self.engine_status = "off"

    def _start_engine_build(self, frame_shape) -> None:
        cfg = self.cfg
        if not self.cuda:
            self.engine_status = "needs an NVIDIA GPU"
            log.warning("TensorRT needs an NVIDIA GPU with CUDA; using PyTorch")
            return
        try:
            import tensorrt
        except ImportError:
            self.engine_status = "not installed (python -m pip install tensorrt)"
            log.warning("TensorRT not installed: python -m pip install tensorrt  (using PyTorch)")
            return
        h, w = engine_size(int(cfg.imgsz), frame_shape)
        fp16 = bool(self.precision)
        path = engine_path(cfg.model, (h, w), fp16, tensorrt.__version__)
        self.engine_status = "building"
        self._engine_thread = threading.Thread(target=self._build_engine, name="tensorrt",
                                               args=(path, (h, w), fp16), daemon=True)
        self._engine_thread.start()

    def _build_engine(self, path: Path, size: tuple, fp16: bool) -> None:
        from ultralytics import YOLO
        try:
            if not path.exists():
                log.info("Building TensorRT engine %s - one time, takes a few minutes", path.name)
                src = YOLO(str(model_path(self.cfg.model)))
                args = dict(format="engine", imgsz=list(size), device=0, verbose=False)
                try:
                    out = src.export(**args, **({"quantize": 16} if fp16 else {}))
                except SyntaxError:     # older ultralytics
                    out = src.export(**args, half=fp16)
                shutil.move(str(out), str(path))
            self.engine = YOLO(str(path), task="detect")
            self.engine_imgsz = size
            self.engine_status = f"on ({size[1]}x{size[0]}{', FP16' if fp16 else ''})"
            log.info("TensorRT engine ready: %s", path.name)
        except Exception as exc:
            self.engine_status = f"failed: {exc}"[:200]
            log.error("TensorRT engine build failed, staying on PyTorch: %s", exc)

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        cfg = self.cfg
        classes = [0, *COCO_VEHICLES]
        if cfg.tensorrt and self._engine_thread is None and self.engine_status == "off":
            self._start_engine_build(frame.shape)
        common = dict(conf=min(cfg.person_conf, cfg.vehicle_conf), classes=classes, verbose=False)
        if self.engine is not None:
            # The engine has a fixed input size and its precision is built in.
            res = self.engine.predict(frame, imgsz=list(self.engine_imgsz), device=cfg.device or 0, **common)[0]
            return self._to_detections(res, frame)
        # No point in upscaling: cap at the frame's long side (multiple of 32).
        imgsz = min(int(cfg.imgsz), (max(frame.shape[:2]) + 31) // 32 * 32)
        kwargs = dict(imgsz=imgsz, device=cfg.device or None, **common, **self.precision)
        try:
            res = self.model.predict(frame, **kwargs)[0]
        except SyntaxError:             # ultralytics too old for `quantize`
            if self.precision.get("quantize") != 16:
                raise
            log.info("Using half=True for FP16 (older ultralytics)")
            self.precision = {"half": True}
            kwargs.pop("quantize")
            res = self.model.predict(frame, **kwargs, half=True)[0]
        return self._to_detections(res, frame)

    def _to_detections(self, res, frame) -> list[Detection]:
        cfg = self.cfg
        dets = []
        for box, conf, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                  res.boxes.cls.tolist()):
            cls = int(cls)
            if cls == 0 and conf >= cfg.person_conf:
                dets.append(Detection(PERSON, tuple(box), conf, "person"))
            elif cls in COCO_VEHICLES and conf >= cfg.vehicle_conf:
                dets.append(Detection(VEHICLE, tuple(box), conf, COCO_VEHICLES[cls]))
        h, w = frame.shape[:2]
        return filter_zones(dets, w, h, cfg)


# ------------------------------------------------------------------ faces

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_FILE,
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/" + YUNET_FILE,
]


def _download(urls: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    errors = []
    for url in urls:
        try:
            # A timeout, so a stalled connection can't hang start-up forever.
            with urllib.request.urlopen(url, timeout=30) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            tmp.replace(path)
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("; ".join(errors))


class FaceDetector:
    """Finds faces (not identities) so we can crop and save the best shot."""

    def __init__(self):
        self.yunet = None
        self.haar = None
        path = MODELS_DIR / YUNET_FILE
        try:
            if not path.exists():
                log.info("Downloading YuNet face model...")
                _download(YUNET_URLS, path)
            self.yunet = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.7, 0.3, 50)
            return
        except Exception as exc:
            log.warning("YuNet face model unavailable (%s). Download %s manually into %s",
                        exc, YUNET_URLS[0], MODELS_DIR)
        # Haar cascades ship with OpenCV 4.x only.
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            self.haar = cv2.CascadeClassifier(
                str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
            log.warning("Using the (less accurate) Haar face detector")
        else:
            log.error("No face detector available: people will be tracked and saved without face close-ups")

    def __call__(self, image: np.ndarray) -> list[tuple[tuple, float]]:
        """Returns [((x1, y1, x2, y2), score)] in `image` coordinates."""
        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return []
        if self.yunet is None and self.haar is None:
            return []
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(image)
            if faces is None:
                return []
            return [((f[0], f[1], f[0] + f[2], f[1] + f[3]), float(f[-1])) for f in faces]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        return [((x, y, x + fw, y + fh), 1.0) for x, y, fw, fh in found]


# ------------------------------------------------------------------ plates


@dataclass
class PlateRead:
    text: str
    conf: float
    box: tuple      # in the coordinates of the image passed in


class PlateReader:
    def __init__(self):
        self.alpr = None
        try:
            from fast_alpr import ALPR

            self.alpr = ALPR(
                detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="cct-xs-v2-global-model",
            )
        except Exception as exc:
            log.warning("Plate reading disabled (%s). pip install fast-alpr[onnx]", exc)

    @property
    def available(self) -> bool:
        return self.alpr is not None

    def __call__(self, image: np.ndarray) -> list[PlateRead]:
        if self.alpr is None:
            return []
        reads = []
        for r in self.alpr.predict(image):
            if r.ocr is None or not r.ocr.text:
                continue
            conf = r.ocr.confidence
            if isinstance(conf, (list, tuple, np.ndarray)):
                conf = float(np.mean(conf)) if len(conf) else 0.0
            bb = r.detection.bounding_box
            text = "".join(ch for ch in r.ocr.text.upper() if ch.isalnum())
            reads.append(PlateRead(text, float(conf), (bb.x1, bb.y1, bb.x2, bb.y2)))
        return reads


def sharpness(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

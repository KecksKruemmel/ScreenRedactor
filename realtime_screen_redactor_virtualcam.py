from __future__ import annotations

import difflib
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import customtkinter as ctk
import mss
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
from rapidocr import EngineType, ModelType, RapidOCR

Box = Tuple[int, int, int, int]

MODEL_TIERS = {
    "tiny": ModelType.TINY,
    "small": ModelType.SMALL,
    "medium": ModelType.MEDIUM,
}

SECRET_WORDS = (
    "password", "passwort", "kennwort", "passwd", "pwd",
    "token", "tocken", "tokken", "t0ken",
    "secret", "apikey", "api_key", "bearer", "authorization",
    "auth", "session", "cookie", "credential", "credentials",
)


def fold_word(word: str) -> str:
    return (
        word.replace("0", "o")
        .replace("1", "l")
        .replace("3", "e")
        .replace("4", "a")
        .replace("5", "s")
        .replace("7", "t")
        .replace("_", "")
    )


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


@dataclass(frozen=True)
class Config:
    monitor: int = 1
    fps: int = 30
    ocr_every: int = 2
    ocr_max_width: int = 960
    min_conf: float = 0.3
    ttl: float = 12.0
    lang: str = "latin"
    model_tier: str = "small"
    det_thresh: float = 0.3
    box_thresh: float = 0.4
    unclip_ratio: float = 1.6
    onnx_threads: int = 4
    mode: str = "black"
    padding: int = 12
    use_dml: bool = True
    track_margin: int = 80
    track_thresh: float = 0.6
    track_max_extend: float = 15.0
    change_threshold: float = 3.0
    change_sample_width: int = 160
    change_max_interval: float = 5.0


@dataclass
class Region:
    box: Box
    updated: float
    created: float
    template: Optional[np.ndarray] = None


class TextMatcher:
    def __init__(self) -> None:
        f = re.IGNORECASE
        self.email = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", f)
        self.secret_word = re.compile(
            r"\b(password|passwort|kennwort|passwd|pwd|token|tocken|tokken|t0ken|secret|"
            r"api[_ -]?key|apikey|access[_ -]?key|private[_ -]?key|bearer|"
            r"authorization|auth|client[_ -]?secret|refresh[_ -]?token|"
            r"session|cookie|credential|credentials)\b",
            f,
        )
        self.api_key = re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{15,}|sk-proj-[A-Za-z0-9_-]{15,}|"
            r"pk_[A-Za-z0-9_-]{15,}|xox[baprs]-[A-Za-z0-9-]{15,}|"
            r"gh[pousr]_[A-Za-z0-9_]{15,}|github_pat_[A-Za-z0-9_]{15,}|"
            r"AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{25,}|"
            r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"
        )
        self.ip = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")

    def is_secret(self, text: str) -> bool:
        cleaned = " ".join(text.split())
        if len(cleaned) < 3:
            return False
        if self.email.search(cleaned) or self.secret_word.search(cleaned):
            return True
        if self.api_key.search(cleaned) or self.ip.search(cleaned):
            return True
        return self._fuzzy(cleaned)

    def classify(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if self.email.search(cleaned):
            return "E-Mail"
        if self.api_key.search(cleaned):
            return "API-Schluessel"
        if self.ip.search(cleaned):
            return "IP-Adresse"
        if self.secret_word.search(cleaned) or self._fuzzy(cleaned):
            return "Passwort/Token"
        return "Sensibler Text"

    def _fuzzy(self, text: str) -> bool:
        words = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        for w in words:
            folded = fold_word(w)
            for target in SECRET_WORDS:
                if folded == target:
                    return True
                if len(folded) >= 5 and difflib.SequenceMatcher(None, folded, target).ratio() >= 0.78:
                    return True
        return False


class ScreenCapture:
    def __init__(self, monitor_index: int) -> None:
        self.sct = mss.mss()
        idx = monitor_index if 0 <= monitor_index < len(self.sct.monitors) else 1
        self.monitor = self.sct.monitors[idx]
        self.width = int(self.monitor["width"])
        self.height = int(self.monitor["height"])

    def grab(self) -> np.ndarray:
        raw = np.asarray(self.sct.grab(self.monitor), dtype=np.uint8)
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)

    def close(self) -> None:
        self.sct.close()


class ChangeDetector:
    def __init__(self, threshold: float, sample_width: int, max_interval: float) -> None:
        self.threshold = threshold
        self.sample_width = sample_width
        self.max_interval = max_interval
        self.last_gray: Optional[np.ndarray] = None
        self.last_submit = 0.0

    def should_submit(self, frame_rgb: np.ndarray) -> bool:
        h, w = frame_rgb.shape[:2]
        scale = self.sample_width / float(w)
        small = cv2.resize(frame_rgb, (self.sample_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        now = time.perf_counter()
        changed = self.last_gray is None or float(cv2.absdiff(gray, self.last_gray).mean()) >= self.threshold
        stale = (now - self.last_submit) >= self.max_interval
        self.last_gray = gray
        if changed or stale:
            self.last_submit = now
            return True
        return False


class StreamerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = "gestoppt"
        self.fps = 0.0
        self.texts = 0
        self.hits_now = 0
        self.hits_total = 0
        self.error = ""
        self.log: deque = deque(maxlen=300)
        self.flip_h = False
        self.flip_v = False

    def set_status(self, status: str) -> None:
        with self.lock:
            self.status = status

    def set_error(self, message: str) -> None:
        with self.lock:
            self.status = "fehler"
            self.error = message

    def update_stats(self, fps: float, texts: int, hits_now: int) -> None:
        with self.lock:
            self.fps = fps
            self.texts = texts
            self.hits_now = hits_now

    def set_flip(self, flip_h: bool, flip_v: bool) -> None:
        with self.lock:
            self.flip_h = flip_h
            self.flip_v = flip_v

    def get_flip(self) -> Tuple[bool, bool]:
        with self.lock:
            return self.flip_h, self.flip_v

    def add_log(self, label: str) -> None:
        with self.lock:
            self.hits_total += 1
            self.log.append((time.strftime("%H:%M:%S"), label))

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "fps": self.fps,
                "texts": self.texts,
                "hits_now": self.hits_now,
                "hits_total": self.hits_total,
                "error": self.error,
                "log": list(self.log),
            }


class OCRWorker:
    def __init__(self, cfg: Config, matcher: TextMatcher, state: StreamerState) -> None:
        self.cfg = cfg
        self.matcher = matcher
        self.state = state
        self.input: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.regions: List[Region] = []
        self.prev_boxes: List[Box] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.ready = False
        self.last_total_texts = 0
        self.last_detect_ms = 0.0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def submit(self, frame: np.ndarray) -> None:
        if self.input.full():
            try:
                self.input.get_nowait()
            except queue.Empty:
                pass
        try:
            self.input.put_nowait(frame.copy())
        except queue.Full:
            pass

    def active(self, frame_rgb: Optional[np.ndarray] = None) -> List[Region]:
        now = time.perf_counter()
        with self.lock:
            self.regions = [r for r in self.regions if now - r.updated <= self.cfg.ttl]
            if frame_rgb is None or not self.regions:
                return self.regions

            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape[:2]
            tracked: List[Region] = []
            for r in self.regions:
                x1, y1, x2, y2 = r.box
                bw, bh = x2 - x1, y2 - y1
                if r.template is None or bw < 6 or bh < 6:
                    tracked.append(r)
                    continue

                m = self.cfg.track_margin
                sx1, sy1 = max(0, x1 - m), max(0, y1 - m)
                sx2, sy2 = min(w, x2 + m), min(h, y2 + m)
                search = gray[sy1:sy2, sx1:sx2]
                if search.shape[0] < bh or search.shape[1] < bw:
                    tracked.append(r)
                    continue

                res = cv2.matchTemplate(search, r.template, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                if score < self.cfg.track_thresh:
                    tracked.append(r)
                    continue

                nx1, ny1 = sx1 + loc[0], sy1 + loc[1]
                nx2, ny2 = nx1 + bw, ny1 + bh
                new_template = gray[ny1:ny2, nx1:nx2].copy()
                within_budget = (now - r.created) <= self.cfg.track_max_extend
                new_updated = now if within_budget else r.updated
                tracked.append(Region((nx1, ny1, nx2, ny2), new_updated, r.created, new_template))

            self.regions = tracked
            return tracked

    def _run(self) -> None:
        tier = MODEL_TIERS.get(self.cfg.model_tier, ModelType.SMALL)
        params = {
            "Global.log_level": "error",
            "Global.use_det": True,
            "Global.use_cls": False,
            "Global.use_rec": True,
            "Global.min_side_len": 24,
            "Global.max_side_len": self.cfg.ocr_max_width,
            "Global.return_word_box": False,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": self.cfg.lang,
            "Det.model_type": tier,
            "Det.limit_side_len": self.cfg.ocr_max_width,
            "Det.limit_type": "max",
            "Det.thresh": self.cfg.det_thresh,
            "Det.box_thresh": self.cfg.box_thresh,
            "Det.unclip_ratio": self.cfg.unclip_ratio,
            "Det.score_mode": "fast",
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": self.cfg.lang,
            "Rec.model_type": tier,
            "Rec.rec_img_shape": [3, 48, 160],
            "Rec.rec_batch_num": 6,
            "EngineConfig.onnxruntime.intra_op_num_threads": self.cfg.onnx_threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
            "EngineConfig.onnxruntime.cpu_ep_cfg.arena_extend_strategy": "kSameAsRequested",
            "EngineConfig.onnxruntime.use_dml": self.cfg.use_dml,
        }
        print(f"[OCR] Lade RapidOCR Modell '{self.cfg.model_tier}' (Sprache: {self.cfg.lang}, DirectML: {self.cfg.use_dml}) ...")
        try:
            engine = RapidOCR(params=params)
        except Exception as exc:
            print(f"[OCR] Laden mit DirectML fehlgeschlagen: {exc}")
            if self.cfg.use_dml:
                print("[OCR] Fallback auf CPU ohne DirectML ...")
                params["EngineConfig.onnxruntime.use_dml"] = False
                try:
                    engine = RapidOCR(params=params)
                except Exception as exc2:
                    print(f"[OCR] FEHLER beim Laden des Modells: {exc2}")
                    self.state.set_error(str(exc2))
                    return
            else:
                self.state.set_error(str(exc))
                return
        self.ready = True
        print("[OCR] Modell bereit. Erkennung laeuft.")
        while not self.stop_event.is_set():
            try:
                frame = self.input.get(timeout=0.1)
            except queue.Empty:
                continue
            regions = self._detect(engine, frame)
            with self.lock:
                self.regions = self._merge(self.regions, regions)

    def _merge(self, old_regions: List[Region], new_regions: List[Region]) -> List[Region]:
        now = time.perf_counter()
        matched: set = set()
        result: List[Region] = []
        for new in new_regions:
            best_i, best_iou = -1, 0.0
            for i, old in enumerate(old_regions):
                if i in matched:
                    continue
                ov = iou(old.box, new.box)
                if ov > best_iou:
                    best_i, best_iou = i, ov
            created = new.created
            if best_i != -1 and best_iou >= 0.15:
                matched.add(best_i)
                created = old_regions[best_i].created
            result.append(Region(new.box, new.updated, created, new.template))
        for i, old in enumerate(old_regions):
            if i not in matched and now - old.updated <= self.cfg.ttl:
                result.append(old)
        return result

    def _detect(self, engine: RapidOCR, frame: np.ndarray) -> List[Region]:
        original = frame
        h, w = frame.shape[:2]
        scale = min(1.0, self.cfg.ocr_max_width / float(w))
        ocr_frame = frame
        if scale < 1.0:
            ocr_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        t0 = time.perf_counter()
        try:
            out = engine(
                ocr_frame,
                text_score=self.cfg.min_conf,
                box_thresh=self.cfg.box_thresh,
                unclip_ratio=self.cfg.unclip_ratio,
            )
        except Exception as exc:
            self.state.set_error(str(exc))
            return []
        self.last_detect_ms = (time.perf_counter() - t0) * 1000.0
        boxes = getattr(out, "boxes", None)
        texts = getattr(out, "txts", None)
        if boxes is None or texts is None:
            self.last_total_texts = 0
            return []
        self.last_total_texts = len(texts)
        inv = 1.0 / scale
        now = time.perf_counter()
        pad = self.cfg.padding
        regions: List[Region] = []
        current_boxes: List[Box] = []
        for box, text in zip(boxes, texts):
            if not text or not self.matcher.is_secret(text):
                continue
            pts = np.asarray(box, dtype=np.float32)
            x1 = int(np.clip(np.min(pts[:, 0]) * inv, 0, w - 1)) - pad
            y1 = int(np.clip(np.min(pts[:, 1]) * inv, 0, h - 1)) - pad
            x2 = int(np.clip(np.max(pts[:, 0]) * inv, 0, w - 1)) + pad
            y2 = int(np.clip(np.max(pts[:, 1]) * inv, 0, h - 1)) + pad
            rect = (max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2))
            bx1, by1, bx2, by2 = rect
            crop = original[by1:by2, bx1:bx2]
            template = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).copy() if crop.size > 0 else None
            regions.append(Region(rect, now, now, template))
            current_boxes.append(rect)
            is_new = all(iou(rect, prev) < 0.3 for prev in self.prev_boxes)
            if is_new:
                self.state.add_log(self.matcher.classify(text))
        self.prev_boxes = current_boxes
        return regions


def redact(frame: np.ndarray, regions: Sequence[Region], mode: str) -> np.ndarray:
    for r in regions:
        x1, y1, x2, y2 = r.box
        if x2 <= x1 or y2 <= y1:
            continue
        if mode == "blur":
            frame[y1:y2, x1:x2] = cv2.blur(frame[y1:y2, x1:x2], (25, 25))
        else:
            frame[y1:y2, x1:x2] = 0
    return frame


class StreamThread(threading.Thread):
    def __init__(self, cfg: Config, state: StreamerState) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.state = state
        self.stop_event = threading.Event()

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        self.state.set_status("laedt")
        try:
            capture = ScreenCapture(self.cfg.monitor)
        except Exception as exc:
            print(f"[CAM] FEHLER: {exc}")
            self.state.set_error(str(exc))
            return
        print(f"[CAM] Monitor {self.cfg.monitor}: {capture.width}x{capture.height}, Ziel-FPS {self.cfg.fps}")
        matcher = TextMatcher()
        ocr = OCRWorker(self.cfg, matcher, self.state)
        detector = ChangeDetector(self.cfg.change_threshold, self.cfg.change_sample_width, self.cfg.change_max_interval)
        ocr.start()
        try:
            with pyvirtualcam.Camera(
                width=capture.width,
                height=capture.height,
                fps=self.cfg.fps,
                fmt=PixelFormat.RGB,
                print_fps=False,
            ) as cam:
                print(f"[CAM] Virtuelle Kamera aktiv: {cam.device}")
                i = 0
                frame_count = 0
                last_stat = time.perf_counter()
                while not self.stop_event.is_set():
                    frame = capture.grab()
                    flip_h, flip_v = self.state.get_flip()
                    if flip_h and flip_v:
                        frame = cv2.flip(frame, -1)
                    elif flip_h:
                        frame = cv2.flip(frame, 1)
                    elif flip_v:
                        frame = cv2.flip(frame, 0)
                    if i % self.cfg.ocr_every == 0 and detector.should_submit(frame):
                        ocr.submit(frame)
                    regions = ocr.active(frame)
                    redact(frame, regions, self.cfg.mode)
                    cam.send(np.ascontiguousarray(frame))
                    cam.sleep_until_next_frame()
                    i += 1
                    frame_count += 1
                    now = time.perf_counter()
                    elapsed = now - last_stat
                    if elapsed >= 1.0:
                        fps = frame_count / elapsed
                        self.state.update_stats(fps, ocr.last_total_texts, len(regions))
                        if ocr.ready and self.state.status != "fehler":
                            self.state.set_status("aktiv")
                        print(
                            f"[STAT] FPS={fps:.1f} erkannte_Textfelder={ocr.last_total_texts} "
                            f"zensiert_jetzt={len(regions)} OCR_Dauer={ocr.last_detect_ms:.0f}ms"
                        )
                        frame_count = 0
                        last_stat = now
        except Exception as exc:
            print(f"[CAM] FEHLER: {exc}")
            self.state.set_error(str(exc))
        finally:
            ocr.stop()
            capture.close()
            print("[DONE] Gestoppt.")
            if self.state.status != "fehler":
                self.state.set_status("gestoppt")


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ScreenRedactor")
        self.geometry("660x670")
        self.resizable(False, False)

        self.rstate = StreamerState()
        self.stream_thread: Optional[StreamThread] = None
        self.logged_count = 0

        self.monitor_var = ctk.StringVar(value="1")
        self.fps_var = ctk.StringVar(value="30")
        self.width_var = ctk.StringVar(value="960")
        self.every_var = ctk.StringVar(value="2")
        self.mode_var = ctk.StringVar(value="black")
        self.model_var = ctk.StringVar(value="small")
        self.lang_var = ctk.StringVar(value="latin")
        self.dml_var = ctk.BooleanVar(value=True)
        self.flip_h_var = ctk.BooleanVar(value=False)
        self.flip_v_var = ctk.BooleanVar(value=False)

        self._build_ui()
        self.after(400, self._refresh)

    def _build_ui(self) -> None:
        cfg_frame = ctk.CTkFrame(self)
        cfg_frame.pack(fill="x", padx=14, pady=14)

        ctk.CTkLabel(cfg_frame, text="Monitor").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ctk.CTkEntry(cfg_frame, textvariable=self.monitor_var, width=110).grid(row=0, column=1, padx=8, pady=6)
        ctk.CTkLabel(cfg_frame, text="FPS").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ctk.CTkEntry(cfg_frame, textvariable=self.fps_var, width=110).grid(row=1, column=1, padx=8, pady=6)
        ctk.CTkLabel(cfg_frame, text="OCR max. Breite").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ctk.CTkEntry(cfg_frame, textvariable=self.width_var, width=110).grid(row=2, column=1, padx=8, pady=6)
        ctk.CTkLabel(cfg_frame, text="OCR alle N Frames").grid(row=3, column=0, sticky="w", padx=8, pady=6)
        ctk.CTkEntry(cfg_frame, textvariable=self.every_var, width=110).grid(row=3, column=1, padx=8, pady=6)

        ctk.CTkLabel(cfg_frame, text="Modus").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ctk.CTkOptionMenu(cfg_frame, values=["black", "blur"], variable=self.mode_var, width=130).grid(row=0, column=3, padx=8, pady=6)
        ctk.CTkLabel(cfg_frame, text="Modell").grid(row=1, column=2, sticky="w", padx=8, pady=6)
        ctk.CTkOptionMenu(cfg_frame, values=["tiny", "small", "medium"], variable=self.model_var, width=130).grid(row=1, column=3, padx=8, pady=6)
        ctk.CTkLabel(cfg_frame, text="Sprache").grid(row=2, column=2, sticky="w", padx=8, pady=6)
        ctk.CTkEntry(cfg_frame, textvariable=self.lang_var, width=130).grid(row=2, column=3, padx=8, pady=6)
        ctk.CTkCheckBox(cfg_frame, text="GPU-Beschleunigung (DirectML)", variable=self.dml_var).grid(row=3, column=2, columnspan=2, sticky="w", padx=8, pady=6)

        flip_frame = ctk.CTkFrame(self)
        flip_frame.pack(fill="x", padx=14, pady=(0, 14))
        ctk.CTkLabel(flip_frame, text="Bildausrichtung:").pack(side="left", padx=(10, 6), pady=10)
        ctk.CTkSwitch(
            flip_frame, text="Horizontal spiegeln", variable=self.flip_h_var, command=self._on_flip_change
        ).pack(side="left", padx=10, pady=10)
        ctk.CTkSwitch(
            flip_frame, text="Vertikal spiegeln", variable=self.flip_v_var, command=self._on_flip_change
        ).pack(side="left", padx=10, pady=10)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(fill="x", padx=14, pady=(0, 14))
        self.start_btn = ctk.CTkButton(btn_frame, text="Start", command=self._start, fg_color="#2e8b57")
        self.start_btn.pack(side="left", padx=8, pady=10)
        self.stop_btn = ctk.CTkButton(btn_frame, text="Stop", command=self._stop, state="disabled", fg_color="#b33a3a")
        self.stop_btn.pack(side="left", padx=8, pady=10)
        self.status_label = ctk.CTkLabel(btn_frame, text="Status: gestoppt", text_color="gray")
        self.status_label.pack(side="left", padx=16)

        stat_frame = ctk.CTkFrame(self)
        stat_frame.pack(fill="x", padx=14, pady=(0, 14))
        self.fps_label = ctk.CTkLabel(stat_frame, text="FPS: -")
        self.fps_label.grid(row=0, column=0, padx=12, pady=8, sticky="w")
        self.texts_label = ctk.CTkLabel(stat_frame, text="Erkannte Textfelder: -")
        self.texts_label.grid(row=0, column=1, padx=12, pady=8, sticky="w")
        self.hits_now_label = ctk.CTkLabel(stat_frame, text="Aktuell zensiert: -")
        self.hits_now_label.grid(row=1, column=0, padx=12, pady=8, sticky="w")
        self.hits_total_label = ctk.CTkLabel(stat_frame, text="Gesamt zensiert: 0")
        self.hits_total_label.grid(row=1, column=1, padx=12, pady=8, sticky="w")

        ctk.CTkLabel(self, text="Zensur-Protokoll").pack(anchor="w", padx=16)
        self.log_box = ctk.CTkTextbox(self, height=230)
        self.log_box.pack(fill="both", expand=True, padx=14, pady=(6, 14))
        self.log_box.configure(state="disabled")

    def _on_flip_change(self) -> None:
        self.rstate.set_flip(self.flip_h_var.get(), self.flip_v_var.get())

    def _start(self) -> None:
        try:
            cfg = Config(
                monitor=int(self.monitor_var.get()),
                fps=int(self.fps_var.get()),
                ocr_every=max(1, int(self.every_var.get())),
                ocr_max_width=max(320, int(self.width_var.get())),
                lang=self.lang_var.get().strip() or "en",
                model_tier=self.model_var.get(),
                mode=self.mode_var.get(),
                use_dml=self.dml_var.get(),
            )
        except ValueError:
            self.status_label.configure(text="Status: ungueltige Einstellungen", text_color="orange")
            return

        self.rstate = StreamerState()
        self.rstate.set_flip(self.flip_h_var.get(), self.flip_v_var.get())
        self.logged_count = 0
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        self.stream_thread = StreamThread(cfg, self.rstate)
        self.stream_thread.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

    def _stop(self) -> None:
        if self.stream_thread is not None:
            self.stream_thread.request_stop()
            self.stream_thread.join(timeout=3.0)
            self.stream_thread = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def _refresh(self) -> None:
        snap = self.rstate.snapshot()
        colors = {"aktiv": "#33cc66", "laedt": "#e6b800", "fehler": "#e63946", "gestoppt": "gray"}
        color = colors.get(snap["status"], "gray")
        text = f"Status: {snap['status']}"
        if snap["error"]:
            text += f" ({snap['error']})"
        self.status_label.configure(text=text, text_color=color)
        self.fps_label.configure(text=f"FPS: {snap['fps']:.1f}")
        self.texts_label.configure(text=f"Erkannte Textfelder: {snap['texts']}")
        self.hits_now_label.configure(text=f"Aktuell zensiert: {snap['hits_now']}")
        self.hits_total_label.configure(text=f"Gesamt zensiert: {snap['hits_total']}")

        log = snap["log"]
        if len(log) > self.logged_count:
            self.log_box.configure(state="normal")
            for ts, label in log[self.logged_count:]:
                self.log_box.insert("end", f"{ts}  {label}\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
            self.logged_count = len(log)

        self.after(400, self._refresh)

    def on_close(self) -> None:
        self._stop()
        self.destroy()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
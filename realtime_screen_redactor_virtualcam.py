from __future__ import annotations

import argparse
import difflib
import os
import queue
import re
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from tkinter import ttk
from typing import List, Optional, Sequence, Tuple

import cv2
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


@dataclass(frozen=True)
class Config:
    monitor: int = 1
    fps: int = 30
    ocr_every: int = 4
    ocr_max_width: int = 1600
    min_conf: float = 0.3
    ttl: float = 12.0
    lang: str = "en"
    model_tier: str = "small"
    det_thresh: float = 0.3
    box_thresh: float = 0.4
    unclip_ratio: float = 1.6
    onnx_threads: int = 2
    mode: str = "black"
    padding: int = 12
    gui: bool = True
    log_file: str = "redactions.log"


@dataclass
class Region:
    box: Box
    updated: float
    label: str


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


class TextMatcher:
    def __init__(self) -> None:
        f = re.IGNORECASE
        self.email = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", f)
        self.keyword = re.compile(
            r"\b(password|passwort|kennwort|passwd|pwd|token|tocken|tokken|t0ken|secret|"
            r"api[_ -]?key|apikey|access[_ -]?key|private[_ -]?key|bearer|"
            r"authorization|auth|client[_ -]?secret|refresh[_ -]?token|"
            r"session|cookie|credential|credentials)\b",
            f,
        )
        self.apikey = re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{15,}|sk-proj-[A-Za-z0-9_-]{15,}|"
            r"pk_[A-Za-z0-9_-]{15,}|xox[baprs]-[A-Za-z0-9-]{15,}|"
            r"gh[pousr]_[A-Za-z0-9_]{15,}|github_pat_[A-Za-z0-9_]{15,}|"
            r"AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{25,}|"
            r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"
        )
        self.ipv4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")

    def label(self, text: str) -> Optional[str]:
        cleaned = " ".join(text.strip().split())
        if len(cleaned) < 3:
            return None
        if self.email.search(cleaned):
            return "email"
        if self.apikey.search(cleaned):
            return "api_key"
        if self.keyword.search(cleaned):
            return "secret_keyword"
        if self.ipv4.search(cleaned):
            return "ip_adresse"
        if self._fuzzy(cleaned):
            return "fuzzy_secret"
        return None

    def _fuzzy(self, text: str) -> bool:
        for word in re.findall(r"[A-Za-z0-9_]{3,}", text.lower()):
            folded = fold_word(word)
            for target in SECRET_WORDS:
                if folded == target:
                    return True
                if len(folded) >= 5 and difflib.SequenceMatcher(None, folded, target).ratio() >= 0.76:
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


class RedactionLog:
    def __init__(self, path: str) -> None:
        self.lock = threading.Lock()
        self.events: deque = deque(maxlen=300)
        self.totals = {}
        self.path = path
        self._last_logged = {}

    def record(self, label: str, count: int) -> None:
        now = time.time()
        last = self._last_logged.get(label, 0.0)
        if now - last < 2.0:
            return
        self._last_logged[label] = now
        with self.lock:
            self.events.append((now, label, count))
            self.totals[label] = self.totals.get(label, 0) + count
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
                fh.write(f"{stamp} label={label} anzahl={count}\n")
        except OSError:
            pass

    def snapshot(self):
        with self.lock:
            return list(self.events), dict(self.totals)


class OCRWorker:
    def __init__(self, cfg: Config, matcher: TextMatcher, log: RedactionLog) -> None:
        self.cfg = cfg
        self.matcher = matcher
        self.log = log
        self.input: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.regions: List[Region] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.ready = False
        self.load_error = ""
        self.passes = 0
        self.last_total_texts = 0
        self.last_hit_count = 0

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

    def active(self) -> List[Region]:
        now = time.perf_counter()
        with self.lock:
            self.regions = [r for r in self.regions if now - r.updated <= self.cfg.ttl]
            return self.regions

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
            "Det.lang_type": "ch",
            "Det.model_type": tier,
            "Det.limit_side_len": 640,
            "Det.limit_type": "min",
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
        }
        print(f"[OCR] Lade RapidOCR Modell '{self.cfg.model_tier}' (Sprache: {self.cfg.lang}) ...")
        try:
            engine = RapidOCR(params=params)
        except Exception as exc:
            self.load_error = str(exc)
            print(f"[OCR] FEHLER beim Laden des Modells: {exc}")
            return
        self.ready = True
        print("[OCR] Modell bereit. Erkennung laeuft.")
        while not self.stop_event.is_set():
            try:
                frame = self.input.get(timeout=0.1)
            except queue.Empty:
                continue
            self.passes += 1
            regions = self._detect(engine, frame)
            with self.lock:
                self.regions = regions

    def _detect(self, engine: RapidOCR, frame: np.ndarray) -> List[Region]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.cfg.ocr_max_width / float(w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        try:
            out = engine(
                frame,
                text_score=self.cfg.min_conf,
                box_thresh=self.cfg.box_thresh,
                unclip_ratio=self.cfg.unclip_ratio,
            )
        except Exception as exc:
            self.load_error = str(exc)
            return []
        boxes = getattr(out, "boxes", None)
        texts = getattr(out, "txts", None)
        if boxes is None or texts is None:
            self.last_total_texts = 0
            self.last_hit_count = 0
            return []
        self.last_total_texts = len(texts)
        inv = 1.0 / scale
        now = time.perf_counter()
        pad = self.cfg.padding
        regions: List[Region] = []
        counts = {}
        for box, text in zip(boxes, texts):
            if not text:
                continue
            label = self.matcher.label(text)
            if label is None:
                continue
            pts = np.asarray(box, dtype=np.float32)
            x1 = int(np.clip(np.min(pts[:, 0]) * inv, 0, w - 1)) - pad
            y1 = int(np.clip(np.min(pts[:, 1]) * inv, 0, h - 1)) - pad
            x2 = int(np.clip(np.max(pts[:, 0]) * inv, 0, w - 1)) + pad
            y2 = int(np.clip(np.max(pts[:, 1]) * inv, 0, h - 1)) + pad
            regions.append(Region((max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)), now, label))
            counts[label] = counts.get(label, 0) + 1
        for label, count in counts.items():
            self.log.record(label, count)
        self.last_hit_count = len(regions)
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


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.model_status = "LAEDT"
        self.texts_found = 0
        self.active_regions = 0
        self.error = ""

    def update(self, **kwargs) -> None:
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(
                fps=self.fps,
                width=self.width,
                height=self.height,
                model_status=self.model_status,
                texts_found=self.texts_found,
                active_regions=self.active_regions,
                error=self.error,
            )


class Pipeline(threading.Thread):
    def __init__(self, cfg: Config, stats: Stats, log: RedactionLog) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.stats = stats
        self.log = log
        self.mode = cfg.mode
        self.stop_event = threading.Event()
        self.matcher = TextMatcher()
        self.ocr = OCRWorker(cfg, self.matcher, log)

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        capture = ScreenCapture(self.cfg.monitor)
        self.stats.update(width=capture.width, height=capture.height)
        self.ocr.start()
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
                    if i % self.cfg.ocr_every == 0:
                        self.ocr.submit(frame)
                    regions = self.ocr.active()
                    redact(frame, regions, self.mode)
                    cam.send(np.ascontiguousarray(frame))
                    cam.sleep_until_next_frame()
                    i += 1
                    frame_count += 1
                    now = time.perf_counter()
                    if now - last_stat >= 1.0:
                        fps = frame_count / (now - last_stat)
                        status = "AKTIV" if self.ocr.ready else ("FEHLER" if self.ocr.load_error else "LAEDT")
                        self.stats.update(
                            fps=fps,
                            model_status=status,
                            texts_found=self.ocr.last_total_texts,
                            active_regions=len(regions),
                            error=self.ocr.load_error,
                        )
                        frame_count = 0
                        last_stat = now
        except Exception as exc:
            print(f"[CAM] FEHLER: {exc}")
            self.stats.update(model_status="FEHLER", error=str(exc))
        finally:
            self.ocr.stop()
            capture.close()
            print("[DONE] Pipeline beendet.")


class App:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.stats = Stats()
        self.log = RedactionLog(cfg.log_file)
        self.pipeline = Pipeline(cfg, self.stats, self.log)
        self.root = tk.Tk()
        self.root.title("ScreenRedactor Monitor")
        self.root.geometry("560x560")
        self.root.configure(bg="#1e1e1e")
        self._log_shown = 0
        self._build()
        self.pipeline.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(300, self._tick)

    def _build(self) -> None:
        fg = "#e8e8e8"
        bg = "#1e1e1e"
        panel = tk.Frame(self.root, bg=bg)
        panel.pack(fill="x", padx=12, pady=10)

        self.status_var = tk.StringVar(value="Status: LAEDT")
        self.res_var = tk.StringVar(value="Aufloesung: -")
        self.fps_var = tk.StringVar(value="FPS: -")
        self.found_var = tk.StringVar(value="Erkannte Textfelder: 0")
        self.active_var = tk.StringVar(value="Aktive Zensuren: 0")
        self.total_var = tk.StringVar(value="Gesamt zensiert seit Start: 0")

        for var in (self.status_var, self.res_var, self.fps_var, self.found_var, self.active_var, self.total_var):
            tk.Label(panel, textvariable=var, fg=fg, bg=bg, anchor="w", font=("Segoe UI", 11)).pack(fill="x")

        controls = tk.Frame(self.root, bg=bg)
        controls.pack(fill="x", padx=12, pady=6)

        tk.Label(controls, text="Modus:", fg=fg, bg=bg).pack(side="left")
        self.mode_var = tk.StringVar(value=self.cfg.mode)
        mode_menu = ttk.Combobox(controls, textvariable=self.mode_var, values=("black", "blur"), width=8, state="readonly")
        mode_menu.pack(side="left", padx=6)
        mode_menu.bind("<<ComboboxSelected>>", lambda e: self.pipeline.set_mode(self.mode_var.get()))

        self.toggle_btn = tk.Button(controls, text="Stop", command=self._toggle, width=10)
        self.toggle_btn.pack(side="right")

        tk.Label(self.root, text="Zensur-Log", fg=fg, bg=bg, anchor="w", font=("Segoe UI", 11, "bold")).pack(fill="x", padx=12)
        self.log_box = tk.Listbox(self.root, bg="#111111", fg="#c8ffc8", font=("Consolas", 10), height=18)
        self.log_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _toggle(self) -> None:
        if self.pipeline.is_alive() and not self.pipeline.stop_event.is_set():
            self.pipeline.stop()
            self.toggle_btn.config(text="Start")
        else:
            self.pipeline = Pipeline(self.cfg, self.stats, self.log)
            self.pipeline.set_mode(self.mode_var.get())
            self.pipeline.start()
            self.toggle_btn.config(text="Stop")

    def _tick(self) -> None:
        s = self.stats.snapshot()
        suffix = f" ({s['error']})" if s["error"] else ""
        self.status_var.set(f"Status: {s['model_status']}{suffix}")
        self.res_var.set(f"Aufloesung: {s['width']}x{s['height']}")
        self.fps_var.set(f"FPS: {s['fps']:.1f}")
        self.found_var.set(f"Erkannte Textfelder: {s['texts_found']}")
        self.active_var.set(f"Aktive Zensuren: {s['active_regions']}")

        events, totals = self.log.snapshot()
        self.total_var.set(f"Gesamt zensiert seit Start: {sum(totals.values())}")

        if len(events) > self._log_shown:
            for stamp_time, label, count in events[self._log_shown:]:
                stamp = time.strftime("%H:%M:%S", time.localtime(stamp_time))
                self.log_box.insert("end", f"{stamp}  {label}  x{count}")
            self._log_shown = len(events)
            self.log_box.yview_moveto(1.0)

        self.root.after(300, self._tick)

    def _on_close(self) -> None:
        self.pipeline.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_console(cfg: Config) -> None:
    stats = Stats()
    log = RedactionLog(cfg.log_file)
    pipeline = Pipeline(cfg, stats, log)
    pipeline.start()
    try:
        while pipeline.is_alive():
            time.sleep(1.0)
            s = stats.snapshot()
            print(
                f"[STAT] FPS={s['fps']:.1f} Status={s['model_status']} "
                f"Textfelder={s['texts_found']} aktiv={s['active_regions']}"
            )
    except KeyboardInterrupt:
        pipeline.stop()


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--ocr-every", type=int, default=4)
    p.add_argument("--ocr-max-width", type=int, default=1600)
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--ttl", type=float, default=12.0)
    p.add_argument("--lang", type=str, default="en")
    p.add_argument("--model-tier", choices=("tiny", "small", "medium"), default="small")
    p.add_argument("--det-thresh", type=float, default=0.3)
    p.add_argument("--box-thresh", type=float, default=0.4)
    p.add_argument("--unclip-ratio", type=float, default=1.6)
    p.add_argument("--onnx-threads", type=int, default=2)
    p.add_argument("--mode", choices=("black", "blur"), default="black")
    p.add_argument("--padding", type=int, default=12)
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--log-file", type=str, default="redactions.log")
    a = p.parse_args()
    return Config(
        monitor=a.monitor,
        fps=a.fps,
        ocr_every=max(1, a.ocr_every),
        ocr_max_width=max(320, a.ocr_max_width),
        min_conf=a.min_confidence,
        ttl=a.ttl,
        lang=a.lang,
        model_tier=a.model_tier,
        det_thresh=a.det_thresh,
        box_thresh=a.box_thresh,
        unclip_ratio=a.unclip_ratio,
        onnx_threads=max(1, a.onnx_threads),
        mode=a.mode,
        padding=a.padding,
        gui=not a.no_gui,
        log_file=a.log_file,
    )


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    cfg = parse_args()
    if cfg.gui:
        App(cfg).run()
    else:
        run_console(cfg)


if __name__ == "__main__":
    main()
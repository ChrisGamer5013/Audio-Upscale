"""Processing pipeline: batch jobs, progress reporting, cancellation.

The GUI and CLI both drive this module - it owns file discovery, engine
instantiation, loudness/dither application and output naming.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import audio_io, dsp
from .engines import create_engine
from .presets import get_preset

ProgressCb = Callable[[int, int, float, str, Path], None]
"""(file_index, total_files, fraction_of_current_file, message, current_path)"""


@dataclass
class JobSettings:
    target_sr: int = 48_000
    fmt: str = "FLAC"
    bit_depth: int = 24
    engine_id: str = "dsp"
    model_path: str | None = None
    device_id: int = 0
    preset: str = "music"
    lufs_enabled: bool = False
    lufs_target: float = -14.0
    dither: bool = True
    out_dir: Path | None = None
    suffix: str = "_upscaled"
    save_spectrograms: bool = False


@dataclass
class FileReport:
    source: Path
    output: Path | None = None
    ok: bool = False
    message: str = ""
    notes: list[str] = field(default_factory=list)
    seconds_in: float = 0.0
    seconds_out: float = 0.0
    elapsed_s: float = 0.0


@dataclass
class BatchSummary:
    reports: list[FileReport] = field(default_factory=list)
    cancelled: bool = False

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.reports if r.ok)


def collect_audio_files(paths: list[str | Path]) -> list[Path]:
    """Expand files/folders into a deduped, ordered list of audio files."""
    found: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        p = Path(raw)
        candidates = [p] if p.is_file() else (
            sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else []
        )
        for c in candidates:
            if c.suffix.lower() in audio_io.INPUT_EXTENSIONS:
                key = str(c).lower()
                if key not in seen:
                    seen.add(key)
                    found.append(c)
    return found


def _progress_wrapper(base: ProgressCb | None, i: int, total: int,
                      path: Path) -> Callable[[float, str], None]:
    def inner(frac: float, msg: str) -> None:
        if base:
            base(i, total, frac, msg, path)

    return inner


def process_file(path: Path, settings: JobSettings, engine,
                 progress: ProgressCb | None, index: int = 0,
                 total: int = 1) -> FileReport:
    report = FileReport(source=path)
    t0 = time.perf_counter()
    cb = _progress_wrapper(progress, index, total, path)
    try:
        cb(0.02, "Reading")
        x, sr = audio_io.read_audio(path)
        report.seconds_in = x.shape[-1] / sr

        result = engine.process(
            x, sr, settings.target_sr, get_preset(settings.preset),
            progress=cb,
        )
        report.notes = result.notes
        y = result.audio
        out_sr = result.sr
        report.seconds_out = y.shape[-1] / out_sr

        if settings.lufs_enabled:
            cb(0.95, f"Normalizing to {settings.lufs_target:g} LUFS")
            y = dsp.normalize_loudness(y, out_sr, settings.lufs_target)

        out_dir = settings.out_dir or path.parent
        stem = f"{path.stem}{settings.suffix}"
        ext = audio_io.OUTPUT_FORMATS[settings.fmt].extension
        out_path = Path(out_dir) / f"{stem}{ext}"
        n = 2
        while out_path.exists():
            out_path = Path(out_dir) / f"{stem}_{n}{ext}"
            n += 1

        cb(0.97, "Writing file")
        audio_io.write_audio(
            out_path, y, out_sr,
            fmt=settings.fmt, bit_depth=settings.bit_depth,
            dither=settings.dither,
        )

        if settings.save_spectrograms:
            cb(1.0, "Rendering spectrogram comparison")
            from .spectrogram import save_comparison

            png = out_path.with_suffix(".compare.png")
            save_comparison(x, sr, y, out_sr, png, title=f"{path.name}")

        report.output = out_path
        report.ok = True
        report.message = "ok"
    except Exception as exc:  # keep batch going
        report.ok = False
        report.message = str(exc)
    finally:
        report.elapsed_s = time.perf_counter() - t0
    return report


def run_batch(
    files: list[Path],
    settings: JobSettings,
    progress: ProgressCb | None = None,
    cancel: threading.Event | None = None,
) -> BatchSummary:
    """Process files sequentially (engines already saturate cores/GPU)."""
    engine = create_engine(settings.engine_id, settings.model_path,
                           settings.device_id)
    summary = BatchSummary()
    total = len(files)
    for i, path in enumerate(files):
        if cancel is not None and cancel.is_set():
            summary.cancelled = True
            break
        rep = process_file(path, settings, engine, progress, i, total)
        summary.reports.append(rep)
        if progress:
            progress(i, total, 1.0, "done" if rep.ok else "FAILED", path)
    return summary

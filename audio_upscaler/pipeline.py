"""Processing pipeline: batch jobs, progress reporting, cancellation.

v0.2 additions: pre-DSP chain (declip, denoise), auto preset picking,
mid/side mode, parallel workers, output name templates, keep-tree folder
mirroring, metadata copy, per-file level stats (LUFS + true peak).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime
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
    # --- v0.2 ---
    denoise: float = 0.0            # 0 = off; 1 mild; 2+ aggressive
    declip: bool = False
    mid_side: bool = False          # process stereo as M/S
    ultra_resample: bool = False    # Kaiser beta 16 resampler
    auto_preset: bool = False       # heuristically choose preset per file
    name_template: str = "{name}{suffix}"
    keep_tree: bool = False         # mirror source folder structure in out_dir
    workers: int = 1                # parallel files
    copy_metadata: bool = True
    use_dml: bool = True            # False -> force CPU for ONNX models
    skip_existing: bool = False     # don't reprocess files whose output exists
    passes: int = 1                 # cascade: upscale the result again, 1-8


@dataclass
class FileReport:
    source: Path
    output: Path | None = None
    ok: bool = False
    message: str = ""
    cancelled: bool = False
    notes: list[str] = field(default_factory=list)
    seconds_in: float = 0.0
    seconds_out: float = 0.0
    elapsed_s: float = 0.0
    in_lufs: float | None = None
    out_lufs: float | None = None
    in_true_peak: float | None = None
    out_true_peak: float | None = None

    def stats_line(self) -> str:
        def f(v, fmt="{:.1f}"):
            return fmt.format(v) if v is not None else "-"

        return (f"LUFS {f(self.in_lufs)}->{f(self.out_lufs)}  "
                f"TP {f(self.in_true_peak, '{:.3f}')}->"
                f"{f(self.out_true_peak, '{:.3f}')}")


@dataclass
class BatchSummary:
    reports: list[FileReport] = field(default_factory=list)
    cancelled: bool = False

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.reports if r.ok)


# ------------------------------------------------------------------ files
def collect_audio_files(paths: list[str | Path]) -> list[Path]:
    files, _ = collect_audio_with_roots(paths)
    return files


def collect_audio_with_roots(
    paths: list[str | Path],
) -> tuple[list[Path], dict[Path, Path]]:
    """Expand files/folders into (files, file->root map) for keep-tree."""
    found: list[Path] = []
    roots: dict[Path, Path] = {}
    seen: set[str] = set()
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            candidates = [(p, p.parent)]
        elif p.is_dir():
            candidates = [(q, p) for q in sorted(p.rglob("*")) if q.is_file()]
        else:
            candidates = []
        for c, root in candidates:
            if c.suffix.lower() in audio_io.INPUT_EXTENSIONS:
                key = str(c).lower()
                if key not in seen:
                    seen.add(key)
                    found.append(c)
                    roots[c] = root
    return found, roots


# ------------------------------------------------------------------ naming
def auto_pick_preset(x, sr: int) -> str:
    """Crude but sensible voice/music/restore heuristic."""
    mono = x.shape[0] == 1
    bw = dsp.estimate_bandwidth(x, sr)
    clipped = float((abs(x) >= 0.999).mean())
    if clipped > 0.002:
        return "restore"
    if mono or bw < 12_000:
        return "voice"
    return "music"


def format_output_name(template: str, source: Path, settings: JobSettings,
                       preset_key: str) -> str:
    now = datetime.now()
    fields = {
        "name": source.stem, "suffix": settings.suffix,
        "sr": settings.target_sr, "preset": preset_key,
        "format": settings.fmt.lower(), "FORMAT": settings.fmt.upper(),
        "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H%M%S"),
    }
    try:
        return template.format(**fields)
    except Exception:
        return f"{source.stem}{settings.suffix}"


def output_path_for(source: Path, settings: JobSettings, root: Path | None,
                    preset_key: str, dedupe: bool = True) -> Path:
    base_dir = Path(settings.out_dir) if settings.out_dir else source.parent
    if settings.keep_tree and root and root != source.parent:
        try:
            rel = source.parent.relative_to(root)
            base_dir = base_dir / rel
        except ValueError:
            pass
    ext = audio_io.OUTPUT_FORMATS[settings.fmt].extension
    stem = format_output_name(settings.name_template, source, settings,
                              preset_key)
    out = base_dir / f"{stem}{ext}"
    if dedupe:
        n = 2
        while out.exists():
            out = base_dir / f"{stem}_{n}{ext}"
            n += 1
    return out


# ------------------------------------------------------------------ per-file
def process_file(path: Path, settings: JobSettings, engine,
                 progress: ProgressCb | None, index: int = 0,
                 total: int = 1, root: Path | None = None,
                 force: threading.Event | None = None) -> FileReport:
    report = FileReport(source=path)
    t0 = time.perf_counter()
    cb = None

    class _Forced(Exception):
        """Raised from the progress callback to abort the current file."""

    def inner(frac: float, msg: str) -> None:
        if force is not None and force.is_set():
            raise _Forced()
        if progress:
            progress(index, total, frac, msg, path)

    if force is not None and force.is_set():
        report.cancelled = True
        report.message = "stopped"
        return report

    try:
        inner(0.02, "Reading")
        x, sr = audio_io.read_audio(path)
        if not settings.target_sr or settings.target_sr <= 0:
            settings = replace(settings, target_sr=sr)   # keep source rate

        preset_key = settings.preset
        if settings.auto_preset:
            preset_key = auto_pick_preset(x, sr)
            report.notes.append(f"auto preset: {preset_key}")
        preset = get_preset(preset_key)

        planned = output_path_for(path, settings, root, preset_key,
                                  dedupe=False)
        if settings.skip_existing and planned.exists():
            report.output = planned
            report.ok = True
            report.message = "skipped (output exists)"
            return report

        report.seconds_in = x.shape[-1] / sr
        report.in_lufs = dsp.measure_loudness(x, sr)
        report.in_true_peak = dsp.true_peak(x, sr)

        if settings.declip:
            inner(0.05, "De-clipping")
            x = dsp.declip(x, sr)
        if settings.denoise and settings.denoise > 0:
            inner(0.08, "Denoising (spectral gate)")
            x = dsp.spectral_gate(x, sr, strength=settings.denoise)

        from_ms = settings.mid_side and x.shape[0] == 2
        proc_in = dsp.to_mid_side(x) if from_ms else x
        if from_ms:
            report.notes.append("mid/side mode")

        n_pass = max(1, min(8, int(settings.passes or 1)))
        cur, cur_sr = proc_in, sr
        engine_notes: list[str] = []
        for p in range(n_pass):
            # cascade: each pass roughly doubles the rate until the target
            # is reached; extra passes re-enhance at the target rate.
            if p == n_pass - 1:
                step_sr = settings.target_sr
            elif settings.target_sr > cur_sr:
                step_sr = min(cur_sr * 2, settings.target_sr)
            else:
                step_sr = cur_sr
            lo, hi = p / n_pass, (p + 1) / n_pass

            def pass_prog(frac: float, msg: str, lo=lo, hi=hi, p=p) -> None:
                if n_pass > 1:
                    msg = f"pass {p + 1}/{n_pass}: {msg}"
                inner(lo + (hi - lo) * min(max(frac, 0.0), 1.0), msg)

            result = engine.process(cur, cur_sr, step_sr, preset,
                                    progress=pass_prog)
            cur, cur_sr = result.audio, result.sr
            if not engine_notes:
                engine_notes = list(result.notes)
        y = dsp.from_mid_side(cur) if from_ms else cur
        out_sr = cur_sr
        report.notes.extend(engine_notes)
        if n_pass > 1:
            report.notes.append(
                f"passes: {n_pass} cascade (final {out_sr} Hz)")
        report.seconds_out = y.shape[-1] / out_sr

        if settings.lufs_enabled:
            inner(0.94, f"Normalizing to {settings.lufs_target:g} LUFS")
            y = dsp.normalize_loudness(y, out_sr, settings.lufs_target)

        out_path = output_path_for(path, settings, root, preset_key)
        inner(0.97, "Writing file")
        audio_io.write_audio(
            out_path, y, out_sr, fmt=settings.fmt,
            bit_depth=settings.bit_depth, dither=settings.dither,
        )
        if settings.copy_metadata:
            _try_copy_metadata(path, out_path)

        report.out_lufs = dsp.measure_loudness(y, out_sr)
        report.out_true_peak = dsp.true_peak(y, out_sr)

        if settings.save_spectrograms:
            inner(0.99, "Rendering spectrogram comparison")
            from .spectrogram import save_comparison

            save_comparison(x, sr, y, out_sr,
                            out_path.with_suffix(".compare.png"),
                            title=path.name)

        report.output = out_path
        report.ok = True
        report.message = "ok"
    except _Forced:
        report.cancelled = True
        report.message = "stopped"
    except Exception as exc:  # keep batch going
        report.ok = False
        report.message = str(exc)
    finally:
        report.elapsed_s = time.perf_counter() - t0
    return report


def _try_copy_metadata(src: Path, dst: Path) -> None:
    """Copy tags when both containers support it (ffmpeg required)."""
    ff = audio_io.ffmpeg_path()
    if not ff:
        return
    import subprocess

    tmp = dst.with_suffix(dst.suffix + ".tmp" + dst.suffix)
    try:
        subprocess.run(
            [ff, "-v", "error", "-y", "-i", str(dst), "-i", str(src),
             "-map", "0:a", "-c", "copy", "-map_metadata", "1", str(tmp)],
            capture_output=True, timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(dst)
    except Exception:
        pass
    finally:
        tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------ batch
def run_batch(
    files: list[Path],
    settings: JobSettings,
    progress: ProgressCb | None = None,
    cancel: threading.Event | None = None,
    roots: dict[Path, Path] | None = None,
    force: threading.Event | None = None,
) -> BatchSummary:
    """Process files; `workers > 1` parallelizes independent files.

    `cancel` stops after the current file; `force` aborts the current file
    immediately (checked via the progress callback).

    DSP math runs in numpy/scipy which releases the GIL, so threads are
    enough; the ONNX session supports concurrent `run` calls.
    """
    engine = create_engine(settings.engine_id, settings.model_path,
                           settings.device_id, use_dml=settings.use_dml)
    summary = BatchSummary()
    total = len(files)

    def done_cb(rep: FileReport, i: int) -> None:
        summary.reports.append(rep)
        if rep.cancelled:
            summary.cancelled = True
        if progress:
            progress(i, total, 1.0, "done" if rep.ok else "FAILED",
                     rep.source)

    def stopped() -> bool:
        return ((cancel is not None and cancel.is_set())
                or (force is not None and force.is_set()))

    workers = max(1, min(settings.workers, 8, total))
    if workers == 1:
        for i, path in enumerate(files):
            if stopped():
                summary.cancelled = True
                break
            rep = process_file(path, settings, engine, progress, i, total,
                               (roots or {}).get(path), force=force)
            done_cb(rep, i)
            if rep.cancelled:
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_file, p, settings, engine, progress, i,
                            total, (roots or {}).get(p), force): i
                for i, p in enumerate(files)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                done_cb(fut.result(), i)
    summary.reports.sort(key=lambda r: str(r.source).lower())
    return summary

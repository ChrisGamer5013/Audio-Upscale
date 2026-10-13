"""Command-line interface: python main.py cli ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .audio_io import OUTPUT_FORMATS, available_output_formats
from .engines import engine_status
from .presets import PRESETS
from .pipeline import JobSettings, collect_audio_files, run_batch

_SAMPLE_RATES = (44_100, 48_000, 88_200, 96_000, 176_400, 192_000)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio-upscaler",
        description="Audio Upscaler HD - bandwidth extension / audio "
        "super-resolution (DSP engine + GPU ONNX engine: AMD/Intel/NVIDIA).",
    )
    p.add_argument("inputs", nargs="*", help="audio files or folders")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output directory (default: same as source)")
    p.add_argument("--sr", type=int, default=48_000, choices=_SAMPLE_RATES,
                   help="target sample rate (default: 48000)")
    p.add_argument("--format", default="FLAC", metavar="FMT",
                   help="output format: " + ", ".join(OUTPUT_FORMATS))
    p.add_argument("--bit-depth", type=int, default=24, choices=(16, 24, 32),
                   help="bit depth for WAV/FLAC (lossless)")
    p.add_argument("--engine", default="dsp", choices=list(engine_status()),
                   help="processing engine (default: dsp)")
    p.add_argument("--model", type=Path, default=None,
                   help=".onnx model path (required for --engine onnx)")
    p.add_argument("--device", type=int, default=0,
                   help="GPU device index for DirectML (default: 0)")
    p.add_argument("--preset", default="music", choices=list(PRESETS),
                   help="DSP preset (default: music)")
    p.add_argument("--lufs", type=float, default=None, metavar="TARGET",
                   help="normalize loudness to TARGET LUFS (e.g. -14)")
    p.add_argument("--no-dither", action="store_true",
                   help="disable TPDF dither on 16-bit output")
    p.add_argument("--suffix", default="_upscaled",
                   help="suffix for output filenames")
    p.add_argument("--spectrograms", action="store_true",
                   help="also save before/after spectrogram PNGs")
    p.add_argument("--gpu-info", action="store_true",
                   help="print detected GPUs / acceleration and exit")
    p.add_argument("--recommend-models", action="store_true",
                   help="print VRAM-aware ONNX model recommendations and exit")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _print_progress(i: int, total: int, frac: float, msg: str,
                    path: Path) -> None:
    width = 28
    filled = int(width * max(0.0, min(1.0, frac)))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\r[{i + 1}/{total}] {bar} {frac * 100:5.1f}%  {msg[:48]:<48}"
    )
    sys.stdout.flush()
    if frac >= 1.0:
        sys.stdout.write("\n")


def run_cli(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    if args.gpu_info:
        from .providers import describe, summarize

        print(describe(summarize()))
        return 0

    if args.recommend_models:
        from .recommendations import recommendation_report

        print(recommendation_report())
        return 0

    files = collect_audio_files(args.inputs or [])
    if not files:
        print("No audio files found. Pass files or folders.")
        return 2

    fmt = args.format.upper()
    if fmt not in OUTPUT_FORMATS:
        print(f"Unknown format '{args.format}'. "
              f"Options: {', '.join(OUTPUT_FORMATS)}")
        return 2
    available = available_output_formats()
    if fmt not in available:
        print(f"Format {fmt} needs ffmpeg, which was not found. "
              f"Available: {', '.join(available)}")
        return 2

    settings = JobSettings(
        target_sr=args.sr,
        fmt=fmt,
        bit_depth=args.bit_depth,
        engine_id=args.engine,
        model_path=str(args.model) if args.model else None,
        device_id=args.device,
        preset=args.preset,
        lufs_enabled=args.lufs is not None,
        lufs_target=args.lufs if args.lufs is not None else -14.0,
        dither=not args.no_dither,
        out_dir=args.out,
        suffix=args.suffix,
        save_spectrograms=args.spectrograms,
    )

    print(f"Audio Upscaler HD {__version__}  |  engine={args.engine} "
          f"preset={args.preset} sr={args.sr} fmt={fmt}")
    print(f"Files: {len(files)}\n")

    try:
        summary = run_batch(files, settings, progress=_print_progress)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"\nERROR: {exc}")
        return 1

    print("\n--- summary ---")
    for rep in summary.reports:
        mark = "OK " if rep.ok else "ERR"
        dst = rep.output.name if rep.output else "-"
        print(f"  [{mark}] {rep.source.name} -> {dst}  ({rep.message})")
    print(f"\n{summary.succeeded}/{len(summary.reports)} succeeded.")
    return 0 if summary.succeeded == len(summary.reports) else 1

"""Command-line interface: python main.py cli ..."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .audio_io import INPUT_EXTENSIONS, OUTPUT_FORMATS, available_output_formats
from .engines import engine_status
from .pipeline import JobSettings, collect_audio_with_roots, run_batch

_SAMPLE_RATES = (44_100, 48_000, 88_200, 96_000, 176_400, 192_000)


def _preset_keys() -> list[str]:
    from .presets import all_presets

    return list(all_presets())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio-upscaler",
        description="Audio Upscaler HD - bandwidth extension / audio "
        "super-resolution (DSP engine + generic ONNX GPU engine: "
        "AMD/Intel/NVIDIA).",
    )
    p.add_argument("inputs", nargs="*", help="audio files or folders")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output directory (default: same as source)")
    p.add_argument("--sr", type=int, default=48_000, choices=_SAMPLE_RATES)
    p.add_argument("--format", default="FLAC", metavar="FMT",
                   help="output format: " + ", ".join(OUTPUT_FORMATS))
    p.add_argument("--bit-depth", type=int, default=24, choices=(16, 24, 32))
    p.add_argument("--engine", default="dsp", choices=list(engine_status()))
    p.add_argument("--model", type=Path, default=None,
                   help=".onnx model path (required for --engine onnx)")
    p.add_argument("--device", type=int, default=0,
                   help="GPU device index for DirectML")
    p.add_argument("--preset", default="music", metavar="KEY",
                   help="DSP preset; also reads user presets JSON "
                        "(--compare for multiple)")
    p.add_argument("--auto-preset", action="store_true",
                   help="heuristically pick voice/music/restore per file")
    p.add_argument("--lufs", type=float, default=None, metavar="TARGET",
                   help="normalize loudness to TARGET LUFS (e.g. -14)")
    p.add_argument("--no-dither", action="store_true")
    p.add_argument("--suffix", default="_upscaled")
    p.add_argument("--name-template", default="{name}{suffix}",
                   help="output name template; fields: {name} {suffix} {sr} "
                        "{preset} {format} {date} {time}")
    p.add_argument("--keep-tree", action="store_true",
                   help="mirror input folder structure inside --out")
    p.add_argument("--no-metadata", action="store_true",
                   help="don't copy tags from source (needs ffmpeg)")
    p.add_argument("--spectrograms", action="store_true")
    p.add_argument("--denoise", type=float, default=0.0, metavar="STRENGTH",
                   help="spectral noise gate before upscaling (1 mild, 2 strong)")
    p.add_argument("--declip", action="store_true",
                   help="repair clipped/limited material first")
    p.add_argument("--mid-side", action="store_true",
                   help="process stereo files as mid/side")
    p.add_argument("--ultra", action="store_true",
                   help="ultra-quality resampler (slower, wider stopband)")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel files (1-8; engines already use all cores/GPU)")
    p.add_argument("--compare", metavar="P1,P2", default=None,
                   help="render one output per preset for A/B comparison")
    p.add_argument("--estimate", action="store_true",
                   help="show per-file stats (sr, duration, bandwidth) without "
                        "processing")
    p.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                   help="watch input folder(s) for new files and process them")
    p.add_argument("--gpu-info", action="store_true",
                   help="print detected GPUs / acceleration and exit")
    p.add_argument("--recommend-models", action="store_true",
                   help="print VRAM-aware ONNX model recommendations and exit")
    p.add_argument("--doctor", action="store_true",
                   help="print full diagnostics (versions, providers, ffmpeg) and exit")
    p.add_argument("--check-update", action="store_true",
                   help="check GitHub for a newer release and exit")
    p.add_argument("--get-starter-model", action="store_true",
                   help="download the starter BWE model into models/ and exit")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _print_progress(i: int, total: int, frac: float, msg: str,
                    path: Path) -> None:
    width = 26
    filled = int(width * max(0.0, min(1.0, frac)))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\r[{i + 1}/{total}] {bar} {frac * 100:5.1f}%  {msg[:44]:<44}")
    sys.stdout.flush()
    if frac >= 1.0:
        sys.stdout.write("\n")


def _estimate(files: list[Path]) -> int:
    from .audio_io import read_audio
    from .dsp import estimate_bandwidth

    print(f"{'file':<46} {'in sr':>8} {'dur':>8} {'ch':>3} {'bandwidth':>10}")
    for f in files:
        try:
            x, sr = read_audio(f)
            dur = x.shape[-1] / sr
            bw = estimate_bandwidth(x, sr)
            print(f"{f.name[:46]:<46} {sr:>8} {dur:>6.1f}s {x.shape[0]:>3} "
                  f"{bw / 1000:>8.1f}k")
        except Exception as exc:
            print(f"{f.name[:46]:<46} ERROR: {exc}")
    return 0


def run_cli(argv: list[str]) -> int:
    from .dsp import ensure_channels  # noqa: F401

    args = build_parser().parse_args(argv)

    # ---- info actions --------------------------------------------------
    if args.gpu_info:
        from .providers import describe, summarize

        print(describe(summarize()))
        return 0
    if args.recommend_models:
        from .recommendations import recommendation_report

        print(recommendation_report())
        return 0
    if args.doctor:
        from .providers import diagnostics

        print(diagnostics())
        return 0
    if args.check_update:
        from .updater import check_for_update

        cur, latest, tag = check_for_update()
        if latest:
            print(f"Update available: v{latest} (you have {cur}) - "
                  f"github.com/ChrisGamer5013/Audio-Upscale/releases")
        else:
            print(f"You are up to date (v{cur}).")
        return 0
    if args.get_starter_model:
        from .updater import ensure_starter_model

        p = ensure_starter_model()
        print(p if p else "download failed")
        return 0 if p else 1

    # ---- collect files --------------------------------------------------
    files, roots = collect_audio_with_roots(args.inputs or [])
    if not files:
        print("No audio files found. Pass files or folders.")
        return 2

    if args.estimate:
        return _estimate(files)

    fmt = args.format.upper()
    if fmt not in OUTPUT_FORMATS:
        print(f"Unknown format '{args.format}'. Options: {', '.join(OUTPUT_FORMATS)}")
        return 2
    if fmt not in available_output_formats():
        print(f"Format {fmt} needs ffmpeg, which was not found. "
              f"Available: {', '.join(available_output_formats())}")
        return 2

    def settings_for(preset: str, suffix: str) -> JobSettings:
        return JobSettings(
            target_sr=args.sr, fmt=fmt, bit_depth=args.bit_depth,
            engine_id=args.engine,
            model_path=str(args.model) if args.model else None,
            device_id=args.device, preset=preset,
            lufs_enabled=args.lufs is not None,
            lufs_target=args.lufs if args.lufs is not None else -14.0,
            dither=not args.no_dither, out_dir=args.out, suffix=suffix,
            save_spectrograms=args.spectrograms,
            denoise=args.denoise, declip=args.declip,
            mid_side=args.mid_side, ultra_resample=args.ultra,
            auto_preset=args.auto_preset, name_template=args.name_template,
            keep_tree=args.keep_tree, workers=args.workers,
            copy_metadata=not args.no_metadata,
        )

    def header(extra: str = "") -> None:
        print(f"Audio Upscaler HD {__version__}  |  engine={args.engine} "
              f"preset={args.preset}{extra} sr={args.sr} fmt={fmt} "
              f"workers={args.workers}")
        print(f"Files: {len(files)}\n")

    def report_summary(summary) -> int:
        print("\n--- summary ---")
        for rep in summary.reports:
            mark = "OK " if rep.ok else "ERR"
            dst = rep.output.name if rep.output else "-"
            print(f"  [{mark}] {rep.source.name} -> {dst}")
            if rep.ok:
                print(f"       {rep.stats_line()}  ({rep.elapsed_s:.1f}s)")
            else:
                print(f"       {rep.message}")
        print(f"\n{summary.succeeded}/{len(summary.reports)} succeeded.")
        return 0 if summary.succeeded == len(summary.reports) else 1

    def do_run(file_list, settings):
        return run_batch(file_list, settings, progress=_print_progress,
                         roots=roots)

    try:
        if args.compare:
            keys = [k.strip() for k in args.compare.split(",") if k.strip()]
            valid = set(_preset_keys())
            keys = [k for k in keys if k in valid] or ["music"]
            header("  [compare: " + ",".join(keys) + "]")
            code = 0
            for key in keys:
                print(f"\n=== preset: {key} ===")
                s = settings_for(key, f"_{key}")
                code |= report_summary(do_run(files, s))
            return code

        settings = settings_for(args.preset, args.suffix)

        if args.watch:
            # poll folders for new files and process them as they appear
            seen = {str(f).lower() for f in files}
            print(f"Watching every {args.watch:g}s - Ctrl+C to stop.")
            settings = settings_for(args.preset, args.suffix)
            try:
                while True:
                    if files:
                        header()
                        report_summary(do_run(files, settings))
                        files = []
                    time.sleep(args.watch)
                    new_files, new_roots = collect_audio_with_roots(args.inputs)
                    fresh = [f for f in new_files
                             if str(f).lower() not in seen]
                    if fresh:
                        for f in fresh:
                            seen.add(str(f).lower())
                        roots.update(new_roots)
                        files = fresh
                        print(f"[watch] {len(fresh)} new file(s)")
            except KeyboardInterrupt:
                print("\nwatch stopped")
            return 0

        header()
        summary = do_run(files, settings)
        return report_summary(summary)

    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"\nERROR: {exc}")
        return 1

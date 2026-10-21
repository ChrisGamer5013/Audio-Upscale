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

_SAMPLE_RATES = (0, 44_100, 48_000, 88_200, 96_000, 176_400, 192_000)


def _preset_keys() -> list[str]:
    from .presets import all_presets

    return list(all_presets())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio-upscaler",
        description="Audio Upscaler HD - bandwidth extension / audio "
        "super-resolution (DSP engine + generic ONNX GPU engine: "
        "AMD/Intel/NVIDIA; optional Keras/TensorFlow engine).",
    )
    p.add_argument("inputs", nargs="*", help="audio files or folders")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output directory (default: same as source)")
    p.add_argument("--sr", type=int, default=48_000, choices=_SAMPLE_RATES,
                   help="target sample rate; 0 = keep the source rate")
    p.add_argument("--format", default="FLAC", metavar="FMT",
                   help="output format: " + ", ".join(OUTPUT_FORMATS))
    p.add_argument("--bit-depth", type=int, default=24, choices=(16, 24, 32))
    p.add_argument("--engine", default="dsp", choices=list(engine_status()))
    p.add_argument("--model", type=Path, default=None,
                   help="model path for --engine onnx/keras. Chain ONNX "
                        "models with '+', or pass a bundle manifest "
                        "(model.json / manifest.json) describing several "
                        "stages. .keras/.h5 models need --engine keras")
    p.add_argument("--device", type=int, default=0,
                   help="GPU device index for DirectML")
    p.add_argument("--no-gpu", action="store_true",
                   help="force CPU for ONNX models (some big models compile "
                        "slowly on DirectML)")
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
    p.add_argument("--passes", type=int, default=1, metavar="N",
                   choices=range(1, 9),
                   help="multi-pass cascade: upscale the result again up to "
                        "N times (each pass roughly doubles the rate until "
                        "--sr is reached; extra passes re-enhance at target)")
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
    p.add_argument("--list-models", nargs="?", const="models", default=None,
                   metavar="FOLDER",
                   help="list models/bundles (default: models/) with detected "
                        "kind and exit")
    p.add_argument("--inspect-model", type=Path, default=None, metavar="PATH",
                   help="print full I/O report + sidecar template for a model "
                        "(ONNX chain, bundle manifest or .keras/.h5) and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="show input -> output mapping (and skips) without "
                        "processing anything")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip files whose output already exists (resume-safe)")
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
    if args.list_models:
        from .bundles import MANIFEST_NAMES, load_bundle
        from .engines.keras_engine import keras_status
        from .engines.onnx_engine import OnnxEngine

        folder = Path(args.list_models)
        if not folder.is_dir():
            print(f"Folder not found: {folder.resolve()}")
            return 1
        rows: list[tuple[Path, str]] = []
        for m in sorted(folder.iterdir()):
            if not (m.is_file() and m.suffix.lower() in (".onnx", ".keras", ".h5")):
                continue
            try:
                kind = (OnnxEngine.inspect(m) if m.suffix.lower() == ".onnx"
                        else f"keras ({keras_status()})")
            except Exception as exc:
                kind = f"? ({exc})"
            rows.append((m, kind))
        manifests: list[Path] = []
        for name in MANIFEST_NAMES:
            cand = folder / name
            if cand.is_file():
                manifests.append(cand)
        for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
            for name in MANIFEST_NAMES:
                cand = sub / name
                if cand.is_file():
                    manifests.append(cand)
                    break
        for cand in manifests:
            try:
                bundle = load_bundle(cand)
            except Exception as exc:
                rows.append((cand, f"bundle? {exc}"))
                continue
            if bundle is None:
                continue
            chain = " -> ".join(s.path.name for s in bundle.stages)
            rows.append((cand, f"bundle '{bundle.name}', "
                               f"{len(bundle.stages)} stage(s): {chain}"))
        if not rows:
            print(f"No .onnx/.keras/.h5 models or bundles found in "
                  f"{folder.resolve()}")
            return 1
        print(f"Models in {folder} (--engine onnx|keras --model <path>; "
              "chain ONNX files with +):\n")
        for m, kind in rows:
            label = str(m.relative_to(folder))
            print(f"  {label:<44} [{kind}]")
        if any(m.suffix.lower() in (".keras", ".h5") for m, _ in rows):
            print(f"\nKeras: {keras_status()}")
        return 0
    if args.inspect_model:
        p = args.inspect_model
        try:
            if p.suffix.lower() in (".keras", ".h5"):
                from .engines.keras_engine import KerasEngine

                print(KerasEngine.inspect_report(p))
            else:
                from .engines.onnx_engine import OnnxEngine

                print(OnnxEngine.inspect_report(p))
        except Exception as exc:
            print(f"Error: {exc}")
            return 1
        return 0

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
            use_dml=not args.no_gpu,
            skip_existing=args.skip_existing,
            passes=args.passes,
        )

    def header(extra: str = "") -> None:
        print(f"Audio Upscaler HD {__version__}  |  engine={args.engine} "
              f"preset={args.preset}{extra} sr={args.sr} fmt={fmt} "
              f"workers={args.workers}")
        print(f"Files: {len(files)}\n")

    def report_summary(summary) -> int:
        print("\n--- summary ---")
        for rep in summary.reports:
            mark = "OK " if rep.ok else ("CANC" if rep.cancelled else "ERR")
            dst = rep.output.name if rep.output else "-"
            print(f"  [{mark}] {rep.source.name} -> {dst}")
            if rep.ok:
                print(f"       {rep.stats_line()}  ({rep.elapsed_s:.1f}s)")
                if rep.message != "ok":
                    print(f"       {rep.message}")
            else:
                print(f"       {rep.message}")
        print(f"\n{summary.succeeded}/{len(summary.reports)} succeeded.")
        return 0 if summary.succeeded == len(summary.reports) else 1

    def do_run(file_list, settings):
        import signal
        import threading

        force = threading.Event()

        def _sig(*_a):
            force.set()
            print("\nStopping… (finishing / aborting safely)")

        prev = signal.getsignal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, _sig)
            return run_batch(file_list, settings, progress=_print_progress,
                             roots=roots, force=force, cancel=force)
        finally:
            try:
                signal.signal(signal.SIGINT, prev)
            except Exception:
                pass

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

        if args.dry_run:
            from .pipeline import output_path_for

            print(f"Dry run - {len(files)} file(s), engine={args.engine}, "
                  f"sr={args.sr or 'keep source'}, fmt={fmt}:\n")
            will_skip = 0
            for f in files:
                out = output_path_for(f, settings, roots.get(f),
                                      settings.preset, dedupe=False)
                exists = out.exists()
                if exists and settings.skip_existing:
                    will_skip += 1
                    mark = "  [exists - will skip]"
                elif exists:
                    mark = "  [exists - will get _2 name]"
                else:
                    mark = ""
                print(f"  {f.name}  ->  {out}{mark}")
            print(f"\n{len(files)} file(s), {will_skip} skipped by "
                  "--skip-existing, 0 processed (dry run).")
            return 0

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

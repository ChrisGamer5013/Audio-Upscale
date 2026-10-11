"""CLI for model training:  python main.py train <audio_folder> [options]"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio-upscaler train",
        description="Train a small super-resolution model from a folder of "
        "high-quality audio and export it to models/ as ONNX.",
    )
    p.add_argument("folder", type=Path, help="folder with training audio (wav/flac/mp3...)")
    p.add_argument("--out", type=Path, default=None,
                   help="output .onnx path (default: models/upscaler_<sr>.onnx)")
    p.add_argument("--sr", type=int, default=48_000, help="model sample rate")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=400, help="steps per epoch")
    p.add_argument("--device", default=None, help="'cpu', 'cuda', ... (auto)")
    return p


def run_train(argv: list[str]) -> int:
    import sys

    for stream in (sys.stdout, sys.stderr):  # torch.onnx logs unicode
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    out = args.out or Path("models") / f"upscaler_{args.sr}.onnx"
    try:
        from .training import train

        train(args.folder, out, sr=args.sr, epochs=args.epochs,
              batch=args.batch, steps_per_epoch=args.steps,
              device=args.device)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    print("Done. Select the model in the GUI (ENGINE: ONNX) or use:")
    print(f"  main.py cli <files> --engine onnx --model {out}")
    return 0

"""Train the small bundled starter BWE model (models/starter_bwe_48k.onnx).

Generates a synthetic dataset (sweeps, chords, plucks, noise bursts,
formant-ish tones), trains the tiny trainer for a few minutes on CPU and
exports ONNX. Fast enough to also run on modest machines.

    python scripts/train_starter.py [--steps 80] [--epochs 2]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio_upscaler.audio_io import write_audio  # noqa: E402

SR = 48_000


def _norm(x: np.ndarray) -> np.ndarray:
    return (0.85 * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def _sweep(rng, dur: float) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    f = np.logspace(np.log10(rng.uniform(80, 300)), np.log10(rng.uniform(2e3, 9e3)), t.size)
    ph = np.cumsum(f) / SR * 2 * np.pi
    x = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    return x


def _chord(rng, dur: float) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    f0 = rng.uniform(110, 440)
    x = np.zeros_like(t)
    for ratio in (1.0, 1.26, 1.5, 2.0, 3.0):
        x += np.sin(2 * np.pi * f0 * ratio * t) / (1 + ratio)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t)
    return x * env


def _pluck(rng, dur: float) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    f0 = rng.uniform(80, 600)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 25))
    env = np.exp(-t * rng.uniform(2, 6))
    return x * env


def _noise_burst(rng, dur: float) -> np.ndarray:
    n = int(SR * dur)
    x = rng.standard_normal(n) * np.exp(-np.linspace(0, 4, n))
    # colored: cascade of smooths
    for _ in range(2):
        x = np.convolve(x, np.ones(9) / 9, mode="same")
    return x


def _formants(rng, dur: float) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    f0 = rng.uniform(90, 220)
    x = np.zeros_like(t)
    for k in range(1, 40):
        f = f0 * k
        if f > 10_000:
            break
        # crude formant shaping: peaks near 500/1500/2500
        g = 0.05
        for f_peak in (500, 1500, 2500):
            g += 0.9 * np.exp(-((f - f_peak) ** 2) / (2 * 300.0 ** 2))
        x += g * np.sin(2 * np.pi * f * t) * rng.uniform(0.8, 1.2)
    return x


GENS = (_sweep, _chord, _pluck, _noise_burst, _formants)


def make_dataset(folder: Path, n_files: int = 10, dur: float = 8.0) -> None:
    rng = np.random.default_rng(42)
    for i in range(n_files):
        gen = GENS[i % len(GENS)]
        seg = _norm(gen(rng, dur)) + 0.002 * rng.standard_normal(int(SR * dur))
        write_audio(folder / f"synth_{i:02d}.wav", seg[None, :].astype(np.float32),
                    SR, "WAV", bit_depth=16)
    print(f"dataset: {n_files} files @ {SR} Hz in {folder}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "models" / "starter_bwe_48k.onnx")
    args = ap.parse_args()

    from audio_upscaler.training import train

    with tempfile.TemporaryDirectory(prefix="au_starter_") as tmp:
        tmp_path = Path(tmp)
        make_dataset(tmp_path)
        train(tmp_path, args.out, sr=SR, epochs=args.epochs,
              batch=args.batch, chunk=args.chunk,
              steps_per_epoch=args.steps)
    print(f"starter model ready: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate a synthetic low-bandwidth demo file for quick testing.

    python scripts/make_demo.py [out.wav]

Creates a 3-second sweep + harmonics at 22.05 kHz so the upscaler has
something to chew on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_upscaler.audio_io import write_audio


def main() -> None:
    sr = 22_050
    dur = 3.0
    t = np.arange(int(sr * dur), dtype=np.float64) / sr

    # sweep 200 Hz -> 8 kHz (source bandwidth ~= 8 kHz at 22.05 kHz)
    f = np.logspace(np.log10(200), np.log10(8000), t.size)
    phase = np.cumsum(f) / sr * 2 * np.pi
    tone = np.sin(phase)

    # add harmonics + a bit of noise so the file has a "mix"
    sig = tone + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    rng = np.random.default_rng(1)
    sig += 0.005 * rng.standard_normal(sig.size)
    sig /= np.max(np.abs(sig)) + 1e-9
    sig = 0.8 * sig

    x = np.stack([sig, sig * 0.9]).astype(np.float32)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_lowband.wav")
    write_audio(out, x, sr, fmt="WAV", bit_depth=16)
    print(f"Wrote {out}  ({x.shape[1] / sr:.1f} s, {sr} Hz, stereo)")


if __name__ == "__main__":
    main()

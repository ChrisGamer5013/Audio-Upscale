"""Train a small waveform super-resolution model and export it to ONNX.

Optional feature - requires PyTorch:  pip install -r requirements-train.txt

The model learns to restore bandwidth: training pairs are built on the fly
by low-passing high-quality audio to a random cutoff, so any folder of good
WAV/FLAC files works as training data.

    .venv\\Scripts\\python main.py train <music_folder> [--epochs 10]

Export target: models/upscaler_<sr>.onnx  (waveform [1, 1, T] -> [1, 1, T],
input is the band-limited signal *already resampled* to the target rate,
output the full-band estimate - exactly what OnnxEngine expects).
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterator

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for training but is not installed.\n"
            "Run:  .venv\\Scripts\\python -m pip install -r requirements-train.txt"
        ) from exc
    return torch, nn


# ---------------------------------------------------------------- model
def build_model(torch, nn, channels: int = 48, n_blocks: int = 6,
                kernel: int = 9):
    """Small residual dilated-conv waveform network."""

    class Block(nn.Module):
        def __init__(self, ch: int, k: int, dilation: int):
            super().__init__()
            pad = (k - 1) // 2 * dilation
            self.c1 = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)
            self.c2 = nn.Conv1d(ch, ch, 1)
            self.act = nn.GELU()

        def forward(self, x):
            y = self.act(self.c1(x))
            y = self.c2(y)
            return self.act(x + y)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            pad = (kernel - 1) // 2
            self.inp = nn.Conv1d(1, channels, kernel, padding=pad)
            blocks = [Block(channels, kernel, 2 ** (i % 4))
                      for i in range(n_blocks)]
            self.body = nn.Sequential(*blocks)
            self.out = nn.Conv1d(channels, 1, kernel, padding=pad)

        def forward(self, x):                    # x: (B, 1, T)
            y = self.body(self.inp(x))
            return torch.tanh(self.out(y) + x)   # global residual + tanh

    return Net()


# ---------------------------------------------------------------- data
def _iter_audio_files(folder: Path) -> Iterator[Path]:
    from .audio_io import INPUT_EXTENSIONS

    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in INPUT_EXTENSIONS:
            yield p


class ChunkSampler:
    """Random (lowpassed, full-band) chunk pairs from a folder of audio."""

    def __init__(self, folder: Path, sr: int, chunk: int,
                 cutoff_min: float = 6_000.0, cutoff_max: float = 18_000.0):
        from .audio_io import read_audio

        self.sr = sr
        self.chunk = chunk
        self.cut = (cutoff_min, cutoff_max)
        self.data: list[np.ndarray] = []
        for p in _iter_audio_files(folder):
            try:
                x, fsrc = read_audio(p)
            except Exception:
                continue
            if fsrc != sr:
                from .dsp import resample_to

                x = resample_to(x, fsrc, sr)
            mono = x.mean(axis=0).astype(np.float32)
            if mono.size > chunk * 2:
                self.data.append(mono)
        if not self.data:
            raise RuntimeError(f"No usable audio files found in {folder}")

    def sample_batch(self, torch, batch: int) -> tuple:
        from scipy.signal import butter, sosfilt

        lows, highs = [], []
        rng = np.random.default_rng()
        for _ in range(batch):
            src = self.data[rng.integers(len(self.data))]
            i = int(rng.integers(0, src.size - self.chunk))
            full = src[i: i + self.chunk].astype(np.float32)

            cutoff = float(rng.uniform(*self.cut))
            sos = butter(8, cutoff, "lowpass", fs=self.sr, output="sos")
            low = sosfilt(sos, full).astype(np.float32)
            g = 1.0 / (np.max(np.abs(full)) + 1e-6)
            g = min(g, 1.0) * 0.95
            lows.append(low * g)
            highs.append(full * g)

        lt = torch.from_numpy(np.stack(lows)[:, None, :])
        ht = torch.from_numpy(np.stack(highs)[:, None, :])
        return lt, ht


# ---------------------------------------------------------------- training
def train(data_folder: Path, out_model: Path, sr: int = 48_000,
          epochs: int = 10, batch: int = 16, chunk: int = 16_384,
          steps_per_epoch: int = 400, lr: float = 2e-4,
          device: str | None = None,
          log=print) -> Path:
    torch, nn = _require_torch()

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={dev}  sr={sr}  chunk={chunk}  batch={batch}")

    model = build_model(torch, nn).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def mrstft_loss(pred, target):
        """L1 + single-resolution spectral magnitude loss."""
        l1 = torch.nn.functional.l1_loss(pred, target)
        nfft = 1024
        w = torch.hann_window(nfft, device=pred.device)
        p = torch.stft(pred.squeeze(1), nfft, hop_length=256, window=w,
                       return_complex=True).abs()
        t = torch.stft(target.squeeze(1), nfft, hop_length=256, window=w,
                       return_complex=True).abs()
        return l1 + torch.nn.functional.l1_loss(torch.log1p(p), torch.log1p(t))

    sampler = ChunkSampler(data_folder, sr, chunk)
    log(f"training pool: {len(sampler.data)} files")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tot = 0.0
        for step in range(steps_per_epoch):
            low, high = sampler.sample_batch(torch, batch)
            low, high = low.to(dev), high.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = mrstft_loss(model(low), high)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        log(f"epoch {epoch:3d}/{epochs}  loss={tot / steps_per_epoch:.5f}  "
            f"({time.perf_counter() - t0:.1f}s)")

    # ------------------------------------------------------------- export
    out_model = Path(out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.randn(1, 1, chunk)
    torch.onnx.export(
        model, (dummy,), str(out_model),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "time"}, "output": {2: "time"}},
        opset_version=17,
    )
    log(f"exported -> {out_model}")
    return out_model

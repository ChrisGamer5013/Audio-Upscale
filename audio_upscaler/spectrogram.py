"""Before/after spectrogram rendering (matplotlib, Agg backend - no display)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal


def _mono_logmag(x: np.ndarray, sr: int, nperseg: int):
    mono = x.mean(axis=0) if x.ndim == 2 else x
    f, t, Z = signal.stft(mono, fs=sr, nperseg=nperseg,
                          noverlap=nperseg * 3 // 4)
    mag = np.abs(Z) + 1e-10
    return f, t, 20.0 * np.log10(mag)


def save_comparison(
    orig: np.ndarray,
    sr_orig: int,
    proc: np.ndarray,
    sr_proc: int,
    out_path: str | Path,
    title: str = "",
    max_freq: float = 24_000.0,
) -> Path:
    """Stack original (top) vs processed (bottom) spectrograms into a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panels = []
    for x, sr in ((orig, sr_orig), (proc, sr_proc)):
        nperseg = min(4096, max(256, x.shape[-1] // 16))
        f, t, db = _mono_logmag(x, sr, nperseg)
        panels.append((f, t, db, sr))

    vmax = max(p[2].max() for p in panels)
    vmin = vmax - 90.0

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, constrained_layout=True
    )
    fig.patch.set_facecolor("#181818")
    labels = (f"Original ({sr_orig} Hz)", f"Upscaled ({sr_proc} Hz)")
    for ax, (f, t, db, sr), label in zip(axes, panels, labels):
        kmax = int(np.searchsorted(f, min(max_freq, sr / 2)))
        mesh = ax.pcolormesh(
            t, f[:kmax] / 1000.0, db[:kmax],
            shading="auto", cmap="magma", vmin=vmin, vmax=vmax,
        )
        ax.set_facecolor("#181818")
        ax.set_ylabel("kHz", color="#dddddd")
        ax.set_title(label, color="#dddddd", fontsize=11, loc="left")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_color("#444444")
    axes[-1].set_xlabel("seconds", color="#dddddd")
    if title:
        fig.suptitle(title, color="#ffffff", fontsize=12)
    fig.colorbar(mesh, ax=axes, label="dBFS").ax.tick_params(colors="#aaaaaa")

    fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path

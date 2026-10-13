"""Spectral front/back-ends for ONNX models that don't eat raw waveforms.

Provides numpy/scipy STFT, HTK mel filterbanks, and Griffin-Lim, plus a
parameter set that can come from a ``<model>.json`` sidecar or be inferred
from the model's mel-bin count.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
from scipy import signal


@dataclass
class SpecParams:
    sr: int = 48_000
    n_fft: int = 1024
    hop: int = 256
    win: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float | None = None          # None -> sr/2 (HTK vocoders often 8k)
    power: float = 1.0                 # 1=|S|, 2=|S|^2
    log: str = "log10"                 # none | ln | log10
    top_db: float = 80.0               # clamp for log mels
    gl_iters: int = 24                 # Griffin-Lim iterations (mel2mel only)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict, sr: int) -> "SpecParams":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        p = cls(**kwargs)
        p.sr = int(data.get("sample_rate", sr))
        p.extra = {k: v for k, v in data.items() if k not in known}
        return p


# Default STFT geometry by mel-bin count (used when no sidecar exists)
_MEL_DEFAULTS: dict[int, dict] = {
    64: dict(n_fft=512, hop=160, win=400, fmax=8000.0),     # speech encoders
    80: dict(n_fft=1024, hop=256, win=1024, fmax=8000.0),   # HiFi-GAN family
    128: dict(n_fft=2048, hop=512, win=2048, fmax=None),    # wide-band mel
}

# Complex-STFT input/output shapes (freq bins) ->
_STFT_DEFAULTS: dict[int, dict] = {
    257: dict(n_fft=512, hop=160, win=400),
    321: dict(n_fft=640, hop=320, win=640),
    513: dict(n_fft=1024, hop=256, win=1024),
    1025: dict(n_fft=2048, hop=512, win=2048),
    2049: dict(n_fft=4096, hop=1024, win=4096),
}


def defaults_for_bins(bins: int, sr: int, spectral: bool) -> SpecParams:
    table = _STFT_DEFAULTS if spectral else _MEL_DEFAULTS
    base = dict(table.get(bins, {}))
    p = SpecParams(sr=sr, **{k: v for k, v in base.items() if k != "n_mels"})
    if not spectral:
        p.n_mels = bins
    return p


def stft_complex(mono: np.ndarray, p: SpecParams) -> np.ndarray:
    _, _, S = signal.stft(
        mono.astype(np.float32), fs=p.sr, window="hann",
        nperseg=p.win, noverlap=p.win - p.hop, nfft=p.n_fft,
        boundary="zeros", padded=True,
    )
    return S.astype(np.complex64)  # (F, T)


def istft_complex(S: np.ndarray, p: SpecParams, length: int) -> np.ndarray:
    _, y = signal.istft(
        S, fs=p.sr, window="hann", nperseg=p.win, noverlap=p.win - p.hop,
        nfft=p.n_fft, boundary=True,
    )
    y = y.astype(np.float32)
    if y.size >= length:
        return y[:length]
    return np.pad(y, (0, length - y.size))


def mel_filterbank(sr: int, n_fft: int, n_mels: int,
                   fmin: float, fmax: float) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

    nfreq = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, nfreq)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    fb = np.zeros((n_mels, nfreq), dtype=np.float32)
    for i in range(1, n_mels + 1):
        lo, ce, hi = hz[i - 1], hz[i], hz[i + 1]
        up = (freqs - lo) / max(ce - lo, 1e-9)
        down = (hi - freqs) / max(hi - ce, 1e-9)
        fb[i - 1] = np.maximum(0.0, np.minimum(up, down))
    return fb


def to_features(S: np.ndarray, p: SpecParams,
                spectral: bool) -> np.ndarray:
    """STFT complex -> model input feature (mel log or masked magnitude)."""
    mag = np.abs(S).astype(np.float32)
    if spectral:
        feat = mag.astype(np.float32)
    else:
        fb = getattr(p, "_fb", None)
        if fb is None:
            fmax = p.fmax if p.fmax else p.sr / 2.0
            fb = mel_filterbank(p.sr, p.n_fft, p.n_mels, p.fmin, fmax)
            p._fb = fb
        feat = fb @ (mag ** p.power)
    if p.log == "log10":
        feat = np.log10(np.maximum(feat, 1e-10))
        if p.top_db:
            feat = np.maximum(feat, feat.max() - p.top_db / 10.0)
    elif p.log == "ln":
        feat = np.log(np.maximum(feat, 1e-10))
    return feat.astype(np.float32)


def mel_to_linear_mag(mel_feat: np.ndarray, p: SpecParams) -> np.ndarray:
    """Invert the mel/log transform approximately (for mel2mel models)."""
    if p.log == "log10":
        feat = 10.0 ** mel_feat
    elif p.log == "ln":
        feat = np.exp(mel_feat)
    else:
        feat = mel_feat
    feat = feat ** (1.0 / p.power)
    fb = getattr(p, "_fb", None)
    if fb is None:
        fmax = p.fmax if p.fmax else p.sr / 2.0
        fb = mel_filterbank(p.sr, p.n_fft, p.n_mels, p.fmin, fmax)
    pinv = np.linalg.pinv(fb)
    lin = np.maximum(pinv @ feat, 0.0)
    return lin.astype(np.float32)


def griffin_lim(mag: np.ndarray, p: SpecParams,
                init_phase: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct a complex spectrogram from a magnitude in-place domain."""
    rng = np.random.default_rng(0)
    if init_phase is None:
        phase = np.exp(1j * rng.uniform(-np.pi, np.pi, mag.shape)).astype(np.complex64)
    else:
        phase = np.exp(1j * init_phase.astype(np.float32))
    S = (mag.astype(np.float64) * phase).astype(np.complex128)
    for _ in range(max(1, p.gl_iters)):
        _, x = signal.istft(S, fs=p.sr, window="hann", nperseg=p.win,
                            noverlap=p.win - p.hop, nfft=p.n_fft,
                            boundary=True)
        _, _, Sx = signal.stft(x.astype(np.float32), fs=p.sr, window="hann",
                               nperseg=p.win, noverlap=p.win - p.hop,
                               nfft=p.n_fft, boundary="zeros", padded=True)
        if Sx.shape != S.shape:  # frame count drift; resize Sx
            t = min(Sx.shape[1], S.shape[1])
            phase_new = np.zeros_like(S, dtype=np.complex128)
            phase_new[:, :t] = Sx[:, :t]
            Sx = phase_new
        S = mag.astype(np.complex128) * np.exp(1j * np.angle(Sx))
    _, x = signal.istft(S, fs=p.sr, window="hann", nperseg=p.win,
                        noverlap=p.win - p.hop, nfft=p.n_fft, boundary=True)
    return x.astype(np.float32)

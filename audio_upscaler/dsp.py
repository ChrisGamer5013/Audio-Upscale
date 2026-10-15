"""Core DSP building blocks: resampling, bandwidth estimation/extension,
exciter, air band, loudness normalization, dither and limiting.

All functions operate on arrays shaped ``(channels, samples)`` float32.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

_EPS = 1e-12


def ensure_channels(x: np.ndarray) -> np.ndarray:
    """Coerce to (channels, samples) float32."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :]
    # soundfile gives (samples, channels); transpose when it looks like it
    if x.shape[1] <= 8 and x.shape[0] > x.shape[1]:
        return np.ascontiguousarray(x.T)
    return x


def remove_dc(x: np.ndarray, sr: int, cutoff_hz: float = 5.0) -> np.ndarray:
    """Remove DC / subsonic rumble with a zero-phase 2nd-order highpass."""
    if x.shape[-1] < 128:
        return (x - np.mean(x, axis=-1, keepdims=True)).astype(np.float32)
    sos = signal.butter(2, cutoff_hz, "highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int,
                quality: str = "high") -> np.ndarray:
    """Transparent polyphase resampling (Kaiser windowed)."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    from fractions import Fraction

    beta = 12.0 if quality == "high" else 16.0  # "ultra" -> wider stopband
    frac = Fraction(int(sr_out), int(sr_in)).limit_denominator(1024)
    y = signal.resample_poly(
        x.astype(np.float64),
        frac.numerator,
        frac.denominator,
        axis=-1,
        window=("kaiser", beta),
    )
    return y.astype(np.float32)


def measure_loudness(x: np.ndarray, sr: int) -> float | None:
    """EBU R128 integrated loudness in LUFS; None when not measurable."""
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        v = meter.integrated_loudness(x.T.astype(np.float64))
        return float(v) if np.isfinite(v) else None
    except Exception:
        return None


def true_peak(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Approximate true peak (oversampled) in linear units."""
    if x.size == 0:
        return 0.0
    up = signal.resample_poly(x.astype(np.float64), oversample, 1, axis=-1,
                              window=("kaiser", 8.0))
    return float(np.max(np.abs(up)))


def declip(x: np.ndarray, sr: int, thresh: float = 0.98) -> np.ndarray:
    """Detect clipped regions and rebuild them by interpolation.

    Simple but effective for ISPs/encoder overs: every run of samples pinned
    at >= thresh is replaced with a smooth interpolation slightly *above* the
    clip level, then gently lowpassed to avoid splats.
    """
    x = x.astype(np.float32, copy=True)
    n = x.shape[-1]
    for ch in range(x.shape[0] if x.ndim > 1 else 1):
        row = np.atleast_2d(x)[ch]
        clip = np.abs(row) >= thresh
        if not clip.any():
            continue
        # find runs
        idx = np.flatnonzero(clip)
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for run in runs:
            a, b = run[0], run[-1]
            if a < 1 or b > n - 2:
                row[run] = np.sign(row[run]) * thresh * 0.995
                continue
            left, right = row[a - 1], row[b + 1]
            interp = np.linspace(left, right, run.size + 2)[1:-1]
            peak = np.sign(left) * max(abs(left), abs(right), thresh) * 1.003
            bulge = np.sin(np.linspace(0, np.pi, run.size))
            row[run] = interp + bulge * (peak - interp) * 0.7
        out = np.atleast_2d(x)
        out[ch] = row
        x = out
    return x


def spectral_gate(x: np.ndarray, sr: int, strength: float = 1.5,
                  percentile: float = 20.0) -> np.ndarray:
    """Lightweight spectral-subtraction / Wiener-ish denoiser.

    Learns a per-frequency noise profile from the quietest frames and
    attenuates towards it. strength ~1.0 mild, ~2.0 aggressive.
    """
    n = x.shape[-1]
    if n < 4096 or strength <= 0:
        return x.astype(np.float32)
    nperseg = 1024
    f, t, Z = signal.stft(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2,
                          axis=-1)
    mag = np.abs(Z)
    frame_e = mag.mean(axis=-2, keepdims=True)  # per frame energy
    thr = np.percentile(frame_e, percentile, axis=-1, keepdims=True)
    quiet = frame_e <= thr
    noise = np.full(mag.shape[:2] + (1,), _EPS, dtype=np.float32)
    if quiet.sum() >= 1:
        sel = np.where(quiet, mag, np.nan)
        noise = np.nanmedian(sel, axis=-1, keepdims=True)
        noise = np.where(np.isfinite(noise), noise, _EPS)
    gain = np.clip((mag - strength * noise) / (mag + _EPS), 0.0, 1.0)
    # smooth mask along freq & time to avoid musical noise
    gain = ndimage.uniform_filter(gain, size=(1, 3, 3))
    Z2 = Z * gain
    _, y = signal.istft(Z2, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    m = min(y.shape[-1], n)
    out = x[..., :m].astype(np.float32) * 0.0
    out[..., :m] = y[..., :m]
    if m < n:
        out = np.concatenate([out, x[..., m:]], axis=-1)
    return out.astype(np.float32)


def to_mid_side(x: np.ndarray) -> np.ndarray:
    """Stereo (2, n) -> mid/side (2, n). Non-stereo passes through."""
    if x.shape[0] != 2:
        return x
    m = (x[0] + x[1]) * 0.5
    s = (x[0] - x[1]) * 0.5
    return np.stack([m, s]).astype(np.float32)


def from_mid_side(x: np.ndarray) -> np.ndarray:
    """mid/side (2, n) -> stereo."""
    if x.shape[0] != 2:
        return x
    l = x[0] + x[1]
    r = x[0] - x[1]
    return np.stack([l, r]).astype(np.float32)


def estimate_bandwidth(x: np.ndarray, sr: int, thresh_db: float = -55.0) -> float:
    """Estimate the highest frequency carrying real content (95th-percentile
    spectral magnitude across time, relative to the broadband peak)."""
    nperseg = 4096
    if x.shape[-1] < nperseg * 2:
        return sr / 2.0
    f, _, Z = signal.stft(
        x, fs=sr, nperseg=nperseg, noverlap=nperseg * 3 // 4, axis=-1
    )
    mag = np.percentile(np.abs(Z), 95, axis=(0, 2))  # (freq,)
    peak = mag.max()
    if peak <= 0:
        return sr / 2.0
    db = 20.0 * np.log10(mag / peak + _EPS)
    idx = np.where(db > thresh_db)[0]
    if idx.size == 0:
        return float(f[len(f) // 2])
    return float(min(f[idx[-1]], sr / 2.0))


def _bandpass(x: np.ndarray, sr: float, lo: float, hi: float, order: int = 4):
    hi = min(hi, sr / 2 * 0.98)
    lo = max(1.0, min(lo, hi - 1.0))
    sos = signal.butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


def _highpass(x: np.ndarray, sr: float, cutoff: float, order: int = 4):
    sos = signal.butter(order, min(cutoff, sr / 2 * 0.98), "highpass",
                        fs=sr, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + _EPS))


def extend_bandwidth(
    x: np.ndarray,
    sr: int,
    f_src: float,
    amount: float,
    hf_cap_hz: float,
    slope_db_per_oct: float,
    gate_db: float = -40.0,
) -> np.ndarray:
    """SBR-style spectral-mirror bandwidth extension.

    Content just below ``f_src`` is mirrored upward with a target spectral
    slope, gated so silence/noise does not bloom, and blended at ``amount``.
    Works purely additively: the original band is left untouched.
    """
    if amount <= 0 or hf_cap_hz <= 0:
        return x
    n = x.shape[-1]
    nperseg = 4096
    if n < nperseg * 2:
        return x

    f_cap = min(sr / 2 * 0.98, hf_cap_hz)
    if f_src >= f_cap - 200.0:
        return x

    noverlap = nperseg * 3 // 4
    f, _, Z = signal.stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap, axis=-1)
    if f.shape[0] < 4:
        return x

    k_band = int(np.argmin(np.abs(f - f_src)))
    k_cap = int(np.argmin(np.abs(f - f_cap)))
    k_mirror_max = min(k_cap, 2 * k_band - 1)
    if k_mirror_max <= k_band + 2:
        return x

    kidx = np.arange(k_band + 1, k_mirror_max + 1)
    ks = 2 * k_band - kidx  # mirrored source bins, >= 0 by construction

    # --- transition fade: rise above band edge, fall near cap/mirror limit ---
    k_rise_end = k_band + max(2, int(0.12 * k_band))
    r = np.clip((kidx - k_band) / max(1, k_rise_end - k_band), 0.0, 1.0)
    rise = 0.5 - 0.5 * np.cos(np.pi * r)
    k_tap0 = k_mirror_max - max(2, int(0.15 * (k_mirror_max - k_band)))
    t = np.clip((kidx - k_tap0) / max(1, k_mirror_max - k_tap0), 0.0, 1.0)
    fall = 0.5 + 0.5 * np.cos(np.pi * t)
    fade = (rise * fall).astype(np.float32)[None, :, None]  # (1, K, 1)

    # --- target spectral slope relative to level at the band edge ---
    mag = np.abs(Z)
    edge_lo = max(0, k_band - 8)
    ref = mag[:, edge_lo:k_band, :].mean(axis=1, keepdims=True)  # (ch,1,T)
    rel = f[kidx] / max(f_src, 1.0)
    slope_mult = np.power(rel, -abs(slope_db_per_oct) / 6.0206).astype(np.float32)
    slope_mult = slope_mult[None, :, None]  # (1,K,1)

    # --- per-frame gate: only pass bins well above the mirrored noise floor ---
    src_mag = mag[:, ks, :]  # (ch,K,T)
    gate_ref = np.maximum(ref, _EPS)
    src_db_rel = 20.0 * np.log10(src_mag / gate_ref + _EPS)
    gate = np.clip((src_db_rel - gate_db) / 12.0, 0.0, 1.0)
    gate = ndimage.uniform_filter1d(gate, size=3, axis=-1).astype(np.float32)

    Z2 = Z.copy()
    Z2[:, kidx, :] += (
        Z[:, ks, :] * fade * slope_mult * gate * np.float32(amount)
    ).astype(Z.dtype)

    # Z layout is (ch, freq, time) -> istft defaults (freq_axis=-2,
    # time_axis=-1) are already correct
    _, y = signal.istft(Z2, fs=sr, nperseg=nperseg, noverlap=noverlap)
    _, base = signal.istft(Z, fs=sr, nperseg=nperseg, noverlap=noverlap)
    m = min(n, y.shape[-1], base.shape[-1])

    delta = y[..., :m] - base[..., :m]
    # keep only what belongs to the new high band, then add back
    delta = _highpass(delta.astype(np.float32), sr, f_src * 0.85)
    x = x.astype(np.float32)
    out = x[..., :m] + amount * delta[..., :m]
    if m < n:  # tail untouched
        out = np.concatenate([out, x[..., m:]], axis=-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def harmonic_exciter(
    x: np.ndarray, sr: int, f_low: float, level: float, drive: float, hf_cap_hz: float
) -> np.ndarray:
    """Gentle even/odd harmonic generator band-limited to [f_low, cap]."""
    if level <= 0 or drive <= 0 or hf_cap_hz <= 0:
        return np.zeros_like(x)
    hp = _highpass(x, sr, f_low * 0.75)
    driven = np.tanh(drive * hp)
    band = _bandpass(driven, sr, f_low, hf_cap_hz)
    ref = _rms(hp)
    scale = level * ref / (_rms(band) + _EPS)
    return (band * scale).astype(np.float32)


def air_band(
    x: np.ndarray, sr: int, f_low: float, level: float, hf_cap_hz: float
) -> np.ndarray:
    """High-band noise following the source envelope - keeps extension
    sounding natural on noisy / vintage material."""
    if level <= 0 or hf_cap_hz <= f_low + 100:
        return np.zeros_like(x)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(x.shape[-1]).astype(np.float32)
    band = _bandpass(noise, sr, f_low, hf_cap_hz)

    hp = _highpass(x, sr, f_low * 0.75)
    win = max(16, int(sr * 0.01))  # 10 ms envelope
    env = np.sqrt(
        ndimage.uniform_filter1d(np.square(hp.mean(axis=0)), size=win) + _EPS
    )
    peak = float(env.max()) + _EPS
    env = (env / peak).astype(np.float32)[None, :]
    return (band * env * level * _rms(hp) / (_rms(band) + _EPS) / peak).astype(
        np.float32
    )


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """EBU R128 integrated-loudness normalization with peak guard."""
    try:
        import pyloudnorm as pyln
    except ImportError:  # pragma: no cover
        return x
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(x.T.astype(np.float64))
    if not np.isfinite(loudness):
        return x
    gain = 10.0 ** ((target_lufs - loudness) / 20.0)
    y = (x * gain).astype(np.float32)
    return soft_limit(y, -1.0)


def soft_limit(x: np.ndarray, ceiling_dbfs: float = -1.0) -> np.ndarray:
    """Transparent ceiling: only attenuates/over-saturates when exceeding."""
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= ceiling or peak <= 0:
        return x
    if peak < ceiling * 1.5:  # close call: plain gain trim is cleaner
        return (x * (ceiling / peak)).astype(np.float32)
    y = np.tanh(x * (1.2 / peak)) * (ceiling / np.tanh(1.2))
    return y.astype(np.float32)


def tpdf_dither(x: np.ndarray, bits: int) -> np.ndarray:
    """Add triangular-PDF dither at 1 LSB for the given word length."""
    if bits >= 32:
        return x
    lsb = 1.0 / float(1 << (bits - 1))
    rng = np.random.default_rng()
    noise = (rng.random(x.shape) + rng.random(x.shape) - 1.0) * lsb
    return np.clip(x + noise, -1.0, 1.0 - lsb).astype(np.float32)

"""Processing presets for the DSP engine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    """Tunable parameters for bandwidth extension and sweetening.

    Attributes:
        key: machine name
        label: human name
        description: short summary for the UI
        amount: blend of synthesized high band (0..1)
        exciter: level of harmonic exciter relative to source band (0..1)
        exciter_drive: tanh drive of the exciter (higher -> more harmonics)
        air: level of noise-matched "air" band relative to source (0..1)
        hf_cap_hz: absolute ceiling for generated content
        slope_db_per_oct: target spectral slope of the extended band
        headroom_dbfs: soft limiter ceiling
    """

    key: str
    label: str
    description: str
    amount: float
    exciter: float
    exciter_drive: float
    air: float
    hf_cap_hz: float
    slope_db_per_oct: float
    headroom_dbfs: float = -1.0


PRESETS: dict[str, Preset] = {
    "music": Preset(
        key="music",
        label="Music (balanced)",
        description="Gentle extension up to ~20 kHz, mild exciter. "
        "Best default for full mixes.",
        amount=0.8,
        exciter=0.12,
        exciter_drive=2.5,
        air=0.02,
        hf_cap_hz=20_000,
        slope_db_per_oct=9.0,
    ),
    "voice": Preset(
        key="voice",
        label="Voice / Podcast",
        description="Focus on speech clarity, extension up to ~16 kHz, "
        "restrained brilliance.",
        amount=0.7,
        exciter=0.08,
        exciter_drive=2.0,
        air=0.015,
        hf_cap_hz=16_000,
        slope_db_per_oct=12.0,
    ),
    "restore": Preset(
        key="restore",
        label="Restore old recording",
        description="Noise-matched air band for tape/78 rips, softer "
        "extension to keep noise believable.",
        amount=0.6,
        exciter=0.06,
        exciter_drive=1.8,
        air=0.05,
        hf_cap_hz=18_000,
        slope_db_per_oct=8.0,
    ),
    "reference": Preset(
        key="reference",
        label="Reference (resample only)",
        description="Pristine polyphase resampling to the target rate with "
        "no additive processing. The 'do no harm' option.",
        amount=0.0,
        exciter=0.0,
        exciter_drive=0.0,
        air=0.0,
        hf_cap_hz=0.0,
        slope_db_per_oct=0.0,
    ),
}

DEFAULT_PRESET = "music"


def get_preset(key: str) -> Preset:
    return PRESETS.get(key, PRESETS[DEFAULT_PRESET])

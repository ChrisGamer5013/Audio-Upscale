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


# ---------------------------------------------------------------- custom
def _user_presets_file() -> "Path":
    import os
    from pathlib import Path

    base = Path(os.environ.get("APPDATA", ".")) / "AudioUpscalerHD"
    base.mkdir(parents=True, exist_ok=True)
    return base / "user_presets.json"


def user_presets() -> dict[str, Preset]:
    """User-defined presets from %APPDATA%/AudioUpscalerHD/user_presets.json"""
    import json

    f = _user_presets_file()
    out: dict[str, Preset] = {}
    if f.is_file():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for key, d in data.items():
                out[key] = Preset(
                    key=key,
                    label=d.get("label", key),
                    description=d.get("description", "custom preset"),
                    amount=float(d.get("amount", 0.8)),
                    exciter=float(d.get("exciter", 0.1)),
                    exciter_drive=float(d.get("exciter_drive", 2.0)),
                    air=float(d.get("air", 0.02)),
                    hf_cap_hz=float(d.get("hf_cap_hz", 20_000)),
                    slope_db_per_oct=float(d.get("slope_db_per_oct", 9.0)),
                    headroom_dbfs=float(d.get("headroom_dbfs", -1.0)),
                )
        except Exception:
            pass
    return out


def all_presets() -> dict[str, Preset]:
    merged = dict(PRESETS)
    merged.update(user_presets())
    return merged


def save_user_preset(p: Preset) -> None:
    import json

    f = _user_presets_file()
    data = {}
    if f.is_file():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[p.key] = {
        "label": p.label, "description": p.description, "amount": p.amount,
        "exciter": p.exciter, "exciter_drive": p.exciter_drive, "air": p.air,
        "hf_cap_hz": p.hf_cap_hz, "slope_db_per_oct": p.slope_db_per_oct,
        "headroom_dbfs": p.headroom_dbfs,
    }
    f.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_preset(key: str) -> Preset:
    return all_presets().get(key, PRESETS[DEFAULT_PRESET])

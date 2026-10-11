"""DSP engine: always-available CPU engine (no model needed).

Pipeline: DC removal -> bandwidth estimation -> polyphase resample ->
SBR-style bandwidth extension -> harmonic exciter -> air band ->
soft limit. The original band is never altered, synthesized content
is strictly additive.
"""
from __future__ import annotations

import numpy as np

from .. import dsp
from ..presets import Preset
from .base import ProcessResult, ProgressCb


class DspEngine:
    name = "dsp"

    def process(
        self,
        x: np.ndarray,
        sr: int,
        target_sr: int,
        preset: Preset,
        progress: ProgressCb | None = None,
    ) -> ProcessResult:
        notes: list[str] = []

        def report(frac: float, msg: str) -> None:
            if progress:
                progress(frac, msg)

        report(0.05, "Removing DC / subsonics")
        x = dsp.ensure_channels(x)
        x = dsp.remove_dc(x, sr)

        report(0.15, "Estimating source bandwidth")
        f_src = dsp.estimate_bandwidth(x, sr)
        if f_src < sr / 2 * 0.95:
            notes.append(f"source bandwidth ~{f_src:,.0f} Hz")
        else:
            notes.append("source already uses the full band")

        report(0.30, f"Polyphase resampling {sr} -> {target_sr} Hz")
        y = dsp.resample_to(x, sr, target_sr)
        f_src_up = f_src * (target_sr / sr)  # freq position is unchanged;
        # contents scales with rate -> keep f_src, this line only clarifies
        f_src_up = f_src

        if preset.amount > 0:
            report(0.55, "Extending bandwidth (spectral mirror)")
            target_f = min(preset.hf_cap_hz, target_sr / 2 * 0.95)
            if f_src < target_f - 200:
                y = dsp.extend_bandwidth(
                    y, target_sr, f_src_up,
                    amount=preset.amount,
                    hf_cap_hz=target_f,
                    slope_db_per_oct=preset.slope_db_per_oct,
                )
                notes.append(f"extended to ~{target_f:,.0f} Hz")
            else:
                notes.append("nothing to extend - band already full")

        if preset.exciter > 0:
            report(0.75, "Adding harmonics (exciter)")
            y = y + dsp.harmonic_exciter(
                y, target_sr, min(f_src, preset.hf_cap_hz * 0.7),
                level=preset.exciter, drive=preset.exciter_drive,
                hf_cap_hz=min(preset.hf_cap_hz, target_sr / 2 * 0.95),
            )

        if preset.air > 0:
            report(0.85, "Matching air band")
            y = y + dsp.air_band(
                y, target_sr, min(f_src, preset.hf_cap_hz * 0.9),
                level=preset.air,
                hf_cap_hz=min(preset.hf_cap_hz, target_sr / 2 * 0.95),
            )

        report(0.92, "Soft limiting to ceiling")
        y = dsp.soft_limit(y, preset.headroom_dbfs)
        y = np.nan_to_num(y, nan=0.0).astype(np.float32)

        report(1.0, "Done")
        return ProcessResult(
            audio=y, sr=target_sr, notes=notes,
            input_sr=sr, detected_bandwidth_hz=f_src,
        )

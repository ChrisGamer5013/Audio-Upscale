"""ONNX engine: runs user-provided super-resolution .onnx models.

Acceleration: DirectML (AMD / Intel / NVIDIA on Windows) with automatic
CPU fallback. Long files are processed in overlapping chunks with linear
crossfading. Models whose output length is an integer multiple of the
input length are treated as doing internal N-times resampling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import dsp
from ..presets import Preset
from .base import ProcessResult, ProgressCb

_CHUNK = 1 << 18  # 262144 samples per inference chunk


def _import_ort():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. Run setup_env.bat or "
            "'pip install onnxruntime-directml'."
        ) from exc
    return ort


class OnnxEngine:
    name = "onnx"

    def __init__(self, model_path: str | Path, device_id: int = 0):
        self.ort = _import_ort()
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        available = self.ort.get_available_providers()
        providers: list = []
        if "DmlExecutionProvider" in available:
            providers.append(("DmlExecutionProvider", {"device_id": device_id}))
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        opts = self.ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = self.ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_rank = len(self.session.get_inputs()[0].shape)
        self.output_name = self.session.get_outputs()[0].name

    # ------------------------------------------------------------------
    def _infer_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """chunk: (samples,) float32 -> (model_output_samples,) float32."""
        if self.input_rank == 2:
            inp = chunk[None, :]
        else:
            inp = chunk[None, None, :]
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def _process_channel(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Overlap-add chunked inference. Returns (audio, out/in ratio)."""
        n = x.shape[-1]
        if n == 0:
            return x.astype(np.float32), 1.0
        if n <= _CHUNK:
            y = self._infer_chunk(x)
            return y, (y.size / n)

        fade_in = _CHUNK // 4
        hop = _CHUNK - fade_in
        acc: np.ndarray | None = None
        norm: np.ndarray | None = None
        ratio = 1.0

        pos = 0
        while pos < n:
            chunk = x[pos: pos + _CHUNK]
            if chunk.size < _CHUNK:
                chunk = np.pad(chunk, (0, _CHUNK - chunk.size))
            y = self._infer_chunk(chunk)
            m = y.size
            if acc is None:
                ratio = m / _CHUNK
                total = int(round(n * ratio)) + int(2 * _CHUNK * ratio)
                acc = np.zeros(total, dtype=np.float32)
                norm = np.zeros(total, dtype=np.float32)

            w = np.ones(m, dtype=np.float32)
            f = min(int(round(fade_in * ratio)), m // 2)
            if f > 0:
                ramp = np.linspace(0.0, 1.0, f, endpoint=False)
                if pos > 0:
                    w[:f] = ramp
                if pos + _CHUNK < n:
                    w[-f:] = ramp[::-1]

            o = int(round(pos * ratio))
            acc[o: o + m] += y * w
            norm[o: o + m] += w
            pos += hop

        valid = norm > 1e-6
        acc[valid] /= norm[valid]
        return acc[: int(round(n * ratio))], ratio

    # ------------------------------------------------------------------
    def process(
        self,
        x: np.ndarray,
        sr: int,
        target_sr: int,
        preset: Preset,
        progress: ProgressCb | None = None,
    ) -> ProcessResult:
        def report(f: float, m: str) -> None:
            if progress:
                progress(f, m)

        x = dsp.ensure_channels(x)
        notes = [f"model: {self.model_path.name}"]

        report(0.05, "Running ONNX model")
        chans = []
        ratio = 1.0
        for ci in range(x.shape[0]):
            y, ratio = self._process_channel(x[ci])
            chans.append(y)
            report(0.05 + 0.85 * (ci + 1) / x.shape[0],
                   f"ONNX inference (ch {ci + 1}/{x.shape[0]})")
        y = np.stack(chans)

        ratio_int = int(round(ratio))
        if ratio_int >= 1 and abs(ratio - ratio_int) < 1e-3:
            model_sr = sr * ratio_int
            if ratio_int > 1:
                notes.append(f"model upsampled x{ratio_int} -> {model_sr} Hz")
            if model_sr != target_sr:
                report(0.94, f"Resampling {model_sr} -> {target_sr} Hz")
                y = dsp.resample_to(y, model_sr, target_sr)
        else:
            # Unknown / non-integer rate conversion: match by output length.
            report(0.94, f"Resampling model output -> {target_sr} Hz")
            from scipy import signal

            new_len = int(round(x.shape[-1] * target_sr / sr))
            y = signal.resample(y, new_len, axis=-1).astype(np.float32)

        y = dsp.soft_limit(y, preset.headroom_dbfs)
        y = np.nan_to_num(y, nan=0.0).astype(np.float32)
        report(1.0, "Done")
        return ProcessResult(
            audio=y, sr=target_sr, notes=notes,
            input_sr=sr, detected_bandwidth_hz=dsp.estimate_bandwidth(x, sr),
        )

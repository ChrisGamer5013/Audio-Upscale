"""Optional Keras / TensorFlow engine for ``.keras`` / ``.h5`` models.

TensorFlow is an *optional* dependency (``requirements-keras.txt``); nothing
here is imported at application start beyond numpy.  Models are treated as
waveform -> waveform super-resolution networks: audio is fed in fixed
windows (channels-last or channels-first auto-detected) and the observed
length ratio is used to resample the result to the requested target rate.

If TensorFlow is unavailable for your Python version, convert the model to
ONNX once with ``scripts/keras_to_onnx.py`` (needs ``tf2onnx``) and use the
regular ONNX engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .. import dsp
from .base import ProcessResult, ProgressCb

_CHUNK = 1 << 16  # 65536 samples per inference window (~1.5 s @ 44.1 kHz)


def _import_tf():
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as exc:  # ImportError, DLL load failures, etc.
        if getattr(sys, "frozen", False):
            hint = ("This portable build does not include TensorFlow. "
                    "Convert the model to ONNX on a machine with TensorFlow:\n"
                    "  python scripts\\keras_to_onnx.py model.keras")
        else:
            hint = ("Install it with:  pip install -r requirements-keras.txt\n"
                    "or convert the model once to ONNX:  "
                    "python scripts\\keras_to_onnx.py model.keras")
        raise RuntimeError(
            "TensorFlow is not installed, so .keras/.h5 models cannot run.\n"
            + hint
        ) from exc
    return tf


def keras_status() -> str:
    """Availability for the UI - does not import TensorFlow (that is slow)."""
    try:
        import importlib.util

        if importlib.util.find_spec("tensorflow") is None:
            raise ImportError("tensorflow")
    except Exception:
        return "unavailable (install requirements-keras.txt)"
    return "ready (TensorFlow installed)"


def _detect_layout(shape: list) -> tuple[bool, int]:
    """Returns (channels_last, channels) for a Keras input shape."""
    if len(shape) == 2:
        return True, 1
    if len(shape) >= 3:
        last, second = shape[-1], shape[1]
        if isinstance(last, int) and last in (1, 2):
            return True, last
        if isinstance(second, int) and second in (1, 2):
            return False, second
    return True, 1


def _fixed_len(shape: list, channels_last: bool) -> int | None:
    if len(shape) < 2:
        return None
    if len(shape) == 2:
        dim = shape[1]
    elif channels_last:
        dim = shape[-2]
    else:
        dim = shape[-1]
    return dim if isinstance(dim, int) and dim > 0 else None


class KerasEngine:
    """Waveform super-resolution with a TensorFlow/Keras model."""

    name = "Keras (TensorFlow)"

    def __init__(self, model_path: str | Path | None = None,
                 device_id: int = 0, use_dml: bool = True, model=None):
        self.model_path = (Path(model_path) if model_path
                           else Path("<in-memory>"))
        if model is None:
            tf = _import_tf()
            self._tf = tf
            try:
                model = tf.keras.models.load_model(str(self.model_path),
                                                  compile=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load Keras model '{self.model_path.name}': "
                    f"{exc}\nIf it uses custom layers, load and re-save it "
                    "with TensorFlow 2.x, or convert it to ONNX with "
                    "scripts\\keras_to_onnx.py."
                ) from exc
        else:
            self._tf = None
        self.model = model
        shape = list(getattr(model, "input_shape", None) or [])
        if shape and isinstance(shape[0], (list, tuple)):
            raise RuntimeError("Multi-input Keras models are not supported.")
        if len(shape) not in (2, 3):
            raise RuntimeError(
                f"Unsupported Keras input shape {shape}; expected 2-D or 3-D "
                "waveform input (batch, samples[, channels]).")
        self.input_shape = shape
        self.channels_last, self.chans = _detect_layout(shape)
        self.fixed_len = _fixed_len(shape, self.channels_last)

    # ---------------------------------------------------------- inference
    def _build_input(self, chunk: np.ndarray) -> np.ndarray:
        c, n = chunk.shape
        want = max(1, self.chans)
        if c < want:
            chunk = np.repeat(chunk[:1], want, axis=0)
        elif c > want:
            chunk = chunk[:want]
        if len(self.input_shape) == 2:
            return chunk[None].astype(np.float32)
        if self.channels_last:
            return np.transpose(chunk, (1, 0))[None].astype(np.float32)
        return chunk[None].astype(np.float32)

    def _infer(self, chunk: np.ndarray) -> np.ndarray:
        inp = self._build_input(chunk)
        try:
            out = self.model(inp, training=False)
        except TypeError:
            out = self.model(inp)
        if hasattr(out, "numpy"):
            out = out.numpy()
        if isinstance(out, (list, tuple)):
            out = out[0]  # multi-head models: first output is the waveform
        y = np.asarray(out, dtype=np.float32)
        if y.ndim == 1:
            return y[None]
        y = np.squeeze(y, axis=0)  # drop the batch axis
        if y.ndim == 1:
            return y[None]
        if self.channels_last:
            # (samples, channels) expected; accept (channels, samples) too
            if y.shape[0] <= 2 and y.shape[1] > y.shape[0]:
                return y
            return np.ascontiguousarray(y.T)
        return y

    def _run_signal(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Run a (channels, samples) block through the model with crossfades."""
        chunk_len = self.fixed_len or _CHUNK
        n = x.shape[-1]
        if n == 0:
            return x.astype(np.float32), 1.0
        if n <= chunk_len:
            c = x
            if c.shape[1] < chunk_len:
                c = np.pad(c, ((0, 0), (0, chunk_len - c.shape[1])))
            y = self._infer(c)
            ratio = y.shape[1] / chunk_len
            return y[:, : max(1, int(round(n * ratio)))], ratio

        fade = chunk_len // 4
        hop = chunk_len - fade
        acc = norm = None
        ratio = 1.0
        pos = 0
        while pos < n:
            c = x[:, pos: pos + chunk_len]
            if c.shape[1] < chunk_len:
                c = np.pad(c, ((0, 0), (0, chunk_len - c.shape[1])))
            y = self._infer(c)
            if acc is None:
                ratio = y.shape[1] / chunk_len
                total = int(round(n * ratio)) + int(2 * chunk_len * ratio)
                acc = np.zeros((y.shape[0], total), np.float32)
                norm = np.zeros(total, np.float32)
            w = np.ones(y.shape[1], np.float32)
            f = min(int(fade * ratio), y.shape[1] // 2)
            if f > 0:
                r = np.linspace(0, 1, f, endpoint=False)
                if pos > 0:
                    w[:f] = r
                if pos + chunk_len < n:
                    w[-f:] = r[::-1]
            o = int(round(pos * ratio))
            acc[:, o: o + y.shape[1]] += y * w
            norm[o: o + y.shape[1]] += w
            pos += hop
        good = norm > 1e-6
        acc[:, good] /= norm[good]
        return acc[:, : max(1, int(round(n * ratio)))], ratio

    # ------------------------------------------------------------ engine
    def process(self, x: np.ndarray, sr: int, target_sr: int, preset,
                progress: ProgressCb | None = None) -> ProcessResult:
        report = progress or (lambda frac, msg: None)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        notes = [f"keras waveform, {self.chans} ch, "
                 f"{'channels-last' if self.channels_last else 'channels-first'}"]
        report(0.05, "Running Keras model")

        if self.chans >= 2:
            xx = x if x.shape[0] == 2 else np.repeat(x[:1], 2, axis=0)
            y2, ratio = self._run_signal(xx)
            if y2.shape[0] == x.shape[0]:
                y = y2
            elif y2.shape[0] >= 2 and x.shape[0] == 1:
                y = y2.mean(axis=0, keepdims=True)
            else:
                y = np.repeat(y2[:1], x.shape[0], axis=0)
            report(0.9, "Keras inference")
        else:
            outs: list[np.ndarray] = []
            ratio = 1.0
            for ci in range(x.shape[0]):
                yy, ratio = self._run_signal(x[ci:ci + 1])
                outs.append(yy)
                report(0.05 + 0.85 * (ci + 1) / x.shape[0],
                       f"Keras inference (ch {ci + 1}/{x.shape[0]})")
            y = np.stack(outs)

        ratio_int = int(round(ratio))
        if ratio_int >= 1 and abs(ratio - ratio_int) < 1e-3:
            model_sr = sr * ratio_int
            if ratio_int > 1:
                notes.append(f"model upsampled x{ratio_int} -> {model_sr} Hz")
            if model_sr != target_sr:
                report(0.94, f"Resampling {model_sr} -> {target_sr} Hz")
                y = dsp.resample_to(y, model_sr, target_sr)
        elif abs(ratio - 1.0) > 1e-3:
            report(0.94, f"Resampling model output -> {target_sr} Hz")
            from scipy import signal

            new_len = int(round(x.shape[-1] * target_sr / sr))
            y = signal.resample(y, new_len, axis=-1).astype(np.float32)

        y = dsp.soft_limit(y, preset.headroom_dbfs)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        report(1.0, "Done")
        return ProcessResult(audio=y, sr=target_sr, notes=notes,
                             input_sr=sr,
                             detected_bandwidth_hz=dsp.estimate_bandwidth(x, sr))

    # ------------------------------------------------------- diagnostics
    @classmethod
    def inspect_report(cls, model_path: str | Path) -> str:
        eng = cls(model_path)
        out_shape = getattr(eng.model, "output_shape", None)
        lines = [
            f"Keras model: {eng.model_path.name}",
            f"input shape : {eng.input_shape}",
            f"output shape: {out_shape}",
            f"layout      : {'channels-last' if eng.channels_last else 'channels-first'} "
            f"({eng.chans} channel(s))",
            f"window      : {eng.fixed_len or _CHUNK} samples",
            "",
            "Keras models run on CPU unless a GPU build of TensorFlow is "
            "installed. Convert to ONNX for DirectML/CUDA acceleration.",
        ]
        return "\n".join(lines)


__all__ = ["KerasEngine", "keras_status"]

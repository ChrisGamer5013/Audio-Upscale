"""ONNX engine: runs user-provided ONNX models of (almost) any I/O layout.

Supported model kinds (auto-detected from input/output tensor shapes, with
optional overrides via a ``<model>.json`` sidecar file):

  waveform     [B, T] / [B, 1, T] -> same layout       (super-res nets, etc.)
  mel2mel      [B, n_mels, F] -> [B, n_mels, F']       (AudioSR, VoiceFixer…)
               output is rebuilt to audio with Griffin-Lim
  mel2wave     [B, n_mels, F] -> [B, 1, T]             (HiFi-GAN vocoders…)
  stft2stft    [B, F, T] / [B, 2, F, T] -> same/comp.  (NSNet2, DeepFilterNet…)
               2-channel output is treated as complex audio; a 1-channel
               output with F == F_in is applied as a time-freq mask
  stft2wave    [B, F, T] -> [B, 1, T]

Sidecar ``<model>.json`` options (all optional)::

    {"kind": "mel2mel", "sample_rate": 44100, "n_fft": 1024, "hop": 256,
     "win": 1024, "n_mels": 80, "fmin": 0, "fmax": 8000, "power": 1,
     "log": "log10", "top_db": 80, "gl_iters": 24,
     "chunk_frames": 300, "overlap_frames": 64}

Acceleration: DirectML (AMD / Intel / NVIDIA on Windows), then CUDA (if
installed), then CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import dsp
from ..presets import Preset
from ..spectral import (
    SpecParams, defaults_for_bins, griffin_lim, istft_complex, mel_to_linear_mag,
    stft_complex, to_features,
)
from .base import ProcessResult, ProgressCb

_CHUNK = 1 << 18            # waveform models: samples per inference chunk
_CHUNK_FRAMES = 300         # spectral models: frames per chunk
_OVERLAP_FRAMES = 64


def _import_ort():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. Run setup_env.bat or "
            "'pip install onnxruntime-directml'."
        ) from exc
    return ort


_SPECTRAL_NAME_HINTS = ("mel", "spec", "stft", "fbank", "noisy", "mag")
_MEL_DIMS = {64, 80, 96, 128, 192, 224, 256}
_STFT_DIMS = {161, 201, 257, 321, 401, 513, 641, 1025, 2049}


def _static_dims(shape) -> list[int]:
    return [d for d in shape if isinstance(d, int)]


def _spectral_hint(name: str, dims: list[int]) -> bool:
    return any(h in name for h in _SPECTRAL_NAME_HINTS) or bool(
        _MEL_DIMS.intersection(dims) or _STFT_DIMS.intersection(dims)
    )


def _bin_kind(shape) -> str:
    dims = _static_dims(shape)[1:]
    if _MEL_DIMS.intersection(dims):
        return "mel"
    return "stft"


def _classify_io(inp, out) -> str:
    ishape = list(inp.shape or [])
    oshape = list(out.shape or [])
    iname = inp.name.lower()
    oname = out.name.lower()
    rank_o = len(oshape)

    i_spec = _spectral_hint(iname, _static_dims(ishape)[1:])
    o_spec = _spectral_hint(oname, _static_dims(oshape)[1:])
    o_wave = rank_o in (2, 3) and not o_spec
    if i_spec and o_wave:
        return "mel2wave" if _bin_kind(ishape) == "mel" else "stft2wave"
    if i_spec and o_spec:
        return "mel2mel" if _bin_kind(ishape) == "mel" else "stft2stft"
    if not i_spec and not o_wave:
        return "generic"
    return "waveform"


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
        try:
            self.session = self.ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=providers
            )
        except Exception:
            # GPU provider present but broken (driver etc.) -> bare CPU
            self.session = self.ort.InferenceSession(
                str(self.model_path), sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        self.ins = self.session.get_inputs()
        self.outs = self.session.get_outputs()

        # sidecar config
        self.sidecar: dict = {}
        side = self.model_path.with_suffix(".json")
        if side.is_file():
            try:
                self.sidecar = json.loads(side.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Unreadable sidecar {side.name}: {exc}")

        self.kind = self.sidecar.get("kind") or self._classify()
        self.input_rank = len(self.ins[0].shape or [])
        self._fixed_len = self._fixed_time_len()
        self._spec_params: SpecParams | None = None
        self._probe()

    @classmethod
    def inspect(cls, model_path: str | Path) -> str:
        """Cheap kind classification without a probe run (for model lists)."""
        ort = _import_ort()
        sopts = ort.SessionOptions()
        sopts.log_severity_level = 3
        sess = ort.InferenceSession(str(model_path), sess_options=sopts,
                                    providers=["CPUExecutionProvider"])
        ins, outs = sess.get_inputs(), sess.get_outputs()
        del sess
        return _classify_io(ins[0], outs[0])

    # ---------------------------------------------------------- classify
    def _classify(self) -> str:
        return _classify_io(self.ins[0], self.outs[0])



    @staticmethod
    def _bin_kind(shape) -> str:
        dims = _static_dims(shape)[1:]
        if _MEL_DIMS.intersection(dims):
            return "mel"
        return "stft"

    def _fixed_time_len(self) -> int | None:
        shape = list(self.ins[0].shape or [])
        d = shape[-1]
        if isinstance(d, int) and d >= 64 and self.kind == "waveform":
            return d
        if self.kind == "waveform" and len(shape) == 2 and isinstance(shape[1], int) \
                and shape[1] >= 256:
            return shape[1]
        return None

    # ---------------------------------------------------------- helpers
    def _feed(self, main: np.ndarray) -> dict:
        """Build the feed dict; extra inputs get zeros of their static shape."""
        feed = {self.ins[0].name: main.astype(np.float32)}
        for meta in self.ins[1:]:
            dims = [d if isinstance(d, int) else 1 for d in (meta.shape or [])]
            feed[meta.name] = np.zeros(dims or [1], dtype=np.float32)
        return feed

    def _run(self, arr: np.ndarray) -> np.ndarray:
        out = self.session.run(None, self._feed(arr))[0]
        return np.asarray(out, dtype=np.float32)

    def _probe(self) -> None:
        """Fail fast at load time with a clear message instead of per-file."""
        try:
            if self.kind == "waveform":
                n = self._fixed_len or 4096
                x = np.zeros(n, dtype=np.float32)
                inp = x[None, :] if self.input_rank == 2 else x[None, None, :]
            else:
                f_dim = self._feature_bins()
                t = int(self.sidecar.get("chunk_frames", _CHUNK_FRAMES))
                rank = self.input_rank
                if rank == 2:
                    inp = np.zeros((f_dim, t), np.float32)
                elif rank == 3:
                    inp = np.zeros((1, f_dim, t), np.float32)
                else:
                    inp = np.zeros((1, 1, f_dim, t), np.float32)
            y = self._run(inp)
        except Exception as exc:
            raise RuntimeError(
                f"Model '{self.model_path.name}' ({self.kind}) failed its "
                f"probe run. Input: '{self.ins[0].name}' "
                f"{list(self.ins[0].shape or [])}.\n"
                f"If the model needs non-default STFT/mel parameters, add a "
                f"{self.model_path.stem}.json sidecar (see models\\README.md)."
                f"\n\nOriginal error:\n{exc}"
            ) from exc
        if y.size == 0 or not np.isfinite(y).all():
            raise RuntimeError(
                f"Model '{self.model_path.name}' produced empty/invalid output "
                "on the probe run."
            )

    def _feature_bins(self) -> int:
        dims = _static_dims(self.ins[0].shape or [])
        for d in dims[1:]:
            if d in _MEL_DIMS or d in _STFT_DIMS:
                return d
        return dims[-1] if dims else 80

    def _params(self, sr: int) -> SpecParams:
        if self.sidecar:
            return SpecParams.from_json(self.sidecar, sr)
        if self._spec_params is None or self._spec_params.sr != sr:
            bins = self._feature_bins()
            spectral = self._bin_kind(self.ins[0].shape) == "stft" \
                or bins in _STFT_DIMS
            self._spec_params = defaults_for_bins(bins, sr, spectral)
        return self._spec_params

    # ---------------------------------------------------------- waveform
    def _infer_wave_chunk(self, chunk: np.ndarray) -> np.ndarray:
        inp = chunk[None, :] if self.input_rank == 2 else chunk[None, None, :]
        return self._run(inp).reshape(-1)

    def _wave_channel(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        chunk_len = self._fixed_len or _CHUNK
        n = x.shape[-1]
        if n == 0:
            return x.astype(np.float32), 1.0
        if n <= chunk_len:
            pad = chunk_len - n
            y = self._infer_wave_chunk(
                np.concatenate([x, np.zeros(pad, np.float32)]))
            k = y.size / chunk_len
            return y[: max(1, int(round(n * k)))], k

        fade, hop = chunk_len // 4, chunk_len - chunk_len // 4
        acc = norm = None
        ratio = 1.0
        pos = 0
        while pos < n:
            c = x[pos: pos + chunk_len]
            if c.size < chunk_len:
                c = np.pad(c, (0, chunk_len - c.size))
            y = self._infer_wave_chunk(c)
            if acc is None:
                ratio = y.size / chunk_len
                total = int(round(n * ratio)) + int(2 * chunk_len * ratio)
                acc = np.zeros(total, np.float32)
                norm = np.zeros(total, np.float32)
            w = np.ones(y.size, np.float32)
            f = min(int(fade * ratio), y.size // 2)
            if f > 0:
                r = np.linspace(0, 1, f, endpoint=False)
                if pos > 0:
                    w[:f] = r
                if pos + chunk_len < n:
                    w[-f:] = r[::-1]
            o = int(round(pos * ratio))
            acc[o: o + y.size] += y * w
            norm[o: o + y.size] += w
            pos += hop
        m = norm > 1e-6
        acc[m] /= norm[m]
        return acc[: int(round(n * ratio))], ratio

    # ---------------------------------------------------------- spectral
    def _spectral_channel(self, x: np.ndarray, sr: int) -> np.ndarray:
        p = self._params(sr)
        S = stft_complex(x, p)
        n_frames = S.shape[1]
        step = int(self.sidecar.get("chunk_frames", _CHUNK_FRAMES))
        ov = int(self.sidecar.get("overlap_frames", min(_OVERLAP_FRAMES, step // 2)))
        hop_f = max(1, step - ov)

        out_audio = np.zeros(x.size + p.n_fft + (step + 4) * p.hop, np.float32)
        norm = np.zeros_like(out_audio)

        pos = 0
        while pos < n_frames:
            t_end = min(pos + step, n_frames)
            Sin = S[:, pos:t_end]
            phase_src = np.angle(Sin)
            feat = to_features(Sin, p, self.kind.startswith("stft"))
            rank = self.input_rank
            if rank == 2:
                inp = feat
            elif rank == 3:
                inp = feat[None]
            else:
                inp = feat[None, None]

            y = self._run(inp).reshape(-1)
            audio_seg = self._spectral_to_audio(y, Sin, phase_src, p)
            o = pos * p.hop
            w = np.ones(audio_seg.size, np.float32)
            f = min(ov * p.hop, audio_seg.size // 2)
            if f > 0:
                r = np.linspace(0, 1, f, endpoint=False)
                if pos > 0:
                    w[:f] = r
                if t_end < n_frames:
                    w[-f:] = r[::-1]
            m = min(audio_seg.size, out_audio.size - o)
            out_audio[o:o + m] += audio_seg[:m] * w[:m]
            norm[o:o + m] += w[:m]
            pos += hop_f

        good = norm > 1e-6
        out_audio[good] /= norm[good]
        return out_audio[: x.size]

    def _spectral_to_audio(self, y: np.ndarray, Sin: np.ndarray,
                           phase_src: np.ndarray, p: SpecParams) -> np.ndarray:
        n_seg = (Sin.shape[1] - 1) * p.hop + p.win
        os_ = list(self.outs[0].shape or [])
        isr = list(self.ins[0].shape or [])

        if self.kind in ("mel2wave", "stft2wave"):
            expected = Sin.shape[1] * p.hop
            if y.size == expected:
                return y[:expected]
            # resample odd ratios to expected length
            return signal_resample_len(y, expected)

        dims = _static_dims(os_)[1:]
        two_ch = 2 in dims[:-1] if len(dims) > 1 else False
        bins_in = self._feature_bins()

        if self.kind == "stft2stft":
            f_out = y.size // max(1, Sin.shape[1] * (2 if two_ch else 1))
            if two_ch or (f_out == Sin.shape[0] and y.size == Sin.size * 2):
                c = y.reshape(2, Sin.shape[0], -1)
                t = min(c.shape[2], Sin.shape[1])
                Sout = (c[0][:, :t] + 1j * c[1][:, :t]).astype(np.complex64)
                if t < Sin.shape[1]:
                    Sout = np.pad(Sout, ((0, 0), (0, Sin.shape[1] - t)))
                return istft_complex(Sout, p, n_seg)
            # magnitude/mask output: apply as mask
            m = y.reshape(-1)[: Sin.size].reshape(Sin.shape) if y.size >= Sin.size \
                else np.pad(y, (0, Sin.size - y.size)).reshape(Sin.shape)
            m = np.clip(m, 0.0, 4.0)
            Sout = (np.abs(Sin) * m.astype(np.float32)
                    * np.exp(1j * phase_src.astype(np.float32)))
            return istft_complex(Sout.astype(np.complex64), p, n_seg)

        # mel2mel
        f_out = _static_dims(os_)[-2] if len(os_) >= 3 else bins_in
        if f_out not in _MEL_DIMS and f_out != bins_in:
            f_out = bins_in
        frames = max(1, y.size // max(1, f_out))
        mel_out = y[: f_out * frames].reshape(f_out, frames)
        if frames < Sin.shape[1]:
            mel_out = np.pad(mel_out, ((0, 0), (0, Sin.shape[1] - frames)),
                             mode="edge")
        mel_out = mel_out[:, : Sin.shape[1]]
        mag = mel_to_linear_mag(mel_out, p)
        if mag.shape[1] != phase_src.shape[1]:
            t = min(mag.shape[1], phase_src.shape[1])
            mag, phase_src = mag[:, :t], phase_src[:, :t]
        return griffin_lim(mag, p, init_phase=phase_src)[:n_seg] \
            if mag.size else np.zeros(n_seg, np.float32)

    # ---------------------------------------------------------- generic
    def _generic_channel(self, x: np.ndarray) -> np.ndarray:
        """Unknown layout: feed the raw waveform, hope for the best."""
        inp = x[None].astype(np.float32)
        rank = self.input_rank
        for _ in range(rank - 2):
            inp = inp[:, None]
        y = self._run(inp)
        return y.reshape(-1)

    # ---------------------------------------------------------- process
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
        notes = [f"model: {self.model_path.name}", f"kind: {self.kind}"]

        if self.kind in ("mel2mel", "mel2wave", "stft2stft", "stft2wave",
                         "generic"):
            # spectral models work at one rate: pre-resample to target
            work = dsp.resample_to(x, sr, target_sr)
            chans = []
            for ci in range(work.shape[0]):
                chans.append(self._spectral_channel(work[ci], target_sr))
                report(0.05 + 0.9 * (ci + 1) / work.shape[0],
                       f"ONNX inference (ch {ci + 1}/{work.shape[0]})")
            y = np.stack(chans)
        else:
            report(0.05, "Running ONNX model")
            chans, ratio = [], 1.0
            for ci in range(x.shape[0]):
                yy, ratio = self._wave_channel(x[ci])
                chans.append(yy)
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
            elif abs(ratio - 1.0) > 1e-3:
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


def signal_resample_len(x: np.ndarray, new_len: int) -> np.ndarray:
    from scipy import signal

    return signal.resample(x, int(new_len)).astype(np.float32)

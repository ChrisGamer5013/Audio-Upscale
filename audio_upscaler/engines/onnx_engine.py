"""ONNX engine: runs user-provided ONNX models of (almost) any I/O layout.

Model kinds (auto-detected; `<model>.json` sidecar can override):

  waveform        [B, T] / [B, 1, T] -> same
  mel2mel         [B, n_mels, F] -> [B, n_mels, F']     (Griffin-Lim back-end)
  mel2wave        [B, n_mels, F] -> [B, 1, T]           (vocoders)
  stft2stft       [B, F, T] masks / [B, 2, F, T] patches (denoisers)
  stft2wave       [B, F, T] -> [B, 1, T]
  *2stft (pair)   two outputs (real, imag) -> complex ISTFT back to audio
  chain           "a.onnx+b.onnx": outputs of a feed inputs of b, then the
                  kind is derived from first input / last output
  generic         best effort

Fix notes (v0.2.1): symbolic batch dims like ['batch', 80, 'frames'] are
handled correctly (bin detection no longer assumes a leading static dim);
name hints decide mel vs stft when dims are symbolic; complex-pair outputs
and fixed-shape patch models are supported.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import dsp
from ..presets import Preset
from ..spectral import (
    SpecParams, defaults_for_bins, griffin_lim, istft_complex, mel_to_linear_mag,
    stft_complex, to_features,
)
from .base import ProcessResult, ProgressCb

_CHUNK = 1 << 18
_CHUNK_FRAMES = 300
_OVERLAP_FRAMES = 64

_MEL_DIMS = {64, 80, 96, 128, 192, 224, 256}
_STFT_DIMS = {161, 201, 257, 321, 401, 513, 641, 1025, 2049}
_MEL_NAME_HINTS = ("mel", "fbank")
_STFT_NAME_HINTS = ("stft", "spec", "mag", "noisy", "dft")
_SPECTRAL_NAME_HINTS = _MEL_NAME_HINTS + _STFT_NAME_HINTS


def _import_ort():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. Run setup_env.bat or "
            "'pip install onnxruntime-directml'."
        ) from exc
    return ort


def _bin_dims(shape) -> list[int]:
    """Static dims that look like frequency bins (all > 4)."""
    return [d for d in shape if isinstance(d, int) and d > 4]


def _bin_kind(shape, name: str = "") -> str:
    n = (name or "").lower()
    if any(h in n for h in _MEL_NAME_HINTS):
        return "mel"
    dims = _bin_dims(shape)
    if _MEL_DIMS.intersection(dims):
        return "mel"
    return "stft"


def _spectral_hint(name: str, shape) -> bool:
    n = (name or "").lower()
    if any(h in n for h in _SPECTRAL_NAME_HINTS):
        return True
    dims = _bin_dims(shape)
    return bool(_MEL_DIMS.intersection(dims) or _STFT_DIMS.intersection(dims))


def _embedding_hint(outs: list) -> bool:
    """Heuristic: a single output that looks like a latent/backbone feature
    matrix (e.g. [batch, frames, 512] 'hidden'), not audio or a spectrogram."""
    if len(outs) >= 2:
        return False
    out = outs[0]
    name = (out.name or "").lower()
    shape = list(out.shape or [])
    last = shape[-1] if shape else None
    hints = ("hidden", "embed", "latent", "feature", "feat")
    return any(h in name for h in hints) or (
        len(shape) == 3 and isinstance(last, int)
        and last in (256, 384, 512, 768, 1024, 1280)
        and not _spectral_hint(out.name, shape)
    )


def _embedding_input_hint(inp) -> bool:
    """Heuristic: input that looks like a latent/backbone feature matrix
    (e.g. [batch, frames, 512] 'hidden'), i.e. a head/decoder stage."""
    name = (inp.name or "").lower()
    shape = list(inp.shape or [])
    last = shape[-1] if shape else None
    hints = ("hidden", "embed", "latent", "feature", "feat")
    return any(h in name for h in hints) or (
        len(shape) == 3 and isinstance(last, int)
        and last in (256, 384, 512, 768, 1024, 1280)
        and not _spectral_hint(inp.name, shape)
    )


def split_models(raw) -> list[Path]:
    """Split a model spec into paths. Separators: '+', ',', ';' (any mix),
    so `a.onnx+b.onnx`, `a.onnx, b.onnx` and `a.onnx;b.onnx` all chain."""
    import re as _re

    return [Path(p.strip()) for p in _re.split(r"[+;,]", str(raw))
            if p and p.strip()]


_WAVE_NAME_HINTS = ("wave", "audio", "trim", "out_sig", "signal")


def _is_complex_pair(outs: list) -> bool:
    """(real, imag) style spectrogram pair vs a model with aux outputs."""
    if len(outs) < 2:
        return False
    n0 = (outs[0].name or "").lower()
    n1 = (outs[1].name or "").lower()
    if "real" in n0 or "imag" in n1 or "phase" in n1 or "mag" in n0:
        return True
    if len(outs) > 2:
        return False
    return not any(h in n0 for h in _WAVE_NAME_HINTS)


def _classify_io(inp, outs: list) -> str:
    i_spec = _spectral_hint(inp.name, inp.shape or [])
    oshape = list(outs[0].shape or [])
    rank_o = len(oshape)
    o_spec = _spectral_hint(outs[0].name, oshape)
    o_wave = rank_o in (2, 3) and not o_spec
    mel = _bin_kind(inp.shape or [], inp.name) == "mel"

    if _is_complex_pair(list(outs)):
        # (real, imag) pair -> complex spectrogram reconstruction
        if i_spec:
            return "mel2stft" if mel else "stft2stft"
        return "generic"
    if i_spec and o_wave:
        return "mel2wave" if mel else "stft2wave"
    if i_spec and o_spec:
        return "mel2mel" if mel else "stft2stft"
    if not i_spec and not o_wave:
        return "generic"
    return "waveform"


_SPEC_KEYS = {"sr", "sample_rate", "n_fft", "hop", "win", "n_mels",
              "fmin", "fmax", "power", "log", "top_db", "gl_iters"}


class OnnxEngine:
    name = "onnx"

    def __init__(self, model_path: str | Path, device_id: int = 0,
                 use_dml: bool = True):
        self.ort = _import_ort()
        parts = split_models(model_path)
        if not parts:
            raise FileNotFoundError("No model given.")
        for p in parts:
            if not p.is_file():
                raise FileNotFoundError(f"Model not found: {p}")
        self.model_paths = parts
        self.model_path = parts[-1]  # display name = last stage

        import os

        available = self.ort.get_available_providers()
        providers: list = []
        if use_dml and os.environ.get("AU_NO_DML") != "1" and \
                "DmlExecutionProvider" in available:
            providers.append(("DmlExecutionProvider", {"device_id": device_id}))
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        opts = self.ort.SessionOptions()
        opts.log_severity_level = 3
        self.sessions = []
        for p in parts:
            try:
                self.sessions.append(self.ort.InferenceSession(
                    str(p), sess_options=opts, providers=providers))
            except Exception:
                self.sessions.append(self.ort.InferenceSession(
                    str(p), sess_options=opts,
                    providers=["CPUExecutionProvider"]))

        self.ins = self.sessions[0].get_inputs()
        self.outs = self.sessions[-1].get_outputs()
        self.complex_pair = _is_complex_pair(list(self.outs))

        # sidecar config: model metadata first, then <model>.json files
        # (JSON sidecars override; later chain stages override earlier ones)
        self.sidecar: dict = {}
        for p in parts:
            for k, v in self._metadata_props(p).items():
                self.sidecar.setdefault(k, v)
        for p in parts:
            side = p.with_suffix(".json")
            if side.is_file():
                try:
                    data = json.loads(side.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise RuntimeError(f"Unreadable sidecar {side.name}: {exc}")
                if isinstance(data, dict):
                    self.sidecar.update(data)

        self.kind = self.sidecar.get("kind") or _classify_io(self.ins[0],
                                                             list(self.outs))
        self.input_rank = len(self.ins[0].shape or [])
        ishape = list(self.ins[0].shape or [])
        self.stereo_in = (self.kind == "waveform" and self.input_rank == 3
                          and len(ishape) > 1 and ishape[1] == 2)
        self.patch_mode = self._detect_patch_mode()
        self._fixed_len = self._fixed_time_len()
        self._spec_params: SpecParams | None = None
        self._guard_embedding_only()
        self._probe()

    @staticmethod
    def _metadata_props(path: Path) -> dict:
        """Read key/value pairs some exporters embed in the ONNX model."""
        try:
            import onnx
        except ImportError:
            return {}
        try:
            m = onnx.load(str(path), load_external_data=False)
        except Exception:
            return {}
        d: dict = {}
        for kv in getattr(m, "metadata_props", []) or []:
            try:
                d[kv.key] = json.loads(kv.value)
            except Exception:
                d[kv.key] = kv.value
        return d

    def _guard_embedding_only(self) -> None:
        """A backbone/latent model alone cannot output audio - tell the user
        to chain it with its head instead of producing garbage."""
        if self.sidecar.get("kind") or self.patch_mode:
            return
        if len(self.sessions) == 1 and _embedding_input_hint(self.ins[0]):
            raise RuntimeError(
                f"'{self.model_path.name}' looks like a head/decoder stage "
                f"(input '{self.ins[0].name}' {list(self.ins[0].shape or [])} "
                "is a latent feature matrix), so it needs its backbone "
                "model in front of it.\\n\\n"
                "Chain it after its backbone using '+':\\n"
                f'  --model "backbone.onnx+{self.model_path.name}"\\n'
                "in the GUI: select the backbone first, then this file (or "
                "type the two paths joined with '+'). See models\\\\README.md."
            )
        if len(self.sessions) > 1 or self.complex_pair:
            return
        if not _embedding_hint(list(self.outs)):
            return
        out = self.outs[0]
        shape = list(out.shape or [])
        raise RuntimeError(
                f"'{self.model_path.name}' looks like a feature/backbone "
                f"stage (output '{out.name}' {shape}), not a complete "
                "model - running it alone would produce noise.\n\n"
                "Chain it with its head model using '+':\n"
                f'  --model "{self.model_path.name}+spec_head.onnx"\n'
                "in the GUI: select both files (drag the second one) or type "
                "them joined with '+'. See models\\README.md."
            )

    @classmethod
    def inspect(cls, model_path: str | Path) -> str:
        """Cheap kind classification without a probe run (for model lists)."""
        ort = _import_ort()
        sopts = ort.SessionOptions()
        sopts.log_severity_level = 3
        parts = split_models(model_path)
        first = ort.InferenceSession(str(parts[0]), sess_options=sopts,
                                     providers=["CPUExecutionProvider"])
        last = first if len(parts) == 1 else ort.InferenceSession(
            str(parts[-1]), sess_options=sopts,
            providers=["CPUExecutionProvider"])
        ins, outs = first.get_inputs(), last.get_outputs()
        del first, last
        if len(parts) == 1 and _embedding_input_hint(ins[0]):
            return "head (chain after backbone: a.onnx+b.onnx)"
        if _embedding_hint(list(outs)):
            return "backbone (chain with head: a.onnx+b.onnx)"
        return _classify_io(ins[0], list(outs))

    @classmethod
    def inspect_report(cls, model_path: str | Path) -> str:
        """Human-readable I/O report + suggested sidecar (GUI/CLI helper)."""
        ort = _import_ort()
        sopts = ort.SessionOptions()
        sopts.log_severity_level = 3
        parts = split_models(model_path)
        lines: list[str] = [f"model chain: {len(parts)} stage(s)"]
        for p in parts:
            if not p.is_file():
                return f"Model not found: {p}"
            s = ort.InferenceSession(str(p), sess_options=sopts,
                                     providers=["CPUExecutionProvider"])
            lines.append(f"\n[{p.name}]")
            for t in s.get_inputs():
                lines.append(f"  in : {t.name:<22} {list(t.shape)}")
            for t in s.get_outputs():
                lines.append(f"  out: {t.name:<22} {list(t.shape)}")
            del s
        try:
            kind = cls.inspect(str(model_path))
        except Exception as exc:
            kind = f"? ({exc})"
        lines.append(f"\ndetected kind: {kind}")
        lines.append("\nsuggested sidecar - save next to the model as "
                     f"'{parts[0].stem}.json' to override parameters:\n"
                     f"{cls.sidecar_template(model_path)}")
        return "\n".join(lines)

    @classmethod
    def sidecar_template(cls, model_path: str | Path) -> str:
        """Default sidecar JSON for a model (kind + geometry guesses)."""
        try:
            kind = cls.inspect(model_path)
        except Exception:
            kind = "generic"
        if any(w in kind for w in ("backbone", "head", "chain after")):
            kind = "generic"
        shape: list = []
        name = ""
        try:
            ort = _import_ort()
            sopts = ort.SessionOptions()
            sopts.log_severity_level = 3
            parts = split_models(model_path)
            s = ort.InferenceSession(str(parts[0]), sess_options=sopts,
                                     providers=["CPUExecutionProvider"])
            m = s.get_inputs()[0]
            shape, name = list(m.shape or []), (m.name or "")
            del s
        except Exception:
            pass
        dims = _bin_dims(shape)
        bins = next((d for d in dims if d in _MEL_DIMS or d in _STFT_DIMS),
                    None)
        data: dict = {"kind": kind, "sample_rate": 48000}
        if bins in _MEL_DIMS:
            data.update({"n_mels": bins, "n_fft": 1024, "hop": 256,
                         "win": 1024, "fmin": 0, "fmax": 8000,
                         "log": "log10", "top_db": 80})
        elif bins:
            n_fft = 2 * (bins - 1)
            data.update({"n_fft": n_fft, "hop": max(64, n_fft // 4),
                         "win": n_fft})
        return json.dumps(data, indent=2)

    # ---------------------------------------------------------- shapes
    def _detect_patch_mode(self) -> bool:
        """Fixed small spectral patches e.g. [1, 2, 63, 257] (denoisers)."""
        shape = list(self.ins[0].shape or [])
        if len(shape) < 3:
            return False
        dims = [d for d in shape if isinstance(d, int)]
        if len(dims) < len(shape) - 1:      # not fully fixed
            return False
        return (2 in dims and any(d in _STFT_DIMS for d in dims)
                and max(dims) <= 2048)

    def _fixed_time_len(self) -> int | None:
        shape = list(self.ins[0].shape or [])
        if self.patch_mode:
            return None
        if self.kind == "waveform":
            d = shape[-1]
            if isinstance(d, int) and d >= 64:
                return d
            if len(shape) == 2 and isinstance(shape[1], int) and shape[1] >= 256:
                return shape[1]
        return None

    def _feature_bins(self) -> int:
        dims = _bin_dims(self.ins[0].shape or [])
        for d in dims:
            if d in _MEL_DIMS or d in _STFT_DIMS:
                return d
        name = self.ins[0].name.lower()
        if any(h in name for h in _MEL_NAME_HINTS):
            return 80
        return 513

    # ---------------------------------------------------------- running
    def _relax(self, meta, arr: np.ndarray) -> np.ndarray:
        """Fit `arr` to a tensor's expected rank (pad/trim leading dims)."""
        want = len(meta.shape or [])
        while arr.ndim < want:
            arr = arr[None]
        while arr.ndim > want:
            arr = arr.reshape(arr.shape[arr.ndim - want:]) if arr.ndim > 2 \
                else arr.reshape((-1,))
        return arr.astype(np.float32)

    def _run_chain_raw(self, arr: np.ndarray) -> list[np.ndarray]:
        """Feed arr through all sessions; returns the raw last-stage outputs
        (original shapes preserved)."""
        cur = [arr]
        for si, sess in enumerate(self.sessions):
            feed = {}
            for idx, meta in enumerate(sess.get_inputs()):
                if idx < len(cur):
                    a = cur[idx]
                    # match the model's expected layout where possible
                    if si == 0:
                        a = self._fit_main_input(a, meta)
                    else:
                        a = self._relax(meta, a)
                else:
                    dims = [d if isinstance(d, int) else 1
                            for d in (meta.shape or [])]
                    a = np.zeros(dims or [1], dtype=np.float32)
                feed[meta.name] = a
            cur = [np.asarray(o, dtype=np.float32)
                   for o in sess.run(None, feed)]
        return cur

    def _run_chain(self, arr: np.ndarray) -> list[np.ndarray]:
        """Feed arr through all sessions; returns flattened last outputs."""
        return [o.reshape(-1) for o in self._run_chain_raw(arr)]

    def _fit_main_input(self, arr: np.ndarray, meta) -> np.ndarray:
        if self.patch_mode:
            return arr.astype(np.float32)
        if arr.ndim >= self.input_rank:
            return arr.astype(np.float32)
        if self.input_rank == 2:
            return arr.reshape(-1)[None, :]
        if self.input_rank == 3 and arr.ndim == 2:
            return arr[None]
        return arr

    def _probe(self) -> None:
        try:
            if self.kind == "waveform":
                n = self._fixed_len or 4096
                if self.stereo_in:
                    inp = np.zeros((1, 2, n), np.float32)
                else:
                    x = np.zeros(n, dtype=np.float32)
                    inp = (x[None, :] if self.input_rank == 2
                           else x[None, None, :])
            elif self.patch_mode:
                shape = [d if isinstance(d, int) else 1
                         for d in (self.ins[0].shape or [])]
                inp = np.zeros(shape or [1, 2, 63, 257], dtype=np.float32)
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
            ys = self._run_chain(inp)
        except Exception as exc:
            raise RuntimeError(
                f"Model '{self.model_path.name}' ({self.kind}) failed its "
                f"probe run. Input: '{self.ins[0].name}' "
                f"{list(self.ins[0].shape or [])}.\n"
                f"For custom STFT/mel parameters add a "
                f"{self.model_path.stem}.json sidecar (models\\README.md)."
                f"\n\nOriginal error:\n{exc}"
            ) from exc
        if not ys or any(y.size == 0 or not np.isfinite(y).all() for y in ys):
            raise RuntimeError(
                f"Model '{self.model_path.name}' produced empty/invalid "
                "output on the probe run."
            )

    # ---------------------------------------------------------- waveform
    def _infer_wave_chunk(self, chunk: np.ndarray) -> np.ndarray:
        inp = chunk[None, :] if self.input_rank == 2 else chunk[None, None, :]
        return self._run_chain(inp)[0]

    def _infer_stereo_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """chunk (2, T) -> output (chans, T2) for [B, 2, T] waveform models."""
        raw = self._run_chain_raw(chunk[None])[0]
        y = np.asarray(raw, dtype=np.float32)
        if y.ndim >= 3:
            y = y[0]
        if y.ndim == 1:
            y = y[None]
        return y

    def _wave_stereo(self, x2: np.ndarray) -> tuple[np.ndarray, float]:
        """Run a stereo (2-channel) waveform model over a (2, n) signal."""
        chunk_len = self._fixed_len or _CHUNK
        n = x2.shape[-1]
        if n == 0:
            return x2.astype(np.float32), 1.0

        def infer(c2: np.ndarray) -> np.ndarray:
            if c2.shape[1] < chunk_len:
                c2 = np.pad(c2, ((0, 0), (0, chunk_len - c2.shape[1])))
            return self._infer_stereo_chunk(c2)

        if n <= chunk_len:
            y = infer(x2)
            ratio = y.shape[1] / chunk_len
            return y[:, : max(1, int(round(n * ratio)))], ratio

        fade, hop = chunk_len // 4, chunk_len - chunk_len // 4
        acc = norm = None
        ratio = 1.0
        m = 1
        pos = 0
        while pos < n:
            y = infer(x2[:, pos: pos + chunk_len])
            if acc is None:
                ratio = y.shape[1] / chunk_len
                m = y.shape[0]
                total = int(round(n * ratio)) + int(2 * chunk_len * ratio)
                acc = np.zeros((m, total), np.float32)
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
        return acc[:, : int(round(n * ratio))], ratio

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
    def _params(self, sr: int) -> SpecParams:
        if _SPEC_KEYS.intersection(self.sidecar):
            return SpecParams.from_json(self.sidecar, sr)
        if self._spec_params is None or self._spec_params.sr != sr:
            bins = self._feature_bins()
            spectral = _bin_kind(self.ins[0].shape or [],
                                 self.ins[0].name) == "stft"
            self._spec_params = defaults_for_bins(bins, sr, spectral)
        return self._spec_params

    def _spectral_channel(self, x: np.ndarray, sr: int,
                          report=None) -> np.ndarray:
        if self.patch_mode:
            return self._patch_channel(x, sr, report=report)
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
            if report and (pos // hop_f) % 8 == 0:
                report(min(1.0, (pos + 1) / max(1, n_frames)),
                       "ONNX inference")
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

            ys = self._run_chain(inp)
            audio_seg = self._spectral_to_audio(ys, Sin, phase_src, p)
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

    def _spectral_to_audio(self, ys: list[np.ndarray], Sin: np.ndarray,
                           phase_src: np.ndarray, p: SpecParams) -> np.ndarray:
        n_seg = (Sin.shape[1] - 1) * p.hop + p.win

        if self.complex_pair and len(ys) >= 2:
            frames = Sin.shape[1]
            F = ys[0].size // max(1, frames)
            if F * frames != ys[0].size or F < 8:
                F = Sin.shape[0]
            re = ys[0][: F * frames].reshape(F, frames)
            im = ys[1][: F * frames].reshape(F, frames)
            S = (re + 1j * im).astype(np.complex64)
            if F != Sin.shape[0]:
                n_fft = 2 * (F - 1)
                p2 = replace(p, n_fft=n_fft, win=max(64, n_fft),
                             hop=min(p.hop, n_fft // 2))
            else:
                p2 = p
            n_seg2 = (frames - 1) * p2.hop + p2.win
            return istft_complex(S, p2, n_seg2)

        y = ys[0] if ys else np.zeros(1, np.float32)
        if self.kind in ("mel2wave", "stft2wave"):
            expected = Sin.shape[1] * p.hop
            if y.size == expected:
                return y[:expected]
            return signal_resample_len(y, expected)

        if self.kind == "stft2stft":
            dims = _bin_dims(self.outs[0].shape or [])
            two_ch = 2 in dims[:-1] if len(dims) > 1 else False
            if two_ch:
                c = y.reshape(2, -1)
                frames = c.shape[1] // max(1, Sin.shape[0])
                Sout = c[0].reshape(Sin.shape[0], -1) + 1j * \
                    c[1].reshape(Sin.shape[0], -1)
                t = min(Sout.shape[1], Sin.shape[1])
                Sout = Sout[:, :t]
                if t < Sin.shape[1]:
                    Sout = np.pad(Sout, ((0, 0), (0, Sin.shape[1] - t)))
                return istft_complex(Sout.astype(np.complex64), p, n_seg)
            m = y[: Sin.size].reshape(Sin.shape) if y.size >= Sin.size else \
                np.pad(y, (0, Sin.size - y.size)).reshape(Sin.shape)
            m = np.clip(m, 0.0, 4.0)
            Sout = np.abs(Sin) * m.astype(np.float32) * \
                np.exp(1j * phase_src.astype(np.float32))
            return istft_complex(Sout.astype(np.complex64), p, n_seg)

        # mel2mel
        f_out = self._feature_bins()
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

    # ------------------------------------------------------- patch models
    def _patch_channel(self, x: np.ndarray, sr: int, report=None) -> np.ndarray:
        """Sliding-window run of fixed spectral-patch models (e.g. [1,2,63,257])."""
        shape = [d if isinstance(d, int) else 1 for d in (self.ins[0].shape or [])]
        f_patch = max(d for d in shape if d in _STFT_DIMS or d in _MEL_DIMS)
        t_patch = max(s for s in shape if s != f_patch and s > 8)
        time_first = shape[-1] == f_patch      # [.., T, F] patch layout
        n_fft = 2 * (f_patch - 1)
        p = replace(self._params(sr), n_fft=n_fft, win=n_fft,
                    hop=int(self.sidecar.get("hop", max(64, n_fft // 4))))
        S = stft_complex(x, p)
        F, T = S.shape
        out = S.copy()
        step = max(1, t_patch - t_patch // 4)
        win = np.hanning(t_patch + 2)[1:-1].astype(np.float32)
        for t0 in range(0, T, step):
            if report and (t0 // step) % 8 == 0:
                report(min(1.0, (t0 + 1) / max(1, T)), "ONNX patch inference")
            t1 = min(t0 + t_patch, T)
            seg = S[:, t0:t1]
            if seg.shape[1] < t_patch:
                pad = t_patch - seg.shape[1]
                seg = np.pad(seg, ((0, 0), (0, pad)))
            seg_ri = np.stack([seg.real, seg.imag]).astype(np.float32)  # 2,F,T
            if time_first:
                ri = np.transpose(seg_ri, (0, 2, 1))[None]  # 1,2,T,F
            else:
                ri = seg_ri[None]                           # 1,2,F,T
            ys = self._run_chain(ri)
            y = ys[0]
            chans = y.size // max(1, f_patch * t_patch)
            if time_first:
                out_ri = y[: chans * f_patch * t_patch].reshape(
                    chans, t_patch, f_patch).transpose(0, 2, 1)
            else:
                out_ri = y[: chans * f_patch * t_patch].reshape(
                    chans, f_patch, t_patch)
            if chans >= 2:
                enh = (out_ri[0] + 1j * out_ri[1]).astype(np.complex64)
            else:
                enh = (np.abs(seg) * np.clip(out_ri[0], 0, 4)).astype(np.complex64)
            take = t1 - t0
            w = win[:take]
            out[:, t0:t1] = out[:, t0:t1] * (1 - w) + enh[:, :take] * w
        return istft_complex(out.astype(np.complex64), p, x.size)

    # ---------------------------------------------------------- generic
    def _generic_channel(self, x: np.ndarray) -> np.ndarray:
        inp = x[None].astype(np.float32)
        rank = self.input_rank
        for _ in range(rank - 2):
            inp = inp[:, None]
        return self._run_chain(inp)[0]

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
        chain = " + ".join(p.name for p in self.model_paths)
        notes = [f"model: {chain}", f"kind: {self.kind}"]
        if self.patch_mode:
            notes.append("patch mode")

        if self.kind in ("mel2mel", "mel2wave", "mel2stft", "stft2stft",
                         "stft2wave", "generic"):
            work = dsp.resample_to(x, sr, target_sr)
            chans = []
            for ci in range(work.shape[0]):
                base = 0.05 + 0.9 * ci / max(1, work.shape[0])
                span = 0.9 / max(1, work.shape[0])

                def _ch_report(f: float, m: str, base=base, span=span,
                               ci=ci) -> None:
                    report(base + span * f,
                           f"ONNX inference (ch {ci + 1}/{work.shape[0]})")

                chans.append(self._spectral_channel(work[ci], target_sr,
                                                    report=_ch_report))
                report(0.05 + 0.9 * (ci + 1) / work.shape[0],
                       f"ONNX inference (ch {ci + 1}/{work.shape[0]})")
            y = np.stack(chans)
        else:
            report(0.05, "Running ONNX model")
            if self.stereo_in:
                x2 = x if x.shape[0] == 2 else np.repeat(x, 2, axis=0)
                yb, ratio = self._wave_stereo(x2)
                notes.append("stereo model ([B, 2, T] input)")
                if yb.shape[0] == x.shape[0]:
                    y = yb
                elif yb.shape[0] >= 2 and x.shape[0] == 1:
                    y = yb.mean(axis=0, keepdims=True)
                else:
                    y = np.repeat(yb[:1], x.shape[0], axis=0)
                report(0.9, "ONNX inference")
            else:
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

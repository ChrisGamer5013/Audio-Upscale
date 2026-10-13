"""Engine / I/O / pipeline tests. Run:  python -m pytest tests -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_upscaler import dsp
from audio_upscaler.audio_io import read_audio, write_audio
from audio_upscaler.engines import create_engine, engine_status
from audio_upscaler.pipeline import JobSettings, collect_audio_files, run_batch
from audio_upscaler.presets import get_preset
from audio_upscaler.providers import summarize


def _sweep(sr: int, dur: float = 1.5, f0: float = 200.0, f1: float = 8000.0):
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    f = np.linspace(f0, f1, t.size)
    phase = np.cumsum(f) / sr * 2 * np.pi
    x = np.sin(phase) + 0.2 * np.sin(2 * phase)
    x /= np.max(np.abs(x))
    return x.astype(np.float32)


def _highband_energy(x: np.ndarray, sr: int, lo: float) -> float:
    mono = x.mean(axis=0) if x.ndim == 2 else x
    f, t, Z = dsp.signal.stft(mono, fs=sr, nperseg=2048, noverlap=1024)
    idx = f >= lo
    return float(np.mean(np.abs(Z[idx])))


def test_dsp_engine_reference_resamples():
    sr_in, sr_out = 22_050, 48_000
    x = _sweep(sr_in)
    eng = create_engine("dsp")
    out = eng.process(x, sr_in, sr_out, get_preset("reference"))
    assert out.sr == sr_out
    assert out.audio.dtype == np.float32
    assert np.isfinite(out.audio).all()
    assert abs(out.audio.shape[-1] / sr_out - 1.5) < 0.05
    # resample-only should not *create* much above the old Nyquist (11 kHz)
    assert _highband_energy(out.audio, sr_out, 12_000) < _highband_energy(x, sr_in, 8_000) * 3


def test_dsp_engine_extends_bandwidth():
    sr_in, sr_out = 22_050, 48_000
    x = _sweep(sr_in)
    eng = create_engine("dsp")
    ref = eng.process(x, sr_in, sr_out, get_preset("reference"))
    up = eng.process(x, sr_in, sr_out, get_preset("music"))
    # "music" preset must add energy in the 12-20 kHz region vs reference
    e_ref = _highband_energy(ref.audio, sr_out, 12_000)
    e_up = _highband_energy(up.audio, sr_out, 12_000)
    assert e_up > e_ref * 1.1
    assert np.max(np.abs(up.audio)) <= 1.0 + 1e-6


def test_remove_dc_and_limit():
    x = np.linspace(-0.8, 0.8, 4410, dtype=np.float32) + 0.1
    y = dsp.remove_dc(x, 44_100)
    assert abs(float(np.mean(y))) < 0.02
    clipped = np.array([2.0, -3.0, 0.5], dtype=np.float32)
    lim = dsp.soft_limit(clipped, -1.0)
    assert np.max(np.abs(lim)) <= 10 ** (-1.0 / 20) + 1e-6


def test_audio_io_roundtrip_wav(tmp_path):
    x = _sweep(44_100, 0.3)
    p = tmp_path / "rt.wav"
    write_audio(p, x, 44_100, fmt="WAV", bit_depth=24)
    y, sr = read_audio(p)
    assert sr == 44_100
    assert y.shape[-1] == x.shape[-1]
    assert y.dtype == np.float32


def test_pipeline_batch(tmp_path):
    x = _sweep(22_050, 0.4)
    src = tmp_path / "in.wav"
    write_audio(src, x, 22_050, fmt="WAV", bit_depth=16)
    out_dir = tmp_path / "out"
    settings = JobSettings(target_sr=48_000, fmt="FLAC", out_dir=out_dir,
                           preset="music", save_spectrograms=False)
    files = collect_audio_files([tmp_path])
    assert src in files
    summary = run_batch(files, settings)
    assert summary.succeeded == 1
    assert summary.reports[0].output is not None
    assert summary.reports[0].output.exists()
    assert summary.reports[0].output.suffix.lower() == ".flac"


def test_collect_audio_files(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.txt").write_bytes(b"")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.flac").write_bytes(b"")
    files = collect_audio_files([tmp_path])
    names = {f.name for f in files}
    assert "a.wav" in names and "c.flac" in names and "b.txt" not in names


def _make_identity_model(path: Path, shape: list, name: str) -> None:
    onnx = pytest.importorskip("onnx")
    x = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)
    node = onnx.helper.make_node("Identity", [name], ["output"])
    graph = onnx.helper.make_graph([node], "g", [x], [y])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


@pytest.mark.skipif(
    not engine_status()["onnx"].startswith("ready"),
    reason="onnxruntime not installed",
)
def test_onnx_waveform_identity_runs(tmp_path):
    from audio_upscaler.engines.onnx_engine import OnnxEngine

    m = tmp_path / "id.onnx"
    _make_identity_model(m, [1, 1, None], "wave")
    eng = OnnxEngine(m)
    assert eng.kind == "waveform"
    x = _sweep(22_050, 0.2)
    out = eng.process(x, 22_050, 48_000, get_preset("reference"))
    assert out.sr == 48_000 and np.isfinite(out.audio).all()
    assert "kind: waveform" in out.notes


@pytest.mark.skipif(
    not engine_status()["onnx"].startswith("ready"),
    reason="onnxruntime not installed",
)
def test_onnx_mel2mel_identity_runs(tmp_path):
    from audio_upscaler.engines.onnx_engine import OnnxEngine

    m = tmp_path / "mel_id.onnx"
    _make_identity_model(m, [1, 80, None], "mel")
    eng = OnnxEngine(m)
    assert eng.kind == "mel2mel"
    x = _sweep(44_100, 0.25)
    out = eng.process(x, 44_100, 48_000, get_preset("reference"))
    assert out.sr == 48_000 and np.isfinite(out.audio).all()
    assert out.audio.shape[-1] > 0


@pytest.mark.skipif(
    not engine_status()["onnx"].startswith("ready"),
    reason="onnxruntime not installed",
)
def test_onnx_stft_mask_identity_runs(tmp_path):
    from audio_upscaler.engines.onnx_engine import OnnxEngine

    m = tmp_path / "mask_id.onnx"
    _make_identity_model(m, [1, 513, None], "noisy_mag")
    eng = OnnxEngine(m)
    assert eng.kind == "stft2stft"
    x = _sweep(44_100, 0.25)
    out = eng.process(x, 44_100, 48_000, get_preset("reference"))
    assert out.sr == 48_000 and np.isfinite(out.audio).all()


def test_provider_summary():
    s = summarize()
    assert isinstance(s.gpus, list)


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    for fn in [
        test_dsp_engine_reference_resamples,
        test_dsp_engine_extends_bandwidth,
        test_remove_dc_and_limit,
        test_provider_summary,
    ]:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"core tests passed in {time.perf_counter() - t0:.2f}s")

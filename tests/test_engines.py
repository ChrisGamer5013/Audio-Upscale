"""Engine / I/O / pipeline tests. Run:  python -m pytest tests -q"""
from __future__ import annotations

import sys
import threading
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


@pytest.mark.skipif(
    not engine_status()["onnx"].startswith("ready"),
    reason="onnxruntime not installed",
)
def test_onnx_symbolic_batch_dims(tmp_path):
    """['batch', 80, 'frames'] mel models must classify as mel, not STFT
    (regression: symbolic batch dims used to slice off the 80)."""
    from audio_upscaler.engines.onnx_engine import OnnxEngine

    m = tmp_path / "bb.onnx"
    _make_identity_model(m, ["batch", 80, "frames"], "mel")
    eng = OnnxEngine(m, use_dml=False)
    assert eng.kind == "mel2mel"
    assert eng._feature_bins() == 80
    x = _sweep(44_100, 0.2)
    out = eng.process(x, 44_100, 48_000, get_preset("reference"))
    assert np.isfinite(out.audio).all()


@pytest.mark.skipif(
    not engine_status()["onnx"].startswith("ready"),
    reason="onnxruntime not installed",
)
def test_onnx_backbone_guard(tmp_path):
    """A bare latent/backbone model must refuse to run with a hint."""
    from audio_upscaler.engines.onnx_engine import OnnxEngine

    m = tmp_path / "backbone.onnx"
    _make_identity_model(m, [1, 80, None], "mel")
    # rewrite output to a latent: [batch, frames, 512] 'hidden'
    onnx = pytest.importorskip("onnx")
    model = onnx.load(str(m))
    out = model.graph.output[0]
    out.name = "hidden"
    out.type.tensor_type.shape.dim[0].dim_param = "batch"
    out.type.tensor_type.shape.dim[1].dim_value = 0  # clear
    out.type.tensor_type.shape.dim[1].dim_param = "frames"
    out.type.tensor_type.shape.dim[2].dim_value = 512
    model.graph.node[0].output[0] = "hidden"
    onnx.save(model, str(m))
    with pytest.raises(RuntimeError, match="[Cc]hain"):
        OnnxEngine(m, use_dml=False)


@pytest.mark.skipif(
    not engine_status()["onnx"].startswith("ready"),
    reason="onnxruntime not installed",
)
def test_onnx_chain_two_stages(tmp_path):
    """Two-stage chain: outputs of stage 1 feed stage 2; last-stage (real,
    imag) pair becomes audio via complex ISTFT."""
    from audio_upscaler.engines.onnx_engine import OnnxEngine

    onnx = pytest.importorskip("onnx")

    def save(nodes, name, ins, outs, path):
        graph = onnx.helper.make_graph(nodes, name, ins, outs)
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
        model.ir_version = 8
        onnx.save(model, str(path))

    h = onnx.helper.make_tensor_value_info("hidden", onnx.TensorProto.FLOAT,
                                           [1, 80, None])
    save([onnx.helper.make_node("Identity", ["mel"], ["hidden"])],
         "backbone",
         [onnx.helper.make_tensor_value_info("mel", onnx.TensorProto.FLOAT,
                                             [1, 80, None])],
         [h], tmp_path / "backbone.onnx")
    save([onnx.helper.make_node("Identity", ["hidden"], ["real"]),
          onnx.helper.make_node("Identity", ["hidden"], ["imag"])],
         "head",
         [onnx.helper.make_tensor_value_info("hidden", onnx.TensorProto.FLOAT,
                                             [1, 80, None])],
         [onnx.helper.make_tensor_value_info("real", onnx.TensorProto.FLOAT,
                                             [1, 80, None]),
          onnx.helper.make_tensor_value_info("imag", onnx.TensorProto.FLOAT,
                                             [1, 80, None])],
         tmp_path / "head.onnx")

    chain = f"{tmp_path / 'backbone.onnx'}+{tmp_path / 'head.onnx'}"
    eng = OnnxEngine(chain, use_dml=False)
    assert eng.kind == "mel2stft" and eng.complex_pair
    x = _sweep(44_100, 0.2)
    out = eng.process(x, 44_100, 48_000, get_preset("reference"))
    assert np.isfinite(out.audio).all() and out.audio.shape[-1] > 0


@pytest.mark.skipif(
    not engine_status()["onnx"].startswith("ready"),
    reason="onnxruntime not installed",
)
def test_onnx_patch_denoiser_runs(tmp_path):
    """Fixed patch models like [1, 2, 63, 257] (time-first) run as a
    sliding window and resynthesize audio."""
    from audio_upscaler.engines.onnx_engine import OnnxEngine

    m = tmp_path / "denoiser.onnx"
    _make_identity_model(m, [1, 2, 63, 257], "spec_ri")
    # rename output and keep shape
    onnx = pytest.importorskip("onnx")
    model = onnx.load(str(m))
    model.graph.output[0].name = "enhanced_ri"
    model.graph.node[0].output[0] = "enhanced_ri"
    onnx.save(model, str(m))

    eng = OnnxEngine(m, use_dml=False)
    assert eng.patch_mode and eng.kind == "stft2stft"
    x = _sweep(22_050, 0.2)
    out = eng.process(x, 22_050, 48_000, get_preset("reference"))
    assert np.isfinite(out.audio).all() and out.audio.shape[-1] > 0


def test_force_stop_mid_file(tmp_path):
    """Force stop aborts the current file mid-way and marks it cancelled."""
    x = _sweep(22_050, 0.6)
    src = tmp_path / "f.wav"
    write_audio(src, x, 22_050, fmt="WAV", bit_depth=16)
    settings = JobSettings(target_sr=48_000, fmt="FLAC", out_dir=tmp_path,
                           preset="music", save_spectrograms=False,
                           copy_metadata=False)
    force = threading.Event()
    calls = {"n": 0}

    def prog(i, total, frac, msg, path):
        calls["n"] += 1
        if calls["n"] >= 2:
            force.set()

    summary = run_batch([src], settings, progress=prog, force=force)
    assert summary.cancelled
    assert len(summary.reports) == 1 and summary.reports[0].cancelled
    assert summary.reports[0].message == "stopped"


def test_provider_summary():
    s = summarize()
    assert isinstance(s.gpus, list)


# ---------------------------------------------------------------- v0.2
def test_declip_restores_headroom():
    sr = 44_100
    t = np.arange(sr, dtype=np.float64) / sr
    x = (np.sin(2 * np.pi * 440 * t) * 1.5).astype(np.float32)  # hard clip
    assert (np.abs(x) >= 0.999).mean() > 0.2
    y = dsp.declip(x[None, :], sr)
    assert (np.abs(y) >= 0.999).mean() < 0.01
    assert np.isfinite(y).all()


def test_spectral_gate_reduces_noise_during_silence():
    sr = 22_050
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(sr) * 0.02).astype(np.float32)
    x = noise.copy()
    x[sr // 2:] += _sweep(sr, 0.5)[: sr - sr // 2] * 0.5  # noise then signal
    y = dsp.spectral_gate(x[None, :], sr, strength=2.0)[0]
    q0 = float(np.sqrt(np.mean(x[: sr // 4] ** 2)))
    q1 = float(np.sqrt(np.mean(y[: sr // 4] ** 2)))
    assert q1 < q0 * 0.7  # noise floor pushed down
    s0 = float(np.sqrt(np.mean(x[-sr // 4:] ** 2)))
    s1 = float(np.sqrt(np.mean(y[-sr // 4:] ** 2)))
    assert s1 > s0 * 0.5  # signal mostly preserved


def test_mid_side_roundtrip():
    x = np.random.default_rng(1).standard_normal((2, 1000)).astype(np.float32)
    ms = dsp.to_mid_side(x)
    back = dsp.from_mid_side(ms)
    assert np.allclose(x, back, atol=1e-6)


def test_name_template_and_keep_tree(tmp_path):
    from audio_upscaler.pipeline import collect_audio_with_roots

    src_root = tmp_path / "in"
    sub = src_root / "album"
    sub.mkdir(parents=True)
    x = _sweep(22_050, 0.2)
    src = sub / "track.wav"
    write_audio(src, x, 22_050, fmt="WAV", bit_depth=16)

    files, roots = collect_audio_with_roots([src_root])
    assert files == [src] and roots[src] == src_root

    settings = JobSettings(
        target_sr=48_000, fmt="WAV", out_dir=tmp_path / "out",
        keep_tree=True, name_template="{name}_{sr}_{preset}",
        preset="voice", save_spectrograms=False,
    )
    summary = run_batch(files, settings, roots=roots)
    assert summary.succeeded == 1
    out = summary.reports[0].output
    assert out.name == "track_48000_voice.wav"
    assert out.parent == tmp_path / "out" / "album"


def test_user_presets_roundtrip():
    from audio_upscaler.presets import Preset, all_presets, save_user_preset

    key = "test_custom_zz"
    save_user_preset(Preset(key=key, label="T", description="d", amount=0.5,
                            exciter=0.1, exciter_drive=2.0, air=0.01,
                            hf_cap_hz=16_000, slope_db_per_oct=10.0))
    assert key in all_presets()


def test_keep_tree_and_requirements_text_exist():
    from audio_upscaler.recommendations import requirements_text

    txt = requirements_text()
    for word in ("DirectML", "CUDA", "ROCm", "XPU"):
        assert word in txt


# ---------------------------------------------------------------- v0.3
def test_skip_existing_and_keep_source_rate(tmp_path):
    """--skip-existing reuses existing outputs; --sr 0 keeps the source rate."""
    x = _sweep(22_050, 0.3)
    src = tmp_path / "sk.wav"
    write_audio(src, x, 22_050, fmt="WAV", bit_depth=16)
    settings = JobSettings(target_sr=48_000, fmt="WAV", out_dir=tmp_path,
                           preset="music", save_spectrograms=False,
                           copy_metadata=False)
    first = run_batch([src], settings)
    out = first.reports[0].output
    assert out is not None and out.exists()

    # second run skips it and does not create a _2 duplicate
    settings2 = JobSettings(target_sr=48_000, fmt="WAV", out_dir=tmp_path,
                            preset="music", save_spectrograms=False,
                            copy_metadata=False, skip_existing=True)
    second = run_batch([src], settings2)
    rep = second.reports[0]
    assert rep.ok and rep.message == "skipped (output exists)"
    assert rep.output == out
    assert not (out.parent / f"{out.stem}_2{out.suffix}").exists()

    # target_sr=0 keeps the source rate
    settings3 = JobSettings(target_sr=0, fmt="WAV",
                            out_dir=tmp_path / "keep", preset="voice",
                            save_spectrograms=False, copy_metadata=False)
    third = run_batch([src], settings3)
    assert third.succeeded == 1
    _, sr_out = read_audio(third.reports[0].output)
    assert sr_out == 22_050


def test_split_models_separators():
    from audio_upscaler.engines.onnx_engine import split_models

    parts = split_models("a.onnx+b.onnx, c.onnx;d.onnx")
    assert [p.name for p in parts] == ["a.onnx", "b.onnx", "c.onnx", "d.onnx"]
    assert split_models("solo.onnx")[0].name == "solo.onnx"


@pytest.mark.skipif(
    not engine_status()["onnx"].startswith("ready"),
    reason="onnxruntime not installed",
)
def test_onnx_stereo_waveform_model(tmp_path):
    """[B, 2, T] waveform models get both channels at once."""
    from audio_upscaler.engines.onnx_engine import OnnxEngine

    m = tmp_path / "stereo.onnx"
    _make_identity_model(m, [1, 2, None], "stereo_wave")
    eng = OnnxEngine(m, use_dml=False)
    assert eng.kind == "waveform" and eng.stereo_in

    x = np.stack([_sweep(22_050, 0.2), _sweep(22_050, 0.2, 300.0, 6000.0)])
    out = eng.process(x, 22_050, 48_000, get_preset("reference"))
    assert out.audio.shape[0] == 2 and np.isfinite(out.audio).all()

    mono = eng.process(x[:1], 22_050, 48_000, get_preset("reference"))
    assert mono.audio.shape[0] == 1 and np.isfinite(mono.audio).all()


@pytest.mark.skipif(
    not engine_status()["onnx"].startswith("ready"),
    reason="onnxruntime not installed",
)
def test_onnx_aux_output_not_treated_as_pair(tmp_path):
    """A waveform output plus an auxiliary output must stay 'waveform'."""
    from audio_upscaler.engines.onnx_engine import OnnxEngine

    onnx = pytest.importorskip("onnx")
    x = onnx.helper.make_tensor_value_info("wave", onnx.TensorProto.FLOAT,
                                           [1, 1, None])
    w = onnx.helper.make_tensor_value_info("wave_out", onnx.TensorProto.FLOAT,
                                           [1, 1, None])
    a = onnx.helper.make_tensor_value_info("aux", onnx.TensorProto.FLOAT,
                                           [1, 8])
    nodes = [onnx.helper.make_node("Identity", ["wave"], ["wave_out"]),
             onnx.helper.make_node("ReduceMean", ["wave"], ["aux"],
                                   keepdims=0)]
    graph = onnx.helper.make_graph(nodes, "g", [x], [w, a])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.ir_version = 8
    m = tmp_path / "aux.onnx"
    onnx.save(model, str(m))

    eng = OnnxEngine(m, use_dml=False)
    assert eng.kind == "waveform" and not eng.complex_pair


@pytest.mark.skipif(
    not engine_status()["onnx"].startswith("ready"),
    reason="onnxruntime not installed",
)
def test_onnx_metadata_props_read(tmp_path):
    """Exporters can embed sidecar values in ONNX metadata_props."""
    from audio_upscaler.engines.onnx_engine import OnnxEngine

    onnx = pytest.importorskip("onnx")
    m = tmp_path / "meta.onnx"
    _make_identity_model(m, [1, 1, None], "wave")
    model = onnx.load(str(m))
    model.metadata_props.add(key="sample_rate", value="22050")
    onnx.save(model, str(m))

    eng = OnnxEngine(m, use_dml=False)
    assert eng.sidecar.get("sample_rate") == 22_050


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

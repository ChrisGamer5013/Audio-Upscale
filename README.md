# Audio Upscaler HD

High-quality audio super-resolution / bandwidth extension for Windows.

- **Two engines:**
  - **DSP Enhance** — always-available CPU engine: polyphase resampling,
    SBR-style spectral-mirror bandwidth extension, harmonic exciter,
    noise-matched "air" band, EBU R128 loudness normalize, TPDF dither,
    soft limiter.
  - **ONNX (GPU)** — runs your own `.onnx` super-resolution models through
    **DirectML**, so one codebase accelerates on **AMD, Intel and NVIDIA**
    GPUs, with automatic CPU fallback.
- **Formats in:** WAV, FLAC, MP3, OGG, AIFF (+ m4a/aac/wma/opus and video
  audio tracks when ffmpeg is present).
- **Formats out:** WAV (16/24/32-float), FLAC (16/24), MP3, OGG Vorbis,
  OPUS (needs ffmpeg).
- **Presets:** Music / Voice / Restore old recording / Reference (resample only).
- **Extras:** batch queue with drag & drop, per-file progress, before/after
  spectrogram comparison PNGs, GUI *and* CLI, cancelling.

---

## Quick start

```bat
setup_env.bat
run_gui.bat
```

`setup_env.bat` creates a virtual environment, installs all Python
dependencies **and auto-installs ffmpeg** (via winget or a direct download
into `tools\ffmpeg`).

### CLI

```bat
.venv\Scripts\python main.py cli --help
.venv\Scripts\python main.py cli *.wav --sr 48000 --preset music
.venv\Scripts\python main.py cli input.mp3 --engine onnx --model models\mysr.onnx --device 0
.venv\Scripts\python main.py cli --gpu-info
```

### Try it instantly

```bat
.venv\Scripts\python scripts\make_demo.py
.venv\Scripts\python main.py cli demo_lowband.wav --spectrograms
```

---

## Portable / installer (no Python needed)

Download `AudioUpscalerHD-standalone.exe` (or `-portable.zip`) from
[Releases](https://github.com/ChrisGamer5013/Audio-Upscale/releases).
Run it anywhere — on first launch it auto-downloads ffmpeg. Or build it
yourself:

```bat
.venv\Scripts\python scripts\build_portable.py
REM -> dist\AudioUpscalerHD-standalone.exe  (single-file, all-in-one)
REM -> dist\AudioUpscalerHD-portable.zip    (portable folder)
```

## Train your own ONNX model

Pick the training backend for your GPU (each replaces the plain torch):

```bat
REM NVIDIA (CUDA):  pip install -r requirements-train-nvidia.txt
REM AMD (ROCm):     pip install -r requirements-train-amd.txt
REM Intel (XPU):    pip install -r requirements-train-intel.txt
REM any CPU:        pip install -r requirements-train.txt
```

Then:

```bat
.venv\Scripts\python main.py train "D:\path\to\your\music" --epochs 20
REM -> models\upscaler_48000.onnx -- select it in the GUI (engine: ONNX)
```

The trainer auto-detects CUDA (NVIDIA or AMD ROCm builds), XPU (Intel)
and torch-directml, probing each backend and falling back to CPU if
needed (`--device` overrides). Training pairs are generated on the fly by
low-passing your high-quality audio, so any folder of WAV/FLAC works.

---

## Project layout

```
main.py                 entry point (GUI default, `cli` subcommand)
audio_upscaler/
  dsp.py                all DSP building blocks
  audio_io.py           soundfile + ffmpeg I/O
  engines/              dsp_engine.py + onnx_engine.py + registry
  pipeline.py           batch queue, progress, cancellation
  gui/app.py            CustomTkinter interface
  cli.py                argparse CLI
  providers.py          GPU / runtime detection
  presets.py            preset definitions
  spectrogram.py        comparison renderer
scripts/
  install_ffmpeg.py     ffmpeg auto-installer
  make_demo.py          synthetic test file generator
tests/test_engines.py   pytest suite
models/                 drop .onnx models here
```

## Adding an ONNX model

Put a `.onnx` file in `models\` (or browse to it). Expected format:
waveform input `float32 [1, samples]` (or `[1, 1, samples]` / `[1, ch, samples]`);
output same layout. Integer output/input length ratios are handled as
built-in N× resampling. Long files are processed in overlapping chunks.

## Troubleshooting

- **"onnxruntime is not installed"** — re-run `setup_env.bat`; on very new
  Python versions install `onnxruntime-directml` manually.
- **OPUS/m4a unavailable** — ffmpeg install failed; re-run
  `.venv\Scripts\python scripts\install_ffmpeg.py` or install ffmpeg manually.
- **GUI won't start** — use the CLI (`main.py cli ...`); GUI needs
  `customtkinter`, `Pillow` and optionally `tkinterdnd2`.

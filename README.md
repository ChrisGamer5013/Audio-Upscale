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
- **Generic ONNX runner:** waveform, mel→mel (Griffin-Lim), mel→wave vocoders,
  STFT mask/complex — auto-detected; `.json` sidecar for custom STFT/mel
  parameters. VRAM-aware model recommender: `main.py cli --recommend-models`.
- **Extras:** batch queue with drag & drop (incl. dropping `.onnx` models),
  per-file progress, before/after spectrogram PNGs, GUI *and* CLI, cancelling.
- **v0.2 extras:** declipper, spectral-gate denoiser, mid/side mode,
  auto-preset per file, watch-folder mode, parallel batch workers, output
  name templates, folder-structure mirroring, metadata copy, LUFS + true-peak
  report per file, ultra resampler, update checker, settings persistence,
  dark/light themes, diagnostics panel, Explorer right-click integration,
  GitHub CI that auto-builds releases, bundled starter BWE model.

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
.venv\Scripts\python main.py cli --gpu-info | --doctor | --recommend-models
.venv\Scripts\python main.py cli --check-update | --get-starter-model
REM v0.2 flags (all also available in the GUI):
.venv\Scripts\python main.py cli album\ --denoise 1.5 --declip --mid-side --workers 4 --keep-tree
.venv\Scripts\python main.py cli song.flac --compare music,voice,restore
.venv\Scripts\python main.py cli inbox\ --watch 10 -o out\ --auto-preset
.venv\Scripts\python main.py cli music\ --estimate
.venv\Scripts\python main.py cli track.wav --name-template "{name}_{sr}k_{preset}"
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

## System requirements per GPU backend

**Inference (upscaling) — all vendors, zero setup:** DirectML runs on any
DX12-capable GPU: AMD, Intel and NVIDIA all work with just a normal graphics
driver. ONNX models of most layouts work (waveform, mel, STFT, vocoders).

**Training (`main.py train`) — pick one:**

| GPU | What you need installed | Project dependency |
|---|---|---|
| NVIDIA | Any R525+ driver. The pip wheel bundles the CUDA runtime — no CUDA toolkit needed | `pip install -r requirements-train-nvidia.txt` (cu128 index) |
| AMD | RDNA2-4 (RX 6000/7000/9000). ROCm-certified Adrenalin driver 25.x+; HIP SDK 7.x (or use the self-contained wheels — no SDK needed) | `pip install -r requirements-train-amd.txt` (stable.repo.amd.com/rocm) |
| Intel | Arc A/B-series or Core Ultra iGPU. Driver 32.0.101.8801+ (Win 11). Level Zero ships in the wheels — no oneAPI install needed | `pip install -r requirements-train-intel.txt` (download.pytorch.org/whl/xpu) |
| none | — | `pip install -r requirements-train.txt` (CPU) |

Verify GPU training: `python -c "import torch; print(torch.cuda.is_available())"`
(AMD ROCm also answers via `torch.cuda`), Intel: `torch.xpu.is_available()`.
In the GUI: **About → GPU backend requirements** shows this in-app.

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

## Explorer integration (optional)

```bat
.venv\Scripts\python scripts\install_context_menu.py          REM install
.venv\Scripts\python scripts\install_context_menu.py remove   REM uninstall
```

Adds right-click "Upscale with Audio Upscaler HD" for common audio types +
a Send-to entry (per-user registry, no admin).

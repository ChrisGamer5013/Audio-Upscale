# AGENTS.md — Audio Upscaler HD

Project: high-quality audio upscaling / bandwidth extension for Windows.
Language: Python 3.14 (`.venv`). Working dir is the project root.

## Commands

- Setup (first run / after env changes): `setup_env.bat`
  (creates `.venv`, installs `requirements.txt`, auto-installs ffmpeg into
  `tools/ffmpeg` via `scripts/install_ffmpeg.py`)
- GUI: `run_gui.bat`  or  `.venv\Scripts\python main.py`
- CLI: `.venv\Scripts\python main.py cli --help`
  (v0.3 helpers: `--list-models [dir]`, `--inspect-model <path>`,
  `--dry-run`, `--skip-existing`, `--sr 0` = keep source rate,
  `--passes N` = multi-pass cascade; GUI "Passes" entry in the Batch tab)
- GUI model picker: **Browse** (multi-select = chain), **Chain…** appends
  further stages, live hint shows stage list / backbone-needs-head warnings;
- Tests: `.venv\Scripts\python -m pytest tests -q`
- Demo file: `.venv\Scripts\python scripts\make_demo.py`
- Train ONNX model: install one of `requirements-train[-nvidia|-amd|-intel].txt`
  (torch CPU / CUDA / ROCm / XPU), then `main.py train <audio_folder> --epochs N`
  (writes to `models/`; `--device auto|cpu|cuda|xpu|dml`)
- Build portable exe: `.venv\Scripts\python scripts\build_portable.py`
  (PyInstaller onedir + onefile `dist/AudioUpscalerHD-standalone.exe`;
  Inno Setup installer when available)

## Structure

- `main.py` — entry: GUI by default, `cli` subcommand for CLI.
- `audio_upscaler/dsp.py` — pure numpy/scipy DSP primitives
  (resample, bandwidth estimation/extension, exciter, air, loudness,
  dither, limiter). All operate on `(channels, samples)` float32.
- `audio_upscaler/engines/` — `dsp_engine.py` (CPU, always available) and
  `onnx_engine.py` (generic ONNX runner, DirectML/CUDA/CPU). Registry in
  `engines/__init__.py` (`create_engine`, `engine_status`).
- `audio_upscaler/spectral.py` — numpy/scipy STFT, mel filterbanks,
  Griffin-Lim, `SpecParams` (sidecar `<model>.json` parsing).
- `audio_upscaler/recommendations.py` — VRAM-aware curated ONNX model list
  (CLI `--recommend-models`, GUI button) + `requirements_text()` with
  per-GPU system requirements (shown in GUI About tab and README).
- `audio_upscaler/updater.py` — GitHub release check + starter-model
  download (`starter_bwe_48k.onnx` from latest release assets).
- `audio_upscaler/gui/{dialogs,settings_store}.py` — reusable text dialogs;
  GUI prefs/queue persistence in `%APPDATA%/AudioUpscalerHD/settings.json`.
- `scripts/train_starter.py` — synthetic-data training of the bundled model;
  `scripts/install_context_menu.py` — Explorer integration (HKCU only).
- `.github/workflows/` — ci.yml (tests, win+linux) and release.yml
  (on `v*` tag: tests → PyInstaller → attach exes to the release).
- `audio_upscaler/pipeline.py` — `JobSettings`, file discovery, batch runner
  with progress callbacks and `threading.Event` cancellation.
- `audio_upscaler/audio_io.py` — soundfile read/write + ffmpeg fallback.
  Local ffmpeg expected at `tools/ffmpeg/bin/ffmpeg.exe` (checked first,
  then PATH).
- `audio_upscaler/gui/app.py` — CustomTkinter dark UI, drag & drop via
  `tkinterdnd2` when installed.
- `audio_upscaler/cli.py`, `providers.py` (GPU detection via WMI +
  onnxruntime EPs), `presets.py`, `spectrogram.py` (matplotlib Agg).
- `audio_upscaler/training.py` + `train_cli.py` — PyTorch training of a
  small dilated-conv super-res net, exports dynamic-time-axis ONNX. Torch is
  an optional dep (`requirements-train.txt`); guard with lazy import.
- `audio_upscaler/bootstrap.py` — ffmpeg auto-installer; also called on
  first run of the frozen exe. `scripts/install_ffmpeg.py` wraps it.
- `models/` — user-supplied `.onnx` models (see `models/README.md`).

## Conventions

- GUI must expose every CLI feature (user requirement): new flags get a GUI
  control in the same change; add prefs to `_PREF_VARS`/`_PREF_SWITCHES`.
- Keep GUI edits small and verify with
  `python -c "from audio_upscaler.gui.app import App; a=App(); a.update(); a.destroy()"`.

- Audio arrays are always `(channels, samples)` float32 internally;
  `dsp.ensure_channels` normalizes.
- Never write above 0 dBFS without going through `dsp.soft_limit`.
- DSP engine must stay pure-CPU and dependency-light; GPU code lives only in
  `onnx_engine.py` behind a guarded `import onnxruntime`.
- ffmpeg must remain optional: core WAV/FLAC/MP3/OGG workflows work without
  it; check via `audio_io.ffmpeg_path()`; frozen builds look in
  `<exe_dir>/tools/ffmpeg/bin` first (bootstrap installs there).
- Progress UX: engines report `progress(frac, message)`; pipeline maps to
  `(file_index, total, frac, message, path)`; GUI polls a `queue.Queue`.
- Adding dependencies → update `requirements.txt` **and this file** if the
  workflow changes.
- Tests must not require a GPU, ffmpeg, network, or a GUI display.

### ONNX engine model kinds

`onnx_engine.OnnxEngine` auto-detects I/O layout from tensor shapes/names:
`waveform` | `mel2mel` (Griffin-Lim back-end) | `mel2wave` | `stft2stft`
(mask or 2-channel complex) | `stft2wave` | `generic`. Override/params via
`<model>.json` sidecar (see `models/README.md`). Every model gets a probe
inference at construction — failures must produce a clear, actionable error.

## Notes

- Python 3.14: if `onnxruntime-directml` wheels are missing, everything
  except the ONNX engine still works; the engine reports "unavailable".

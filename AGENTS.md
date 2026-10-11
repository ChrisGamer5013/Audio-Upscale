# AGENTS.md — Audio Upscaler HD

Project: high-quality audio upscaling / bandwidth extension for Windows.
Language: Python 3.14 (`.venv`). Working dir is the project root.

## Commands

- Setup (first run / after env changes): `setup_env.bat`
  (creates `.venv`, installs `requirements.txt`, auto-installs ffmpeg into
  `tools/ffmpeg` via `scripts/install_ffmpeg.py`)
- GUI: `run_gui.bat`  or  `.venv\Scripts\python main.py`
- CLI: `.venv\Scripts\python main.py cli --help`
- Tests: `.venv\Scripts\python -m pytest tests -q`
- Demo file: `.venv\Scripts\python scripts\make_demo.py`

## Structure

- `main.py` — entry: GUI by default, `cli` subcommand for CLI.
- `audio_upscaler/dsp.py` — pure numpy/scipy DSP primitives
  (resample, bandwidth estimation/extension, exciter, air, loudness,
  dither, limiter). All operate on `(channels, samples)` float32.
- `audio_upscaler/engines/` — `dsp_engine.py` (CPU, always available) and
  `onnx_engine.py` (DirectML: AMD/Intel/NVIDIA + CPU fallback). Registry in
  `engines/__init__.py` (`create_engine`, `engine_status`).
- `audio_upscaler/pipeline.py` — `JobSettings`, file discovery, batch runner
  with progress callbacks and `threading.Event` cancellation.
- `audio_upscaler/audio_io.py` — soundfile read/write + ffmpeg fallback.
  Local ffmpeg expected at `tools/ffmpeg/bin/ffmpeg.exe` (checked first,
  then PATH).
- `audio_upscaler/gui/app.py` — CustomTkinter dark UI, drag & drop via
  `tkinterdnd2` when installed.
- `audio_upscaler/cli.py`, `providers.py` (GPU detection via WMI +
  onnxruntime EPs), `presets.py`, `spectrogram.py` (matplotlib Agg).
- `models/` — user-supplied `.onnx` models (see `models/README.md`).

## Conventions

- Audio arrays are always `(channels, samples)` float32 internally;
  `dsp.ensure_channels` normalizes.
- Never write above 0 dBFS without going through `dsp.soft_limit`.
- DSP engine must stay pure-CPU and dependency-light; GPU code lives only in
  `onnx_engine.py` behind a guarded `import onnxruntime`.
- ffmpeg must remain optional: core WAV/FLAC/MP3/OGG workflows work without
  it; check via `audio_io.ffmpeg_path()`.
- Progress UX: engines report `progress(frac, message)`; pipeline maps to
  `(file_index, total, frac, message, path)`; GUI polls a `queue.Queue`.
- Adding dependencies → update `requirements.txt` **and this file** if the
  workflow changes.
- Tests must not require a GPU, ffmpeg, network, or a GUI display.

## Notes

- Python 3.14: if `onnxruntime-directml` wheels are missing, everything
  except the ONNX engine still works; the engine reports "unavailable".

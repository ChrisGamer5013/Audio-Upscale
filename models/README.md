# models/

Drop `.onnx` audio models here. The engine **auto-detects the model's I/O
layout** and runs it on GPU via DirectML (AMD / Intel / NVIDIA) with CPU
fallback. Long files are processed in overlapping chunks with crossfading.

## Supported model kinds (auto-detected)

| kind | model input/output | examples |
|---|---|---|
| `waveform` | `[B, T]` / `[B, 1, T]` → same | super-res nets, our own trainer output |
| `mel2mel` | `[B, n_mels, F]` → `[B, n_mels, F']` | AudioSR, VoiceFixer, NVSR, Apollo |
| `mel2wave` | `[B, n_mels, F]` → `[B, 1, T]` | HiFi-GAN / vocoders |
| `stft2stft` | `[B, F, T]` → mask or `[B, 2, F, T]` complex | NSNet2, DTLN, DeepFilterNet |
| `stft2wave` | `[B, F, T]` → `[B, 1, T]` | spectral decoders |

Detection uses tensor names (`mel`, `spec`, `noisy`…) and bin counts
(64/80/128 mel, 257/513/1025 STFT…) and works with symbolic batch dims
(`['batch', 80, 'frames']`). Outputs that are a **(real, imag) pair**
(two output tensors) are rebuilt into audio through a complex ISTFT.

**Fixed-size patch models** (`[1, 2, 63, 257]` etc., both `[.., F, T]` and
`[.., T, F]` layouts) are run as a sliding window over the whole spectrogram
and crossfaded back together - denoisers like `denoiser_core.onnx` work
out of the box.

If detection is wrong or your model needs non-default analysis parameters,
add a sidecar file named like the model, e.g. `mymodel.json` next to
`mymodel.onnx`:

```json
{
  "kind": "mel2mel",
  "sample_rate": 44100,
  "n_fft": 2048, "hop": 512, "win": 2048,
  "n_mels": 128, "fmin": 0, "fmax": null,
  "power": 1, "log": "log10", "top_db": 80,
  "gl_iters": 24,
  "chunk_frames": 300, "overlap_frames": 64,
  "hop": 128
}
```

- `mel2mel` models get audio back via Griffin-Lim (`gl_iters`).
- `stft2stft` outputs with two channels are treated as complex audio;
  single-channel outputs are applied as time-frequency masks.
- Extra (secondary) model inputs are fed zeros of matching static shape —
  best-effort for stateful/LSTM models.
- For patch models `hop` sets the STFT hop when resynthesizing
- **Stereo waveform models** with input `[B, 2, T]` receive both channels at
  once; mono files are duplicated and the model output is downmixed.
- Waveform models with an **auxiliary second output** (e.g. `[wave, mask]`)
  are still treated as waveform — only real/imag-style pairs become complex
  spectrogram output.
- Many exporters can embed sidecar values directly in the model
  (`metadata_props`, e.g. `sample_rate`, `kind`, `n_fft`). A `<model>.json`
  sidecar always overrides embedded metadata.

Not sure what a model expects? Ask the engine:

```bat
main.py cli --list-models                 REM kind of every model in models\
main.py cli --inspect-model path\to.onnx  REM full I/O report + sidecar template
```

In the GUI, the buttons under the model picker do the same: **Info** (I/O
report), **Test** (2 s synthetic benchmark, reports kind + realtime speed),
**Sidecar** (creates `<model>.json` from a template and opens it), and
**Folder** (opens the models directory).

## Chaining split models (backbone + head)

Many projects split a vocoder into stages, e.g.:

- `backbone.onnx`: `mel [B, 80, F]` → `hidden [B, F, 512]`
- `spec_head.onnx`: `hidden [B, F, 512]` → `real [B, Fn, F]`, `imag [B, Fn, F]`

Join them with **`+`** (or `,` / `;`) anywhere a model path is accepted:

```bat
main.py cli song.wav --engine onnx --model "backbone.onnx+spec_head.onnx"
```

In the GUI: select/drop both files (multiple = chain), or type the two
paths joined with `+` in the model box. The engine feeds every output of
one stage into the matching input of the next stage, and the *overall*
kind is derived from the first input / last output.

Running a backbone **alone** is rejected with a hint to chain it — a
latent model alone cannot output audio. Sidecars from every stage are
merged (later stages override on conflicts).

## GPU (DirectML) notes

The ONNX engine uses DirectML first (AMD / Intel / NVIDIA). Some large
models compile *very* slowly on old GPU drivers - if loading seems stuck,
disable GPU:
- GUI: uncheck *GPU acceleration (DirectML)*
- CLI: `--no-gpu`
- env: set `AU_NO_DML=1`

Need ideas? Run the recommender (VRAM-aware):

```bat
main.py cli --recommend-models
```

Or train your own waveform model (always compatible):

```bat
main.py train <folder_with_good_audio> --epochs 20
```

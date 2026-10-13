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
(64/80/128 mel, 257/513/1025 STFT…). If detection is wrong or your model
needs non-default analysis parameters, add a sidecar file named like the
model, e.g. `mymodel.json` next to `mymodel.onnx`:

```json
{
  "kind": "mel2mel",
  "sample_rate": 44100,
  "n_fft": 2048, "hop": 512, "win": 2048,
  "n_mels": 128, "fmin": 0, "fmax": null,
  "power": 1, "log": "log10", "top_db": 80,
  "gl_iters": 24,
  "chunk_frames": 300, "overlap_frames": 64
}
```

- `mel2mel` models get audio back via Griffin-Lim (`gl_iters`).
- `stft2stft` outputs with two channels are treated as complex audio;
  single-channel outputs are applied as time-frequency masks.
- Extra (secondary) model inputs are fed zeros of matching static shape —
  best-effort for stateful/LSTM models.

Need ideas? Run the recommender (VRAM-aware):

```bat
main.py cli --recommend-models
```

Or train your own waveform model (always compatible):

```bat
main.py train <folder_with_good_audio> --epochs 20
```

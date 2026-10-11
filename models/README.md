# models/

Drop your own `.onnx` audio super-resolution models here. They will appear in
the GUI / CLI model picker automatically.

## Expected model format (ONNX engine)

- **Input**: waveform tensor, `float32`, shape `[1, samples]`, `[1, 1, samples]`
  or `[batch, channels, samples]` (batch 1 / mono or stereo chunks are handled).
- **Output**: tensor of the same layout. If the output length is an integer
  multiple of the input length, the model is treated as doing *xN resampling*
  internally (e.g. a 22.05 kHz -> 44.1 kHz model has ratio 2).
- Variable-length (`None` / dynamic) sample axes are fully supported; long
  files are processed in overlapping chunks with Hann crossfading.

## Acceleration

On Windows the engine runs through **DirectML**, so one model executes on
**AMD, Intel and NVIDIA** GPUs alike, with automatic CPU fallback.

Good candidates to convert to ONNX: AudioSR, NVSR, or your own trained
waveform-to-waveform models (`torch.onnx.export` with a dynamic time axis).

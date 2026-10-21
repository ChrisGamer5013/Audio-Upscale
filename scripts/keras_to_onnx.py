"""Convert a .keras / .h5 model to ONNX with tf2onnx.

ONNX is the recommended format for this app: it runs on DirectML (AMD / Intel
/ NVIDIA), CUDA or CPU, supports sidecars and multi-stage bundles.  TensorFlow
is only needed for the conversion, not for inference afterwards.

Usage (from the project root):
    .venv\\Scripts\\pip install -r requirements-keras.txt
    .venv\\Scripts\\python scripts\\keras_to_onnx.py models\\mymodel.keras
    .venv\\Scripts\\python main.py cli --engine onnx --model models\\mymodel.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path, help=".keras or .h5 model file")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output .onnx path (default: next to the source)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args(argv)

    try:
        import tensorflow as tf
        import tf2onnx
    except Exception as exc:
        print("This helper needs TensorFlow and tf2onnx:\n"
              "  pip install -r requirements-keras.txt\n"
              f"Import error: {exc}")
        return 1

    src: Path = args.model
    if not src.is_file():
        print(f"Model not found: {src}")
        return 1
    if src.suffix.lower() not in (".keras", ".h5"):
        print(f"Expected a .keras or .h5 file, got: {src.name}")
        return 1
    out = args.out or src.with_suffix(".onnx")

    print(f"Loading {src.name} ...")
    try:
        model = tf.keras.models.load_model(str(src))
    except Exception as exc:
        print(f"Could not load the Keras model: {exc}")
        print("Custom layers need to be available on PYTHONPATH before "
              "loading; otherwise re-save the model in a plain TF session.")
        return 1

    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list) and shape and isinstance(shape[0], (list, tuple)):
        print("Multi-input models are not supported by this helper.")
        return 1
    if not shape:
        print("The model does not expose input_shape; cannot build a signature.")
        return 1
    dims = [None if d is None else int(d) for d in shape]
    spec = [tf.TensorSpec(dims, tf.float32, name="audio")]
    print(f"input shape: {dims}\nConverting to {out} ...")
    try:
        tf2onnx.convert.from_keras(model, input_signature=spec,
                                   opset=args.opset, output_path=str(out))
    except Exception as exc:
        print(f"Conversion failed: {exc}")
        return 1
    print(f"Wrote {out}")
    print("Tip: use --inspect-model to check the I/O layout, and add a "
          f"'{out.stem}.json' sidecar if the engine needs kind/params hints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

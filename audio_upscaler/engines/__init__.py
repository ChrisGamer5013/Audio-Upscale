"""Engine registry."""
from __future__ import annotations

from ..providers import onnx_provider_names
from .base import Engine, ProcessResult
from .dsp_engine import DspEngine
from .keras_engine import KerasEngine, keras_status
from .onnx_engine import OnnxEngine

ENGINE_IDS = ("dsp", "onnx", "keras")


def engine_status() -> dict[str, str]:
    """Engine availability for the UI."""
    status = {"dsp": "ready (CPU, always available)"}
    if onnx_provider_names():
        accel = " + ".join(
            p.replace("ExecutionProvider", "")
            for p in onnx_provider_names()
            if p != "CPUExecutionProvider"
        )
        status["onnx"] = f"ready ({accel or 'CPU'})"
    else:
        status["onnx"] = "unavailable (onnxruntime not installed)"
    status["keras"] = keras_status()
    return status


def create_engine(engine_id: str, model_path: str | None = None,
                  device_id: int = 0, use_dml: bool = True) -> Engine:
    if engine_id == "onnx":
        if not model_path:
            raise ValueError("The ONNX engine requires a .onnx model file.")
        return OnnxEngine(model_path=model_path, device_id=device_id,
                          use_dml=use_dml)
    if engine_id == "keras":
        if not model_path:
            raise ValueError("The Keras engine requires a .keras or .h5 model "
                             "file.")
        return KerasEngine(model_path=model_path, device_id=device_id,
                           use_dml=use_dml)
    return DspEngine()


__all__ = [
    "ENGINE_IDS", "Engine", "ProcessResult", "DspEngine", "OnnxEngine",
    "KerasEngine", "create_engine", "engine_status", "keras_status",
]

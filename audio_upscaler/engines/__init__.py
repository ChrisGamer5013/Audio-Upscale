"""Engine registry."""
from __future__ import annotations

from ..providers import onnx_provider_names
from .base import Engine, ProcessResult
from .dsp_engine import DspEngine
from .onnx_engine import OnnxEngine

ENGINE_IDS = ("dsp", "onnx")


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
    return status


def create_engine(engine_id: str, model_path: str | None = None,
                  device_id: int = 0) -> Engine:
    if engine_id == "onnx":
        if not model_path:
            raise ValueError("The ONNX engine requires a .onnx model file.")
        return OnnxEngine(model_path=model_path, device_id=device_id)
    return DspEngine()


__all__ = [
    "ENGINE_IDS", "Engine", "ProcessResult", "DspEngine", "OnnxEngine",
    "create_engine", "engine_status",
]

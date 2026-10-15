"""Hardware / execution-provider detection (AMD, Intel, NVIDIA)."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


@dataclass
class GpuInfo:
    name: str
    vendor: str  # "NVIDIA" | "AMD" | "Intel" | "Other"


@dataclass
class ProviderSummary:
    gpus: list[GpuInfo] = field(default_factory=list)
    onnxruntime_available: bool = False
    onnx_providers: list[str] = field(default_factory=list)
    directml: bool = False
    cuda: bool = False
    ffmpeg: bool = False


def _classify_vendor(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n:
        return "NVIDIA"
    if "amd" in n or "radeon" in n:
        return "AMD"
    if "intel" in n or "arc" in n or "iris" in n or "uhd" in n:
        return "Intel"
    return "Other"


@lru_cache(maxsize=1)
def list_gpus() -> list[GpuInfo]:
    """Enumerate display adapters via WMI (Windows), best effort elsewhere."""
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=_CREATE_NO_WINDOW,
        )
        names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return [GpuInfo(name=n, vendor=_classify_vendor(n)) for n in names]
    except Exception:
        return []


@lru_cache(maxsize=1)
def onnx_provider_names() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _ffmpeg_available() -> bool:
    if shutil.which("ffmpeg"):
        return True
    local = (
        Path(__file__).resolve().parent.parent
        / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    )
    return local.is_file()


def summarize() -> ProviderSummary:
    providers = onnx_provider_names()
    return ProviderSummary(
        gpus=list_gpus(),
        onnxruntime_available=bool(providers),
        onnx_providers=providers,
        directml="DmlExecutionProvider" in providers,
        cuda="CUDAExecutionProvider" in providers,
        ffmpeg=_ffmpeg_available(),
    )


def diagnostics() -> str:
    """Everything useful for a bug report, in one text blob."""
    import platform
    import sys

    lines = [f"Python {sys.version.split()[0]} on {platform.platform()}"]
    for mod in ("numpy", "scipy", "soundfile", "onnxruntime", "customtkinter",
                "matplotlib", "pyloudnorm"):
        try:
            m = __import__(mod)
            lines.append(f"{mod} {getattr(m, '__version__', '?')}")
        except ImportError:
            lines.append(f"{mod} not installed")
    s = summarize()
    lines.append("GPUs:")
    for g in s.gpus:
        lines.append(f"  - {g.vendor}: {g.name}")
    lines.append(f"ONNX providers: {', '.join(s.onnx_providers) or 'none'}")
    lines.append(f"ffmpeg: {_ffmpeg_available()}")
    try:
        from .engines import engine_status

        for k, v in engine_status().items():
            lines.append(f"engine {k}: {v}")
    except Exception as exc:
        lines.append(f"engine status failed: {exc}")
    return "\n".join(lines)


def describe(summary: ProviderSummary | None = None) -> str:
    """One-paragraph human-readable hardware summary for UI/CLI."""
    s = summary or summarize()
    lines: list[str] = []
    if s.gpus:
        for g in s.gpus:
            lines.append(f"{g.vendor}: {g.name}")
    else:
        lines.append("No discrete GPU detected (CPU only)")
    if s.onnxruntime_available:
        accel = []
        if s.directml:
            accel.append("DirectML (AMD/Intel/NVIDIA)")
        if s.cuda:
            accel.append("CUDA")
        lines.append(
            "ONNX Runtime: " + (", ".join(accel) if accel else "CPU only")
        )
    else:
        lines.append("ONNX Runtime not installed - DSP engine only")
    lines.append("ffmpeg: " + ("found" if s.ffmpeg else "not found (MP3/FLAC/OGG still fine)"))
    return "\n".join(lines)

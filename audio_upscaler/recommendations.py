"""Curated ONNX-exportable model recommendations for common tasks, with
rough VRAM budgets. Combined with detected GPU VRAM to suggest what fits.

NOTE: VRAM figures are practical estimates for real-time-ish use of the
full pipeline; smaller model variants often exist.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


@dataclass(frozen=True)
class ModelRec:
    name: str
    purpose: str          # e.g. "denoise", "bandwidth extension"
    io_kind: str          # waveform | mel2mel | stft2stft | mel2wave | stft2wave
    min_vram_gb: float
    link: str
    notes: str = ""


RECOMMENDATIONS: list[ModelRec] = [
    ModelRec(
        "NSNet2 (Microsoft)", "speech denoise",
        "stft2stft", 0.5,
        "https://github.com/microsoft/DNS-Challenge/tree/master/NSNet2-baseline",
        "official ONNX model ships in the repo; needs a .json sidecar "
        "with {\"kind\": \"stft2stft\", \"n_fft\": 640, \"hop\": 320}"
    ),
    ModelRec(
        "DTLN", "speech denoise", "stft2stft", 0.5,
        "https://github.com/breizhn/DTLN",
        "Keras model; convert to ONNX; mask-style stft2stft input"
    ),
    ModelRec(
        "DeepFilterNet 3", "speech denoise + voice restoration",
        "stft2stft", 1.5,
        "https://github.com/Rikorose/DeepFilterNet",
        "has official ONNX export (df/export.py); complex-complex mapping – "
        "use sidecar n_fft=960 hop=480",
    ),
    ModelRec(
        "HiFi-GAN SR (bandwidth vocoder)", "bandwidth extension 16->44.1 kHz",
        "mel2wave", 2.0,
        "https://github.com/jik876/hifi-gan",
        "export vocoder to ONNX; sidecar: mel 80, hop 256, fmax 8000"
    ),
    ModelRec(
        "VoiceFixer", "voice restoration (declip + BWE + denoise)",
        "mel2mel", 2.0,
        "https://github.com/haoheliu/voicefixer",
        "mel128 front-end + UNIVNET vocoder; export both to ONNX; moderate size"
    ),
    ModelRec(
        "AudioSR", "general audio super-resolution to 48 kHz (music+voice)",
        "mel2mel", 4.0,
        "https://github.com/haoheliu/versatile_audio_super_resolution",
        "Latent-diffusion mel model; exportable to ONNX (needs conversion effort);"
        " pair output mel with a 48k vocoder",
    ),
    ModelRec(
        "NVSR (NVIDIA)", "speech super-resolution 16->48 kHz",
        "mel2mel", 6.0,
        "https://github.com/NVIDIA/NeMo",
        "Mel-bandwidth-extension Transformer; export via NeMo; heavy on VRAM"
    ),
    ModelRec(
        "Apollo", "music restoration / lossy-codec repair",
        "mel2mel", 6.0,
        "https://github.com/JusperLee/Apollo",
        "strong on music; export mel model + use 44.1k vocoder front-end"
    ),
    ModelRec(
        "Versal/D3Small BWE (waveform)", "bandwidth extension, real-time budget",
        "waveform", 0.8,
        "custom: train with  main.py train <folder> --sr 48000",
        "our built-in tiny trainer exports a waveform ONNX directly compatible",
    ),
]


@lru_cache(maxsize=1)
def gpu_vram_map() -> list[tuple[str, float]]:
    """Best-effort GPU name + VRAM (GB) via WMI."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=15,
            creationflags=_CREATE_NO_WINDOW)
        import json as _json
        data = _json.loads(out.stdout or "[]")
        if isinstance(data, dict):
            data = [data]
        res = []
        for d in data:
            name = str(d.get("Name", "GPU"))
            ram = d.get("AdapterRAM") or 0
            # AdapterRAM is bytes (uint); some drivers cap at 4 GiB
            gb = round(int(ram) / (1024 ** 3), 1) if ram else 0.0
            if gb == 4.0:
                gb = 0.0   # unreliable cap marker -> unknown
            res.append((name, gb))
        return res
    except Exception:
        return []


def total_vram_gb() -> float:
    v = [g for _, g in gpu_vram_map() if g]
    return max(v) if v else 0.0


def fits(rec: ModelRec, vram_gb: float) -> bool:
    if vram_gb <= 0:
        return True   # unknown VRAM -> show everything
    return rec.min_vram_gb <= vram_gb


def recommendation_report() -> str:
    gpus = gpu_vram_map()
    vram = total_vram_gb()
    lines = []
    if gpus:
        for name, gb in gpus:
            gb_txt = f"{gb:g} GB VRAM" if gb else "VRAM unknown"
            lines.append(f"  {name}  ({gb_txt})")
    else:
        lines.append("  No GPU info available")
    lines.append("")
    for i, r in enumerate(RECOMMENDATIONS, 1):
        mark = "fits" if fits(r, vram) else "needs more VRAM"
        lines.append(
            f"{i}. {r.name} — {r.purpose}\n"
            f"   I/O: {r.io_kind} | ~{r.min_vram_gb:g} GB VRAM | {mark}\n"
            f"   {r.link}\n"
            f"   {r.notes}\n"
        )
    return "\n".join(lines)

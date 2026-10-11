"""Audio file I/O.

Reads: WAV/FLAC/MP3/OGG/AIFF via soundfile. Anything else (m4a/aac/wma/opus,
audio tracks inside video files) via ffmpeg *if* it is on PATH.

Writes: WAV (16/24/32f), FLAC (16/24), MP3, OGG Vorbis via soundfile;
OPUS via ffmpeg when available.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .dsp import ensure_channels, tpdf_dither

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

INPUT_EXTENSIONS = {
    ".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac",
    ".wma", ".aiff", ".aif", ".mp4", ".mka", ".mkv",
}

# format -> {bit_depth: soundfile subtype}
_OUTPUT_SUBTYPES: dict[str, dict[int, str]] = {
    "WAV": {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"},
    "FLAC": {16: "PCM_16", 24: "PCM_24"},
    "MP3": {},
    "OGG": {},
    "OPUS": {},
}
_LOSSY = {"MP3", "OGG", "OPUS"}


@dataclass(frozen=True)
class OutputFormat:
    name: str
    extension: str
    bit_depths: tuple[int, ...]
    needs_ffmpeg: bool = False


OUTPUT_FORMATS: dict[str, OutputFormat] = {
    "WAV": OutputFormat("WAV", ".wav", (16, 24, 32)),
    "FLAC": OutputFormat("FLAC", ".flac", (16, 24)),
    "MP3": OutputFormat("MP3", ".mp3", ()),
    "OGG": OutputFormat("OGG Vorbis", ".ogg", ()),
    "OPUS": OutputFormat("OPUS", ".opus", (), needs_ffmpeg=True),
}


def _local_bin() -> Path:
    if getattr(__import__("sys"), "frozen", False):
        return Path(__import__("sys").executable).resolve().parent / "tools" / "ffmpeg" / "bin"
    return Path(__file__).resolve().parent.parent / "tools" / "ffmpeg" / "bin"


_LOCAL_BIN = _local_bin()


def ffmpeg_path() -> str | None:
    """Local project copy (tools/ffmpeg) first, then PATH."""
    local = _LOCAL_BIN / ("ffmpeg.exe" if "\\" in str(_LOCAL_BIN) else "ffmpeg")
    if local.is_file():
        return str(local)
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    local = _LOCAL_BIN / ("ffprobe.exe" if "\\" in str(_LOCAL_BIN) else "ffprobe")
    if local.is_file():
        return str(local)
    return shutil.which("ffprobe")


def available_output_formats() -> list[str]:
    names = []
    for key, fmt in OUTPUT_FORMATS.items():
        if fmt.needs_ffmpeg and not ffmpeg_path():
            try:  # soundfile may still know an OPUS subtype
                sf.available_formats()
                info = sf.available_subtypes("OGG")
                if "OPUS" not in info:
                    continue
            except Exception:
                continue
        names.append(key)
    return names


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read audio -> ((channels, samples) float32, sample_rate)."""
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return ensure_channels(data), int(sr)
    except Exception as sf_err:
        if not ffmpeg_path():
            raise RuntimeError(
                f"Could not decode '{path.name}' and ffmpeg was not found. "
                "Install ffmpeg to support this container."
            ) from sf_err
        return _read_via_ffmpeg(path)


def _read_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    probe = ffprobe_path()
    if not probe:
        raise RuntimeError("ffprobe not found (needed for this format).")
    info = subprocess.run(
        [probe, "-v", "quiet", "-print_format", "json",
         "-show_streams", str(path)],
        capture_output=True, text=True, timeout=120,
        creationflags=_CREATE_NO_WINDOW,
    )
    streams = json.loads(info.stdout or "{}").get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise RuntimeError(f"No audio stream in '{path.name}'.")
    sr = int(audio["sample_rate"])
    ch = int(audio["channels"])

    raw = subprocess.run(
        [ffmpeg_path(), "-v", "error", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True, timeout=None,
        creationflags=_CREATE_NO_WINDOW,
    )
    if raw.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on '{path.name}': {raw.stderr.decode(errors='replace')[:300]}"
        )
    data = np.frombuffer(raw.stdout, dtype=np.float32)
    frames = data.size // ch
    return data[: frames * ch].reshape(frames, ch).T.copy(), sr


def write_audio(
    path: str | Path,
    x: np.ndarray,
    sr: int,
    fmt: str = "WAV",
    bit_depth: int = 24,
    dither: bool = True,
) -> Path:
    """Write (channels, samples) float32 to *path* in the requested format."""
    path = Path(path)
    fmt = fmt.upper()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {fmt}")

    x = ensure_channels(x)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:  # never hand the encoder a hot signal
        x = x * (0.999 / peak)

    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "OPUS" and ffmpeg_path():
        return _write_opus_ffmpeg(path, x, sr)

    if fmt in _LOSSY:
        if fmt == "OPUS":
            sf.write(str(path), x.T, sr, format="OGG", subtype="OPUS")
        elif fmt == "OGG":
            sf.write(str(path), x.T, sr, format="OGG", subtype="VORBIS")
        else:  # MP3 via libsndfile >= 1.1
            sf.write(str(path), x.T, sr, format="MP3")
        return path

    depth = bit_depth if bit_depth in _OUTPUT_SUBTYPES[fmt] else 24
    subtype = _OUTPUT_SUBTYPES[fmt][depth]
    if dither and depth == 16:
        x = tpdf_dither(x, 16)
    sf.write(str(path), x.T, sr, format=fmt, subtype=subtype)
    return path


def _write_opus_ffmpeg(path: Path, x: np.ndarray, sr: int,
                       bitrate: str = "192k") -> Path:
    ch = x.shape[0]
    pcm = np.ascontiguousarray(x.T).tobytes()
    proc = subprocess.run(
        [ffmpeg_path(), "-v", "error", "-y",
         "-f", "f32le", "-ar", str(sr), "-ac", str(ch), "-i", "-",
         "-c:a", "libopus", "-b:a", bitrate, str(path)],
        input=pcm, capture_output=True,
        creationflags=_CREATE_NO_WINDOW,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg OPUS encode failed: {proc.stderr.decode(errors='replace')[:300]}"
        )
    return path

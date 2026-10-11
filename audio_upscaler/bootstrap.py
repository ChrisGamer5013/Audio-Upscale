"""Self-contained dependency bootstrapper for the portable/frozen app.

Currently: ensures ffmpeg is available (local tools\\ffmpeg copy first, then
winget, then direct download). Safe to call on every startup -- returns
immediately when ffmpeg is already present.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

Log = Callable[[str], None]


def app_dir() -> Path:
    """Folder that owns tools/ (exe dir when frozen, repo root otherwise)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def local_ffmpeg() -> Path | None:
    exe = app_dir() / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    return exe if exe.is_file() else None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None or local_ffmpeg() is not None


def _copy_from(bin_dir: Path, log: Log) -> bool:
    ffmpeg, ffprobe = bin_dir / "ffmpeg.exe", bin_dir / "ffprobe.exe"
    if not ffmpeg.is_file():
        return False
    dest = app_dir() / "tools" / "ffmpeg" / "bin"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ffmpeg, dest / "ffmpeg.exe")
    if ffprobe.is_file():
        shutil.copy2(ffprobe, dest / "ffprobe.exe")
    log(f"ffmpeg installed -> {dest}")
    return True


def _try_winget(log: Log) -> bool:
    if not shutil.which("winget"):
        return False
    log("Trying: winget install Gyan.FFmpeg ...")
    try:
        proc = subprocess.run(
            ["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--silent",
             "--accept-source-agreements", "--accept-package-agreements",
             "--disable-interactivity"],
            timeout=900, creationflags=_CREATE_NO_WINDOW)
    except Exception as exc:
        log(f"  winget failed to run: {exc}")
        return False
    if proc.returncode not in (0, -1978335189, -1978335212):
        log(f"  winget exited with code {proc.returncode}")
    links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"
    if (links / "ffmpeg.exe").is_file():
        return _copy_from(links, log)
    pkg_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for pkg_dir in pkg_root.glob("Gyan.FFmpeg*"):
        for exe in pkg_dir.rglob("ffmpeg.exe"):
            return _copy_from(exe.parent, log)
    log("  winget install didn't expose binaries this process can see.")
    return False


def _download(log: Log) -> bool:
    log(f"Downloading a static ffmpeg build (~90 MB) ...")
    tmp = Path(tempfile.mkdtemp(prefix="au_ffmpeg_"))
    zip_path = tmp / "ffmpeg.zip"
    try:
        urllib.request.urlretrieve(FFMPEG_URL, zip_path)
        log("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        for exe in tmp.rglob("ffmpeg.exe"):
            return _copy_from(exe.parent, log)
        log("  ffmpeg.exe not found inside the archive")
        return False
    except Exception as exc:
        log(f"  download failed: {exc}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ensure_ffmpeg(log: Log = print) -> bool:
    """Install ffmpeg if missing. Returns True when available afterwards."""
    if os.name != "nt":
        return shutil.which("ffmpeg") is not None
    if shutil.which("ffmpeg"):
        return True
    if local_ffmpeg():
        return True
    if _try_winget(log) and local_ffmpeg():
        return True
    if _download() and local_ffmpeg():
        return True
    log("ffmpeg unavailable; WAV/FLAC/MP3/OGG still work, OPUS and exotic "
        "containers don't.")
    return False

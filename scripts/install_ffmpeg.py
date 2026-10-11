"""Auto-installer for ffmpeg on Windows.

Strategy, in order:
  1. ffmpeg already on PATH              -> done
  2. local copy at tools/ffmpeg/bin      -> done
  3. `winget install Gyan.FFmpeg`        -> copy binaries into tools/ffmpeg
  4. direct download from gyan.dev       -> extract into tools/ffmpeg

The app looks for tools/ffmpeg/bin/ffmpeg.exe first, so no PATH changes
are required. Run directly:  python scripts/install_ffmpeg.py
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

ROOT = Path(__file__).resolve().parent.parent
LOCAL_BIN = ROOT / "tools" / "ffmpeg" / "bin"
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def _local_ffmpeg() -> Path | None:
    exe = LOCAL_BIN / "ffmpeg.exe"
    return exe if exe.is_file() else None


def _winget_available() -> bool:
    return shutil.which("winget") is not None


def _try_winget() -> bool:
    print("Trying: winget install Gyan.FFmpeg ...")
    try:
        proc = subprocess.run(
            [
                "winget", "install", "--id", "Gyan.FFmpeg", "-e", "--silent",
                "--accept-source-agreements", "--accept-package-agreements",
                "--disable-interactivity",
            ],
            timeout=900,
            creationflags=_CREATE_NO_WINDOW,
        )
    except Exception as exc:
        print(f"  winget failed to run: {exc}")
        return False
    if proc.returncode not in (0, -1978335189, -1978335212):
        # 0 = ok; the hex codes are "already installed" / "no upgrade" variants
        print(f"  winget exited with code {proc.returncode}")
    # portable packages land under WinGet Links / Packages
    links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"
    if (links / "ffmpeg.exe").is_file():
        return _copy_from(links)
    pkg_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for pkg_dir in pkg_root.glob("Gyan.FFmpeg*"):
        for exe in pkg_dir.rglob("ffmpeg.exe"):
            return _copy_from(exe.parent)
    print("  winget install didn't expose binaries this process can see.")
    return False


def _copy_from(bin_dir: Path) -> bool:
    ffmpeg = bin_dir / "ffmpeg.exe"
    ffprobe = bin_dir / "ffprobe.exe"
    if not ffmpeg.is_file():
        return False
    LOCAL_BIN.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ffmpeg, LOCAL_BIN / "ffmpeg.exe")
    if ffprobe.is_file():
        shutil.copy2(ffprobe, LOCAL_BIN / "ffprobe.exe")
    print(f"  installed -> {LOCAL_BIN}")
    return True


def _download() -> bool:
    print(f"Downloading a static build (~90 MB) from:\n  {FFMPEG_URL}")
    tmp = Path(tempfile.mkdtemp(prefix="au_ffmpeg_"))
    zip_path = tmp / "ffmpeg.zip"
    try:
        def _progress(count, block, total):
            if total > 0:
                pct = min(100, count * block * 100 // total)
                print(f"\r  {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(FFMPEG_URL, zip_path, _progress)
        print()
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        for exe in tmp.rglob("ffmpeg.exe"):
            return _copy_from(exe.parent)
        print("  ffmpeg.exe not found inside the archive??")
        return False
    except Exception as exc:
        print(f"\n  download failed: {exc}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    if os.name != "nt":
        print("Non-Windows detected - please install ffmpeg via your package "
              "manager (apt / dnf / brew).")
        return 0

    if shutil.which("ffmpeg"):
        print(f"ffmpeg found on PATH: {shutil.which('ffmpeg')}")
        return 0
    if (exe := _local_ffmpeg()):
        print(f"ffmpeg already installed locally: {exe}")
        return 0

    if _winget_available() and _try_winget() and _local_ffmpeg():
        return 0
    if _download() and _local_ffmpeg():
        return 0

    print(
        "\nWARNING: ffmpeg could not be installed automatically.\n"
        "Everything still works without it (WAV/FLAC/MP3/OGG). Only OPUS\n"
        "output and exotic containers (m4a/aac/wma, audio in video files)\n"
        "will be unavailable. Manual install: https://ffmpeg.org/download.html"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

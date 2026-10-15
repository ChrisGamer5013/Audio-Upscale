"""Add 'Upscale with Audio Upscaler HD' to the Explorer right-click menu
for audio files, and a 'Send to Audio Upscaler HD' entry.

    python scripts/install_context_menu.py          # install
    python scripts/install_context_menu.py remove   # uninstall

Per-user registry only (HKCU) - no admin needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"

AUDIO_EXTS = [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac",
              ".wma", ".aiff", ".aif"]

MENU_LABEL = "Upscale with Audio Upscaler HD"


def _target() -> str:
    exe = Path(sys.executable)
    # prefer pythonw from the venv if present
    if PYTHONW.is_file():
        exe = PYTHONW
    return f'"{exe}" "{ROOT / "main.py"}"'


def _bat_target() -> str:
    """CLI launcher bat for the Send-to entry."""
    bat = ROOT / "upscale_files.bat"
    bat.write_text(
        "@echo off\r\n"
        f"\"{ROOT / '.venv' / 'Scripts' / 'python.exe'}\" \"{ROOT / 'main.py'}\" "
        "cli %*\r\npause\r\n",
        encoding="utf-8",
    )
    return str(bat)


def install() -> None:
    import winreg

    cmd = _target()
    for ext in AUDIO_EXTS:
        key_path = rf"Software\Classes\SystemFileAssociations\{ext}\shell\AudioUpscalerHD"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as k:
            winreg.SetValueEx(k, None, 0, winreg.REG_SZ, MENU_LABEL)
            winreg.SetValueEx(k, "Icon", 0, winreg.REG_SZ,
                              str(ROOT / ".venv" / "Scripts" / "pythonw.exe"))
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path + r"\command") as k:
            winreg.SetValueEx(k, None, 0, winreg.REG_SZ,
                              f'{cmd} cli "%1"')

    # SendTo entry
    import os

    sendto = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "SendTo"
    sendto.mkdir(parents=True, exist_ok=True)
    lnk_bat = sendto / "Audio Upscaler HD (CLI).bat"
    lnk_bat.write_text(
        "@echo off\r\n"
        f'"{ROOT / ".venv" / "Scripts" / "python.exe"}" '
        f'"{ROOT / "main.py"}" cli %*\r\npause\r\n',
        encoding="utf-8",
    )
    print(f"Installed context menu for {len(AUDIO_EXTS)} audio types.")
    print(f"SendTo launcher: {lnk_bat}")
    print("Runs the GUI with the file already queued via the CLI menu entries.")


def remove() -> None:
    import os
    import winreg

    for ext in AUDIO_EXTS:
        key_path = rf"Software\Classes\SystemFileAssociations\{ext}\shell\AudioUpscalerHD"
        for sub in (key_path + r"\command", key_path):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, sub)
            except FileNotFoundError:
                pass
    bat = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "SendTo" / "Audio Upscaler HD (CLI).bat"
    bat.unlink(missing_ok=True)
    print("Context menu entries removed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "remove":
        remove()
    else:
        install()

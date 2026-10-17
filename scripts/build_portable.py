"""Build the portable all-in-one release:

  dist/AudioUpscalerHD/            portable folder (run AudioUpscalerHD.exe)
  dist/AudioUpscalerHD-portable.zip
  dist/installer/AudioUpscalerHD_Setup.exe   (if Inno Setup is available)

The exe auto-downloads ffmpeg on first run via audio_upscaler.bootstrap.

Run:  .venv\\Scripts\\python scripts\\build_portable.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
APP_NAME = "AudioUpscalerHD"

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def _python() -> Path:
    """venv python locally, system python on CI."""
    if VENV_PY.is_file():
        return VENV_PY
    return Path(sys.executable)


def pyinstaller_build() -> Path:
    if not _python().is_file():
        sys.exit("no usable python - run setup_env.bat first")
    run([str(_python()), "-m", "pip", "install", "-q", "-r",
         "requirements-build.txt"])
    base = [
        "--noconfirm", "--clean",
        "--console",                        # keep CLI usable from the exe
        "--collect-all", "customtkinter",
        "--collect-all", "tkinterdnd2",
        "--collect-all", "onnxruntime",
        "--exclude-module", "torch",
        "--exclude-module", "pytest",
    ]
    # portable folder build (fast startup, installer source)
    run([str(_python()), "-m", "PyInstaller", *base,
         "--name", APP_NAME, "main.py"])
    # single-file all-in-one exe (everything embedded; self-extracts on run)
    run([str(_python()), "-m", "PyInstaller", *base,
         "--onefile", "--name", f"{APP_NAME}-standalone", "main.py"])
    onefile = ROOT / "dist" / f"{APP_NAME}-standalone.exe"
    if not onefile.is_file():
        sys.exit("onefile build failed - exe not found")
    app_dir = ROOT / "dist" / APP_NAME
    if not (app_dir / f"{APP_NAME}.exe").is_file():
        sys.exit("PyInstaller build failed - exe not found")
    # bundle models/README + our bundled starter BWE model only
    # (never ship the user's own/test models in the portable build)
    (app_dir / "models").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "models" / "README.md", app_dir / "models" / "README.md")
    for name in ("starter_bwe_48k.onnx",):
        model = ROOT / "models" / name
        if model.is_file():
            shutil.copy2(model, app_dir / "models" / name)
    return app_dir


def make_zip(app_dir: Path) -> Path:
    zip_path = ROOT / "dist" / f"{APP_NAME}-portable.zip"
    if zip_path.exists():
        zip_path.unlink()
    print(f"zipping -> {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for p in sorted(app_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(app_dir.parent))
    return zip_path


_INNO = Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe")


def _ensure_inno() -> Path | None:
    if _INNO.is_file():
        return _INNO
    if shutil.which("winget"):
        print("Installing Inno Setup via winget ...")
        subprocess.run(
            ["winget", "install", "--id", "JRSoftware.InnoSetup", "-e",
             "--silent", "--accept-source-agreements",
             "--accept-package-agreements", "--disable-interactivity"],
            creationflags=_CREATE_NO_WINDOW, timeout=900)
    return _INNO if _INNO.is_file() else None


def make_installer(app_dir: Path) -> Path | None:
    iscc = _ensure_inno()
    if not iscc:
        print("Inno Setup not available - skipping installer (zip is fine)")
        return None
    iss = ROOT / "dist" / "installer.iss"
    out_dir = ROOT / "dist" / "installer"
    out_dir.mkdir(parents=True, exist_ok=True)
    iss.write_text(f"""
[Setup]
AppName=Audio Upscaler HD
AppVersion={_version()}
DefaultDirName={{autopf}}\\Audio Upscaler HD
DefaultGroupName=Audio Upscaler HD
OutputDir={out_dir}
OutputBaseFilename={APP_NAME}_Setup
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "{app_dir}\\*"; DestDir: "{{app}}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{{group}}\\Audio Upscaler HD"; Filename: "{{app}}\\{APP_NAME}.exe"
Name: "{{autodesktop}}\\Audio Upscaler HD"; Filename: "{{app}}\\{APP_NAME}.exe"

[Run]
Filename: "{{app}}\\{APP_NAME}.exe"; Description: "Launch now"; Flags: postinstall nowait shellexec
""".strip())
    run([str(iscc), str(iss)])
    setup = out_dir / f"{APP_NAME}_Setup.exe"
    return setup if setup.is_file() else None


def _version() -> str:
    sys.path.insert(0, str(ROOT))
    from audio_upscaler import __version__

    return __version__


def main() -> int:
    app_dir = pyinstaller_build()
    zip_path = make_zip(app_dir)
    installer = make_installer(app_dir)
    print("\n=== build artifacts ===")
    print(f"  portable folder: {app_dir}")
    print(f"  portable zip:    {zip_path}")
    if installer:
        print(f"  installer:       {installer}")
    else:
        print("  installer:       (skipped - Inno Setup unavailable)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        sys.exit(f"build failed (step exited {exc.returncode})")

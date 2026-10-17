"""CustomTkinter GUI for Audio Upscaler HD.

Dark, modern interface: file queue with drag & drop, engine/preset selection,
output options, progress bar, cancellation, and before/after spectrogram preview.
"""
from __future__ import annotations

import os
import queue
import re
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from .. import APP_NAME, __version__
from ..audio_io import OUTPUT_FORMATS, available_output_formats
from ..engines import create_engine, engine_status
from ..presets import PRESETS
from ..pipeline import JobSettings, run_batch
from ..providers import list_gpus

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

MODEL_LABELS = {"dsp": "DSP Enhance (built-in)", "onnx": "ONNX (GPU)"}
_SR_CHOICES = ["Keep source", "44,100", "48,000", "88,200", "96,000",
               "176,400", "192,000"]
_LOSSY = {"MP3", "OGG", "OPUS"}


def _models_dir() -> Path:
    from ..bootstrap import app_dir

    d = app_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self._msg_q: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._force = threading.Event()
        self._worker: threading.Thread | None = None
        self._files: list[Path] = []
        self._formats = available_output_formats()
        self._summary = None
        self._out_dir: Path | None = None
        self._failed: list[Path] = []
        self._chain_cache: dict[str, str] = {}
        self._chain_job: str | None = None
        self._batch_t0: float | None = None
        self._eta_prev: tuple[float, float] | None = None
        self._eta_speed: float | None = None

        if _HAS_DND:
            try:
                self.TkdndVersion = TkinterDnD._require(self)
            except Exception:
                pass

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title(f"{APP_NAME}  v{__version__}")
        self.geometry("1150x740")
        self.minsize(1000, 680)

        self._build_ui()
        self._poll_queue()
        self._load_prefs()
        self.after(2500, self._check_updates)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------ prefs
    _PREF_VARS = (
        ("engine", "engine_label_var"), ("preset", "preset_var"),
        ("sr", "sr_var"), ("fmt", "fmt_var"), ("bit", "bit_var"),
        ("lufs", "lufs_var"), ("denoise", "denoise_var"),
        ("template", "template_var"), ("workers", "workers_var"),
        ("outdir", "outdir_var"), ("model", "model_var"),
        ("device", "device_var"), ("theme", "theme_var"),
        ("accent", "accent_var"),
    )
    _PREF_SWITCHES = (
        ("loud", "loud_var"), ("dither", "dither_var"), ("spec", "spec_var"),
        ("auto_preset", "auto_preset_var"), ("declip", "declip_var"),
        ("mid_side", "ms_var"), ("ultra", "ultra_var"),
        ("keep_tree", "keeptree_var"), ("nometa", "nometa_var"),
        ("gpu", "use_dml_var"), ("skip_existing", "skip_exist_var"),
        ("sound", "sound_var"), ("watch_auto", "watch_auto_var"),
    )

    def _collect_prefs(self) -> dict:
        prefs = {}
        for key, attr in self._PREF_VARS:
            var = getattr(self, attr, None)
            if var is not None:
                prefs[key] = var.get()
        for key, attr in self._PREF_SWITCHES:
            var = getattr(self, attr, None)
            if var is not None:
                prefs[key] = bool(var.get())
        prefs["queue"] = [str(p) for p in self._files]
        try:
            prefs["geometry"] = self.geometry()
        except Exception:
            pass
        return prefs

    def _load_prefs(self) -> None:
        from .settings_store import load_settings

        prefs = load_settings()
        for key, attr in self._PREF_VARS:
            if key in prefs:
                var = getattr(self, attr, None)
                if var is not None:
                    try:
                        var.set(prefs[key])
                    except Exception:
                        pass
        for key, attr in self._PREF_SWITCHES:
            if key in prefs:
                var = getattr(self, attr, None)
                if var is not None:
                    try:
                        var.set(bool(prefs[key]))
                    except Exception:
                        pass
        try:
            if prefs.get("theme") and prefs["theme"] != "dark":
                ctk.set_appearance_mode(prefs["theme"])
        except Exception:
            pass
        try:
            if prefs.get("geometry"):
                self.geometry(prefs["geometry"])
        except Exception:
            pass
        for q in prefs.get("queue", [])[:200]:
            p = Path(q)
            if p.exists():
                self._files.append(p)
        if self._files:
            self._refresh_list()

    def _on_close(self) -> None:
        from .settings_store import save_settings

        save_settings(self._collect_prefs())
        self.destroy()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # ---------------- left: queue ----------------
        left = ctk.CTkFrame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            left, text="Queue", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=8, pady=(8, 4), sticky="w")
        self.queue_count_lbl = ctk.CTkLabel(
            left, text="0 items", text_color="#999",
            font=ctk.CTkFont(size=11))
        self.queue_count_lbl.grid(row=0, column=2, columnspan=2, padx=8,
                                  pady=(8, 4), sticky="e")

        list_holder = ctk.CTkFrame(left, fg_color="transparent")
        list_holder.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=6)
        list_holder.grid_columnconfigure(0, weight=1)
        list_holder.grid_rowconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(
            list_holder, bg="#2b2b2b", fg="#e0e0e0", highlightthickness=0,
            selectbackground="#1f6aa5", activestyle="none", selectmode="extended",
        )
        self.file_listbox.bind("<Double-Button-1>", self._preview_selected)
        scroll = tk.Scrollbar(list_holder, command=self.file_listbox.yview)
        self.file_listbox.config(yscrollcommand=scroll.set)
        self.file_listbox.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        if _HAS_DND:
            try:
                self.file_listbox.drop_target_register(DND_FILES)
                self.file_listbox.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass

        btns = [("Add Files", self._add_files), ("Add Folder", self._add_folder),
                ("Remove", self._remove_selected), ("Clear", self._clear_files)]
        btn_fr = ctk.CTkFrame(left, fg_color="transparent")
        btn_fr.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=6)
        for i, (text, cmd) in enumerate(btns):
            btn_fr.grid_columnconfigure(i, weight=1)
            ctk.CTkButton(btn_fr, text=text, command=cmd).grid(
                row=0, column=i, padx=2, sticky="ew")

        btns2 = [("↑ Up", self._move_up), ("↓ Down", self._move_down),
                 ("Sort A-Z", self._sort_queue),
                 ("Re-queue failed", self._requeue_failed)]
        btn_fr2 = ctk.CTkFrame(left, fg_color="transparent")
        btn_fr2.grid(row=4, column=0, columnspan=4, sticky="ew", padx=6, pady=(0, 6))
        for i, (text, cmd) in enumerate(btns2):
            btn_fr2.grid_columnconfigure(i, weight=1)
            ctk.CTkButton(btn_fr2, text=text, command=cmd).grid(
                row=0, column=i, padx=2, sticky="ew")

        hint = "Drag & drop files/folders here" if _HAS_DND else "Use the buttons to add audio files"
        ctk.CTkLabel(left, text=hint, text_color="#888").grid(
            row=3, column=0, columnspan=4, pady=(0, 6))

        # ---------------- right: settings ----------------
        right = ctk.CTkFrame(main, width=340)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Settings",
                     font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, padx=10, pady=(8, 4), sticky="w")
        r = 1

        def label(text):
            nonlocal r
            ctk.CTkLabel(right, text=text).grid(row=r, column=0, padx=10, sticky="w")
            r += 1

        def row_widget(w):
            nonlocal r
            w.grid(row=r, column=0, padx=10, pady=2, sticky="ew")
            r += 1

        # engine
        label("Engine")
        self.engine_label_var = tk.StringVar(value=MODEL_LABELS["dsp"])
        engine_menu = ctk.CTkOptionMenu(
            right, variable=self.engine_label_var,
            values=list(MODEL_LABELS.values()),
            command=self._on_engine_change,
        )
        row_widget(engine_menu)
        status = engine_status()
        self.engine_status_lbl = ctk.CTkLabel(
            right, text=", ".join(f"{k}: {v}" for k, v in status.items()),
            wraplength=300, justify="left", text_color="#999", font=ctk.CTkFont(size=11))
        row_widget(self.engine_status_lbl)

        # model (shown only for onnx)
        self.model_frame = ctk.CTkFrame(right, fg_color="transparent")
        self.model_var = tk.StringVar()
        ctk.CTkLabel(self.model_frame, text="ONNX model (.onnx)").pack(anchor="w")
        mrow = ctk.CTkFrame(self.model_frame, fg_color="transparent")
        mrow.pack(fill="x")
        ctk.CTkEntry(mrow, textvariable=self.model_var).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(mrow, text="Browse", width=60, command=self._browse_model).pack(
            side="left", padx=(4, 0))
        ctk.CTkButton(mrow, text="Chain…", width=58,
                      command=self._add_chain_stage).pack(side="left", padx=(4, 0))
        ctk.CTkButton(mrow, text="✕", width=28,
                      command=self._clear_model).pack(side="left", padx=(4, 0))
        self.chain_lbl = ctk.CTkLabel(
            self.model_frame, text="", justify="left", wraplength=300,
            text_color="#7fb3d5", font=ctk.CTkFont(size=11))
        self.chain_lbl.pack(anchor="w", pady=(2, 0))
        self.model_var.trace_add("write", lambda *_: self._on_model_changed())
        hw = list_gpus()
        dev_lbl = ctk.CTkLabel(self.model_frame,
                               text="GPU device index (DirectML)")
        dev_lbl.pack(anchor="w", pady=(6, 0))
        self.device_var = tk.StringVar(value="0")
        ctk.CTkEntry(self.model_frame, textvariable=self.device_var, width=60).pack(anchor="w")
        self.use_dml_var = tk.BooleanVar(value=True)
        ctk.CTkSwitch(self.model_frame,
                      text="GPU acceleration (DirectML)",
                      variable=self.use_dml_var).pack(anchor="w", pady=(4, 0))
        ctk.CTkLabel(
            self.model_frame,
            text="Chain models with + (e.g. backbone.onnx+spec_head.onnx). "
                 "Uncheck GPU if a big model takes very long to compile.",
            wraplength=300, justify="left", text_color="#999",
            font=ctk.CTkFont(size=11)).pack(anchor="w")
        from ..recommendations import gpu_vram_map
        vram = dict(gpu_vram_map())
        if hw:
            lines = []
            for i, g in enumerate(hw):
                gb = vram.get(g.name, 0.0)
                lines.append(f"  [{i}] {g.name}" + (f" ({gb:g} GB)" if gb else ""))
            gtxt = "\n".join(lines)
        else:
            gtxt = "  (no GPU detected)"
        ctk.CTkLabel(self.model_frame, text=gtxt, justify="left",
                     text_color="#999", font=ctk.CTkFont(size=11)).pack(anchor="w")
        pick = ctk.CTkFrame(self.model_frame, fg_color="transparent")
        pick.pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(pick, text="Models in models\\:").pack(anchor="w")
        self.model_pick = ctk.CTkOptionMenu(
            pick, values=["(refresh)"], command=self._on_model_pick)
        self.model_pick.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(pick, text="Refresh", width=62,
                      command=self._refresh_model_list).pack(side="left", padx=(4, 0))
        ctk.CTkButton(
            self.model_frame, text="Model recommendations (VRAM-aware)",
            command=self._show_recommendations,
        ).pack(anchor="w", pady=(6, 0))
        mrow2 = ctk.CTkFrame(self.model_frame, fg_color="transparent")
        mrow2.pack(fill="x", pady=(6, 0))
        model_tools = (("Info", self._model_info), ("Test", self._model_test),
                       ("Sidecar", self._model_sidecar),
                       ("Folder", self._open_models_folder))
        for i, (text, cmd) in enumerate(model_tools):
            mrow2.grid_columnconfigure(i, weight=1)
            ctk.CTkButton(mrow2, text=text, command=cmd, height=24).grid(
                row=0, column=i, padx=2, sticky="ew")
        self._model_pick_map: dict[str, str] = {}
        self.after(300, self._refresh_model_list)

        # preset
        label("Preset (DSP engine)")
        self.preset_var = tk.StringVar(value="music")
        row_widget(ctk.CTkOptionMenu(right, variable=self.preset_var,
                                     values=list(PRESETS.keys())))
        self.preset_desc = ctk.CTkLabel(right, text=PRESETS["music"].description,
                                        wraplength=300, justify="left",
                                        text_color="#999", font=ctk.CTkFont(size=11))
        row_widget(self.preset_desc)
        self.preset_var.trace_add("write", lambda *_: self.preset_desc.configure(
            text=PRESETS[self.preset_var.get()].description))

        # sample rate
        label("Target sample rate")
        self.sr_var = tk.StringVar(value="48,000")
        row_widget(ctk.CTkOptionMenu(right, variable=self.sr_var, values=_SR_CHOICES))

        # format + bit depth
        label("Output format")
        self.fmt_var = tk.StringVar(
            value="FLAC" if "FLAC" in self._formats else (self._formats[0] if self._formats else "WAV"))
        fmt_menu = ctk.CTkOptionMenu(
            right, variable=self.fmt_var,
            values=[f"{k} - {OUTPUT_FORMATS[k].name}" for k in self._formats],
            command=self._on_fmt_change)
        row_widget(fmt_menu)

        label("Bit depth (lossless only)")
        self.bit_var = tk.StringVar(value="24")
        self.bit_menu = ctk.CTkOptionMenu(right, variable=self.bit_var,
                                          values=["16", "24", "32 float"])
        row_widget(self.bit_menu)

        # toggles
        self.loud_var = tk.BooleanVar(value=False)
        self.loud_sw = ctk.CTkSwitch(right, text="Loudness normalize",
                                     variable=self.loud_var)
        row_widget(self.loud_sw)
        lufs_row = ctk.CTkFrame(right, fg_color="transparent")
        ctk.CTkLabel(lufs_row, text="Target LUFS:").pack(side="left")
        self.lufs_var = tk.StringVar(value="-14")
        ctk.CTkEntry(lufs_row, textvariable=self.lufs_var, width=60).pack(side="left", padx=6)
        row_widget(lufs_row)

        self.dither_var = tk.BooleanVar(value=True)
        row_widget(ctk.CTkSwitch(right, text="TPDF dither (16-bit)",
                                 variable=self.dither_var))
        self.spec_var = tk.BooleanVar(value=True)
        row_widget(ctk.CTkSwitch(right, text="Save before/after spectrograms",
                                 variable=self.spec_var))

        # ---- tabs: Enhance / Batch / About ----
        tabs = ctk.CTkTabview(right, height=250)
        row_widget(tabs)
        tab_enh = tabs.add("Enhance")
        tab_batch = tabs.add("Batch")
        tab_log = tabs.add("Log")
        tab_about = tabs.add("About")
        for t in (tab_enh, tab_batch, tab_log, tab_about):
            t.grid_columnconfigure(0, weight=1)

        # Enhance / Batch / Log / About tabs (methods below)
        self._build_enhance_tab(tab_enh)
        self._build_batch_tab(tab_batch)
        self._build_log_tab(tab_log)
        self._build_about_tab(tab_about)

        # ---------------- bottom: output + actions ----------------
        bottom = ctk.CTkFrame(main)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        bottom.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(bottom, text="Output:").grid(row=0, column=0, padx=8, pady=4, sticky="w")
        self.outdir_var = tk.StringVar()
        ctk.CTkEntry(bottom, textvariable=self.outdir_var).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4)
        ctk.CTkButton(bottom, text="Browse", width=70,
                      command=self._browse_outdir).grid(row=0, column=2, padx=4)

        act = ctk.CTkFrame(bottom, fg_color="transparent")
        act.grid(row=1, column=0, columnspan=3, sticky="ew", padx=4, pady=6)
        act.grid_columnconfigure(2, weight=1)

        self.start_btn = ctk.CTkButton(act, text="▶ Start", command=self._start_batch,
                                       fg_color="#2b8a3e", hover_color="#20712f")
        self.start_btn.grid(row=0, column=0, padx=4)
        self.cancel_btn = ctk.CTkButton(act, text="■ Cancel", command=self._cancel_batch,
                                        state="disabled", fg_color="#8a2b2b",
                                        hover_color="#712020")
        self.cancel_btn.grid(row=0, column=1, padx=4)
        self.force_btn = ctk.CTkButton(act, text="✖ Force Stop",
                                       command=self._force_stop_batch,
                                       state="disabled", fg_color="#5a1f1f",
                                       hover_color="#431515")
        self.force_btn.grid(row=0, column=2, padx=4)
        self.preview_btn = ctk.CTkButton(act, text="Preview", width=80,
                                         command=self._preview)
        self.preview_btn.grid(row=0, column=3, padx=4)
        self.open_btn = ctk.CTkButton(act, text="Open output folder", width=130,
                                      command=self._open_output, state="disabled")
        self.open_btn.grid(row=0, column=4, padx=4)

        self.progress = ctk.CTkProgressBar(bottom)
        self.progress.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(2, 2))
        self.progress.set(0.0)
        self.file_progress = ctk.CTkProgressBar(bottom, height=6,
                                                progress_color="#4c7fb8")
        self.file_progress.grid(row=3, column=0, columnspan=3, sticky="ew",
                                padx=8, pady=(0, 3))
        self.file_progress.set(0.0)
        self.progress_lbl = ctk.CTkLabel(bottom, text="Ready", text_color="#aaa")
        self.progress_lbl.grid(row=4, column=0, columnspan=3, sticky="w", padx=8)
        self.file_lbl = ctk.CTkLabel(bottom, text="", text_color="#888",
                                     font=ctk.CTkFont(size=11))
        self.file_lbl.grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))

        self._on_engine_change(self.engine_label_var.get())
        self._on_fmt_change(None)

    # ------------------------------------------------------------------ helpers
    def _build_enhance_tab(self, tab) -> None:
        self.auto_preset_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(tab, text="Auto preset per file",
                      variable=self.auto_preset_var).grid(
            row=0, column=0, padx=8, pady=3, sticky="w")
        dn = ctk.CTkFrame(tab, fg_color="transparent")
        dn.grid(row=1, column=0, padx=8, pady=3, sticky="ew")
        ctk.CTkLabel(dn, text="Denoise (0=off, 1 mild, 2 strong):").pack(side="left")
        self.denoise_var = tk.StringVar(value="0")
        ctk.CTkEntry(dn, textvariable=self.denoise_var, width=46).pack(side="left", padx=6)
        self.declip_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(tab, text="Declip (repair clipped audio)",
                      variable=self.declip_var).grid(
            row=2, column=0, padx=8, pady=3, sticky="w")
        self.ms_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(tab, text="Mid/Side mode (stereo)",
                      variable=self.ms_var).grid(
            row=3, column=0, padx=8, pady=3, sticky="w")
        self.ultra_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(tab, text="Ultra resampler (slower)",
                      variable=self.ultra_var).grid(
            row=4, column=0, padx=8, pady=3, sticky="w")

    def _build_batch_tab(self, tab) -> None:
        wr = ctk.CTkFrame(tab, fg_color="transparent")
        wr.grid(row=0, column=0, padx=8, pady=3, sticky="ew")
        ctk.CTkLabel(wr, text="Parallel workers (1-8):").pack(side="left")
        self.workers_var = tk.StringVar(value="1")
        ctk.CTkEntry(wr, textvariable=self.workers_var, width=42).pack(side="left", padx=6)

        ctk.CTkLabel(tab, text="Name template:  {name} {suffix} {sr} {preset} "
                     "{format} {date} {time}", wraplength=280, justify="left",
                     text_color="#999", font=ctk.CTkFont(size=10)).grid(
            row=1, column=0, padx=8, sticky="w")
        self.template_var = tk.StringVar(value="{name}{suffix}")
        ctk.CTkEntry(tab, textvariable=self.template_var).grid(
            row=2, column=0, padx=8, pady=2, sticky="ew")
        self.keeptree_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(tab, text="Keep folder structure (keep-tree)",
                      variable=self.keeptree_var).grid(
            row=3, column=0, padx=8, pady=3, sticky="w")
        self.nometa_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(tab, text="Skip metadata copy",
                      variable=self.nometa_var).grid(
            row=4, column=0, padx=8, pady=3, sticky="w")
        self.skip_exist_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(tab, text="Skip existing outputs (resume-safe)",
                      variable=self.skip_exist_var).grid(
            row=5, column=0, padx=8, pady=3, sticky="w")
        self.sound_var = tk.BooleanVar(value=True)
        ctk.CTkSwitch(tab, text="Play a sound when a batch finishes",
                      variable=self.sound_var).grid(
            row=6, column=0, padx=8, pady=3, sticky="w")

        rowb = ctk.CTkFrame(tab, fg_color="transparent")
        rowb.grid(row=7, column=0, padx=8, pady=4, sticky="ew")
        ctk.CTkButton(rowb, text="Estimate", width=80,
                      command=self._show_estimate).pack(side="left")
        ctk.CTkButton(rowb, text="Compare presets", width=110,
                      command=self._compare_presets).pack(side="left", padx=6)

        wl = ctk.CTkFrame(tab, fg_color="transparent")
        wl.grid(row=8, column=0, padx=8, pady=4, sticky="ew")
        self.watch_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(wl, text="Watch folder", variable=self.watch_var,
                      command=self._toggle_watch).pack(side="left")
        ctk.CTkButton(wl, text="Pick…", width=52,
                      command=self._pick_watch_dir).pack(side="left", padx=6)
        self.watch_auto_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(wl, text="auto-start on new files",
                      variable=self.watch_auto_var).pack(side="left", padx=6)
        self.watch_dir: Path | None = None
        self._watch_seen: set[str] = set()
        self._watch_job: str | None = None

    def _build_log_tab(self, tab) -> None:
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.grid(row=0, column=0, padx=8, pady=(4, 0), sticky="ew")
        ctk.CTkButton(top, text="Clear", width=60,
                      command=self._clear_log).pack(side="right")
        self.log_box = ctk.CTkTextbox(tab, wrap="word",
                                      font=ctk.CTkFont(size=11))
        self.log_box.grid(row=1, column=0, padx=8, pady=4, sticky="nsew")
        tab.grid_rowconfigure(1, weight=1)
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        try:
            self.log_box.configure(state="normal")
            self.log_box.delete("0.0", "end")
            self.log_box.configure(state="disabled")
        except Exception:
            pass

    def _log(self, text: str) -> None:
        box = getattr(self, "log_box", None)
        if box is None:
            return
        try:
            box.configure(state="normal")
            box.insert("end", f"{time.strftime('%H:%M:%S')}  {text}\n")
            box.see("end")
            box.configure(state="disabled")
        except Exception:
            pass

    def _build_about_tab(self, tab) -> None:
        ctk.CTkButton(tab, text="Diagnostics (copy for bug reports)",
                      command=self._show_diagnostics).grid(
            row=0, column=0, padx=8, pady=4, sticky="ew")
        ctk.CTkButton(tab, text="Check for updates",
                      command=self._check_updates).grid(
            row=1, column=0, padx=8, pady=4, sticky="ew")
        self.update_lbl = ctk.CTkLabel(tab, text="", text_color="#8fd18b",
                                       font=ctk.CTkFont(size=11))
        self.update_lbl.grid(row=2, column=0, padx=8, sticky="w")
        ctk.CTkButton(tab, text="Download starter model",
                      command=self._download_starter).grid(
            row=3, column=0, padx=8, pady=4, sticky="ew")
        ctk.CTkButton(tab, text="GPU backend requirements",
                      command=self._show_requirements).grid(
            row=4, column=0, padx=8, pady=4, sticky="ew")
        th = ctk.CTkFrame(tab, fg_color="transparent")
        th.grid(row=5, column=0, padx=8, pady=4, sticky="ew")
        ctk.CTkLabel(th, text="Theme:").pack(side="left")
        self.theme_var = tk.StringVar(value="dark")
        ctk.CTkOptionMenu(th, variable=self.theme_var,
                          values=["dark", "light", "system"],
                          command=lambda v: ctk.set_appearance_mode(v)
                          ).pack(side="left", padx=4)
        self.accent_var = tk.StringVar(value="blue")
        ctk.CTkOptionMenu(th, variable=self.accent_var,
                          values=["blue", "green", "dark-blue"]
                          ).pack(side="left", padx=4)
        ctk.CTkLabel(tab, text="Accent applies after restart.",
                     text_color="#888", font=ctk.CTkFont(size=10)).grid(
            row=6, column=0, padx=8, sticky="w")

    def _on_engine_change(self, label_text: str) -> None:
        if label_text == MODEL_LABELS["onnx"]:
            self.model_frame.grid(row=3, column=0, padx=10, pady=2, sticky="ew")
        else:
            self.model_frame.grid_forget()

    def _engine_id(self) -> str:
        for k, v in MODEL_LABELS.items():
            if v == self.engine_label_var.get():
                return k
        return "dsp"

    def _fmt_key(self) -> str:
        v = self.fmt_var.get()
        # "FLAC - FLAC" style values
        key = v.split(" - ")[0].strip().upper()
        return key if key in OUTPUT_FORMATS else v.upper()

    def _on_fmt_change(self, _val) -> None:
        fmt = self._fmt_key()
        lossy = fmt in _LOSSY
        state = "disabled" if lossy else "normal"
        self.bit_menu.configure(state=state)

    def _on_drop(self, event) -> None:
        data = event.data
        paths = re.findall(r"\{([^}]*)\}|(\S+)", data)
        model_paths: list[Path] = []
        for brace, bare in paths:
            p = Path(brace or bare)
            if not p.name:
                continue
            if p.suffix.lower() == ".onnx" and p.is_file():
                model_paths.append(p)
            else:
                self._add_paths([p])
        if model_paths:
            self._select_models(model_paths)

    def _select_models(self, models: list[Path]) -> None:
        """Copy model files into models/ and select them (multiple = chain)."""
        import shutil as _sh

        dests: list[Path] = []
        for p in models:
            dest = _models_dir() / p.name
            try:
                if p.resolve() != dest.resolve():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    _sh.copy2(p, dest)
            except Exception as exc:
                messagebox.showerror("Model import failed", str(exc))
                continue
            dests.append(dest)
        if not dests:
            return
        self.engine_label_var.set(MODEL_LABELS["onnx"])
        self._on_engine_change(MODEL_LABELS["onnx"])
        if len(dests) == 1:
            self.model_var.set(str(dests[0]))
        else:
            self.model_var.set("+".join(str(d) for d in dests))
            self.progress_lbl.configure(
                text=f"Chained {len(dests)} models with '+'")
        self._refresh_model_list()

    def _add_paths(self, paths: list[Path]) -> None:
        added = 0
        for p in paths:
            if p.is_dir():
                for ext in sorted({e for e in os.listdir(p)}):
                    pass  # let pipeline.collect filter; just add dir itself below
                self._files.append(p)
                added += 1
            elif p.is_file():
                self._files.append(p)
                added += 1
        if added:
            self._refresh_list()

    def _add_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted({
            ".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac",
            ".wma", ".aiff", ".aif"}))
        paths = filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=[("Audio", exts), ("All files", "*.*")])
        if paths:
            self._add_paths([Path(p) for p in paths])

    def _add_folder(self) -> None:
        d = filedialog.askdirectory(title="Select folder")
        if d:
            self._files.append(Path(d))
            self._refresh_list()

    def _refresh_list(self) -> None:
        seen = []
        for p in self._files:
            if p not in seen:
                seen.append(p)
        self._files = seen
        self.file_listbox.delete(0, "end")
        for p in self._files:
            self.file_listbox.insert("end", str(p))
        try:
            self.queue_count_lbl.configure(
                text=f"{len(self._files)} item(s)")
        except Exception:
            pass

    def _move_up(self) -> None:
        sel = set(self.file_listbox.curselection())
        if not sel:
            return
        for i in sorted(sel):
            if i > 0 and (i - 1) not in sel:
                self._files[i - 1], self._files[i] = \
                    self._files[i], self._files[i - 1]
        self._refresh_list()
        for i in sorted(sel):
            self.file_listbox.selection_set(max(0, i - 1))

    def _move_down(self) -> None:
        sel = set(self.file_listbox.curselection())
        if not sel:
            return
        for i in sorted(sel, reverse=True):
            if i < len(self._files) - 1 and (i + 1) not in sel:
                self._files[i + 1], self._files[i] = \
                    self._files[i], self._files[i + 1]
        self._refresh_list()
        for i in sorted(sel):
            self.file_listbox.selection_set(
                min(len(self._files) - 1, i + 1))

    def _sort_queue(self) -> None:
        self._files.sort(key=lambda p: str(p).lower())
        self._refresh_list()

    def _requeue_failed(self) -> None:
        if not self._failed:
            messagebox.showinfo("Re-queue", "No failed files from the last run.")
            return
        added = 0
        for p in self._failed:
            if p not in self._files:
                self._files.append(p)
                added += 1
        self._failed = []
        self._refresh_list()
        self.progress_lbl.configure(text=f"Re-queued {added} failed file(s)")

    def _preview_selected(self, _event=None) -> None:
        sel = self.file_listbox.curselection()
        if not sel:
            return
        src = self._files[sel[0]]
        if src.is_file():
            threading.Thread(target=self._preview_worker, args=(src,),
                             daemon=True).start()

    def _remove_selected(self) -> None:
        sel = list(self.file_listbox.curselection())
        for i in reversed(sel):
            del self._files[i]
        self._refresh_list()

    def _clear_files(self) -> None:
        self._files.clear()
        self.file_listbox.delete(0, "end")

    def _refresh_model_list(self) -> None:
        def work():
            from ..engines.onnx_engine import OnnxEngine

            items, mapping = [], {}
            for p in sorted(_models_dir().glob("*.onnx")):
                try:
                    kind = OnnxEngine.inspect(p)
                except Exception:
                    kind = "?"
                label = f"{p.name}  [{kind}]"
                items.append(label)
                mapping[label] = str(p)
            self._msg_q.put(("models", items or ["(none found)"], mapping))

        threading.Thread(target=work, daemon=True).start()

    def _on_model_pick(self, label: str) -> None:
        path = getattr(self, "_model_pick_map", {}).get(label)
        if path:
            self.model_var.set(path)

    def _browse_model(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select ONNX model(s) — multiple = chain (e.g. backbone + head)",
            filetypes=[("ONNX model", "*.onnx")])
        if paths:
            self.model_var.set(
                "+".join(str(Path(p)) for p in paths))

    def _add_chain_stage(self) -> None:
        """Append one or more models to the current spec as chain stages."""
        from ..engines.onnx_engine import split_models

        paths = filedialog.askopenfilenames(
            title="Add model stage(s) to the chain (fed in order)",
            filetypes=[("ONNX model", "*.onnx")])
        if not paths:
            return
        parts = [str(p) for p in split_models(self.model_var.get())]
        parts += [str(Path(p)) for p in paths]
        self.model_var.set("+".join(parts))

    def _clear_model(self) -> None:
        self.model_var.set("")

    def _on_model_changed(self) -> None:
        """Debounced chain hint refresh (inspect() opens sessions)."""
        if self._chain_job:
            try:
                self.after_cancel(self._chain_job)
            except Exception:
                pass
        self._chain_job = self.after(500, self._update_chain_hint)

    def _update_chain_hint(self) -> None:
        self._chain_job = None
        lbl = getattr(self, "chain_lbl", None)
        if lbl is None:
            return
        spec = self.model_var.get().strip()
        if not spec:
            lbl.configure(text="")
            return
        from ..engines.onnx_engine import split_models

        parts = split_models(spec)
        if len(parts) > 1:
            lbl.configure(
                text_color="#6fcf97",
                text=f"Chain of {len(parts)} stages: "
                     + " → ".join(p.stem for p in parts))
            return
        hint = self._chain_cache.get(spec)
        if hint is None:
            try:
                from ..engines.onnx_engine import OnnxEngine

                kind = OnnxEngine.inspect(spec)
            except Exception:
                kind = "unreadable"
            hint = kind
            self._chain_cache[spec] = hint
        if "backbone" in hint or "head" in hint:
            lbl.configure(
                text_color="#e0a030",
                text=f"{parts[0].stem}: {hint}\nClick Chain… to add the "
                     "other stage.")
        else:
            lbl.configure(text_color="#7fb3d5", text=f"kind: {hint}")

    def _selected_model_spec(self) -> str | None:
        spec = self.model_var.get().strip()
        if spec:
            return spec
        messagebox.showinfo("Model", "Pick a model first (Browse or drag & drop).")
        return None

    def _model_info(self) -> None:
        spec = self._selected_model_spec()
        if not spec:
            return
        from .dialogs import text_dialog_async
        from ..engines.onnx_engine import OnnxEngine

        text_dialog_async(self, "Model info",
                          lambda: OnnxEngine.inspect_report(spec),
                          loading_text="Inspecting model I/O…")

    def _model_sidecar(self) -> None:
        spec = self._selected_model_spec()
        if not spec:
            return
        from ..engines.onnx_engine import OnnxEngine, split_models

        side = split_models(spec)[0].with_suffix(".json")
        if not side.exists():
            try:
                side.write_text(OnnxEngine.sidecar_template(spec),
                                encoding="utf-8")
            except Exception as exc:
                messagebox.showerror("Sidecar", str(exc))
                return
            self._log(f"sidecar created: {side.name}")
        try:
            os.startfile(side)
        except Exception:
            pass
        messagebox.showinfo(
            "Sidecar",
            f"{side}\n\nEdit the JSON, save, then use 'Test' to verify. "
            "Delete it to go back to auto-detection.")
        self._chain_cache.clear()
        self._refresh_model_list()

    def _open_models_folder(self) -> None:
        try:
            os.startfile(_models_dir())
        except Exception as exc:
            messagebox.showerror("Models folder", str(exc))

    def _model_test(self) -> None:
        spec = self._selected_model_spec()
        if not spec:
            return
        settings = self._gather_settings()
        if settings is None:
            return
        self.progress_lbl.configure(text="Testing model on 2s of synthetic audio…")
        self._log(f"testing model: {spec}")

        def work():
            try:
                import numpy as np
                import time as _t
                from ..engines.onnx_engine import OnnxEngine
                from ..presets import get_preset

                eng = OnnxEngine(spec, settings.device_id,
                                 use_dml=settings.use_dml)
                t = np.arange(2 * 22050, dtype=np.float64) / 22050.0
                x = (0.5 * np.sin(2 * np.pi * 440 * t)
                     + 0.2 * np.sin(2 * np.pi * 1760 * t)).astype(np.float32)
                t0 = _t.perf_counter()
                res = eng.process(x[None, :], 22050, 48000,
                                  get_preset("reference"))
                dt = _t.perf_counter() - t0
                rt = (res.audio.shape[-1] / res.sr) / max(dt, 1e-9)
                self._msg_q.put(("info", (
                    f"Model '{eng.model_path.name}' is working.\n\n"
                    f"kind: {eng.kind}\n"
                    f"speed: 2 s of audio in {dt:.1f} s "
                    f"({rt:.2f}x realtime on "
                    f"{'GPU (DirectML)' if settings.use_dml else 'CPU'})\n"
                    f"output: {res.sr} Hz, {res.audio.shape[0]} ch")))
            except Exception as exc:
                self._msg_q.put(("error", f"Model test failed:\n{exc}"))

        threading.Thread(target=work, daemon=True).start()

    def _browse_outdir(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.outdir_var.set(d)

    def _open_output(self) -> None:
        if self._out_dir and self._out_dir.is_dir():
            os.startfile(self._out_dir)

    def _gather_settings(self) -> JobSettings | None:
        sr_raw = self.sr_var.get().replace(",", "").strip()
        try:
            sr = 0 if sr_raw.lower().startswith("keep") else int(sr_raw)
        except ValueError:
            messagebox.showerror("Invalid", "Sample rate is invalid")
            return None
        lufs = None
        try:
            lufs = float(self.lufs_var.get()) if self.loud_var.get() else None
        except ValueError:
            pass

        out_dir = self.outdir_var.get().strip()
        out_path = Path(out_dir) if out_dir else None

        bit_raw = self.bit_var.get()
        bit = 32 if "32" in bit_raw else (16 if "16" in bit_raw else 24)

        def _float(var, default=0.0):
            try:
                return float(var.get())
            except (ValueError, AttributeError):
                return default

        def _int(var, default=1):
            try:
                return int(var.get())
            except (ValueError, AttributeError):
                return default

        return JobSettings(
            target_sr=sr,
            fmt=self._fmt_key(),
            bit_depth=bit,
            engine_id=self._engine_id(),
            model_path=self.model_var.get().strip() or None,
            device_id=int(self.device_var.get() or 0),
            preset=self.preset_var.get(),
            lufs_enabled=lufs is not None,
            lufs_target=lufs if lufs is not None else -14.0,
            dither=self.dither_var.get(),
            out_dir=out_path,
            suffix="_upscaled",
            save_spectrograms=self.spec_var.get(),
            denoise=_float(self.denoise_var, 0.0),
            declip=self.declip_var.get(),
            mid_side=self.ms_var.get(),
            ultra_resample=self.ultra_var.get(),
            auto_preset=self.auto_preset_var.get(),
            name_template=self.template_var.get() or "{name}{suffix}",
            keep_tree=self.keeptree_var.get(),
            workers=max(1, min(8, _int(self.workers_var, 1))),
            copy_metadata=not self.nometa_var.get(),
            use_dml=self.use_dml_var.get(),
            skip_existing=self.skip_exist_var.get(),
        )

    # ------------------------------------------------- About-tab actions
    def _show_diagnostics(self) -> None:
        from .dialogs import text_dialog_async
        from ..providers import diagnostics

        text_dialog_async(self, "Diagnostics", diagnostics,
                          loading_text="Probing hardware, runtimes…")

    def _show_requirements(self) -> None:
        from .dialogs import text_dialog
        from ..recommendations import requirements_text

        text_dialog(self, "GPU backend requirements", requirements_text())

    def _check_updates(self) -> None:
        def work():
            from ..updater import check_for_update

            cur, latest, tag = check_for_update()
            self._msg_q.put(("update", cur, latest))

        self.update_lbl.configure(text="checking…", text_color="#999")
        threading.Thread(target=work, daemon=True).start()

    def _download_starter(self) -> None:
        def work():
            from ..updater import ensure_starter_model

            p = ensure_starter_model(log=lambda s: None)
            self._msg_q.put(("starter", str(p) if p else ""))

        self.update_lbl.configure(text="downloading starter model…",
                                  text_color="#999")
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------ Estimate / Compare
    def _show_estimate(self) -> None:
        from .dialogs import text_dialog_async
        from ..pipeline import collect_audio_files

        files = collect_audio_files(self._files)
        if not files:
            messagebox.showinfo("Estimate", "Queue is empty.")
            return
        fmt = self._fmt_key()
        tmpl = self.template_var.get() or "{name}{suffix}"
        outdir = self.outdir_var.get().strip()
        preset = self.preset_var.get()

        def load():
            from ..audio_io import read_audio
            from ..dsp import estimate_bandwidth
            from ..pipeline import JobSettings, output_path_for

            probe = JobSettings(fmt=fmt, name_template=tmpl,
                                out_dir=Path(outdir) if outdir else None)
            out = [f"{'file':<36} {'in sr':>7} {'dur':>7} {'ch':>3} "
                   f"{'bandwidth':>9}  planned output"]
            for f in files:
                try:
                    x, sr = read_audio(f)
                    bw = estimate_bandwidth(x, sr)
                    planned = output_path_for(f, probe, None, preset,
                                              dedupe=False)
                    note = " (exists)" if planned.exists() else ""
                    out.append(
                        f"{f.name[:36]:<36} {sr:>7} {x.shape[-1] / sr:>5.1f}s "
                        f"{x.shape[0]:>3} {bw / 1000:>7.1f}k  "
                        f"{planned.name}{note}")
                except Exception as exc:
                    out.append(f"{f.name[:36]:<36} ERROR: {exc}")
            return "\n".join(out)

        text_dialog_async(self, "Estimate", load)

    def _compare_presets(self) -> None:
        sel = self.file_listbox.curselection()
        if sel:
            src = self._files[sel[0]]
        elif self._files:
            src = self._files[0]
        else:
            messagebox.showinfo("Compare", "Add a file first.")
            return
        if src.is_dir():
            messagebox.showinfo("Compare", "Select a file (not a folder).")
            return
        settings = self._gather_settings()
        if settings is None:
            return
        threading.Thread(target=self._compare_worker,
                         args=(src, settings), daemon=True).start()
        self.progress_lbl.configure(text="Comparing presets…")

    def _compare_worker(self, src: Path, settings: JobSettings) -> None:
        try:
            from ..pipeline import process_file
            from ..presets import PRESETS

            engine = create_engine(settings.engine_id, settings.model_path,
                                   settings.device_id,
                                   use_dml=settings.use_dml)
            reports = []
            total = len(PRESETS)
            for i, key in enumerate(PRESETS):
                s = JobSettings(**{**settings.__dict__, "preset": key,
                                   "suffix": f"_{key}",
                                   "save_spectrograms": False,
                                   "out_dir": settings.out_dir or src.parent})
                reports.append(process_file(src, s, engine, None, i, total))
                self._msg_q.put(("progress", i, total, 1.0,
                                 f"compare:{key}"))
            ok = sum(1 for r in reports if r.ok)
            self._msg_q.put(("info", f"{ok}/{total} preset variants written "
                                     f"next to the file."))
        except Exception as exc:
            self._msg_q.put(("error", str(exc)))

    # --------------------------------------------------------------- watch
    def _toggle_watch(self) -> None:
        if self.watch_var.get():
            if not self.watch_dir:
                self._pick_watch_dir()
            if not self.watch_dir:
                self.watch_var.set(False)
                return
            self._watch_seen = {str(f).lower() for f in self._all_audio(self.watch_dir)}
            self._watch_job = self.after(3000, self._watch_tick)
            self.progress_lbl.configure(text=f"Watching {self.watch_dir}")
            self._log(f"watch folder: {self.watch_dir}")
        else:
            if self._watch_job:
                self.after_cancel(self._watch_job)
                self._watch_job = None
            self.progress_lbl.configure(text="Watch stopped")
            self._log("watch stopped")

    def _pick_watch_dir(self) -> None:
        d = filedialog.askdirectory(title="Pick watch folder")
        if d:
            self.watch_dir = Path(d)

    @staticmethod
    def _all_audio(folder: Path):
        from ..audio_io import INPUT_EXTENSIONS

        return [p for p in folder.rglob("*") if p.is_file()
                and p.suffix.lower() in INPUT_EXTENSIONS]

    def _watch_tick(self) -> None:
        if not self.watch_var.get() or not self.watch_dir:
            return
        added = 0
        try:
            for p in self._all_audio(self.watch_dir):
                key = str(p).lower()
                if key not in self._watch_seen:
                    self._watch_seen.add(key)
                    self._files.append(p)
                    added += 1
            if added:
                self._refresh_list()
                self._log(f"watch: +{added} new file(s)")
        except Exception:
            pass
        try:
            idle = self._worker is None or not self._worker.is_alive()
        except Exception:
            idle = True
        if added and self.watch_auto_var.get() and idle and self._files:
            self._log("watch: auto-starting batch")
            self._start_batch()
        self._watch_job = self.after(3000, self._watch_tick)

    # ------------------------------------------------------------------ actions
    def _start_batch(self) -> None:
        if not self._files:
            messagebox.showinfo("Nothing to do", "Add audio files or folders first.")
            return
        settings = self._gather_settings()
        if settings is None:
            return
        if settings.engine_id == "onnx" and not settings.model_path:
            messagebox.showerror("Missing model",
                                 "The ONNX engine needs a .onnx model file.")
            return
        if settings.out_dir is None:
            settings.out_dir = self._files[0].parent if self._files[0].is_file() else self._files[0]
        self._out_dir = settings.out_dir

        from .settings_store import save_settings

        save_settings(self._collect_prefs())

        self._cancel.clear()
        self._force.clear()
        self._batch_t0 = time.time()
        self._eta_prev = None
        self._eta_speed = None
        self._failed = []
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.force_btn.configure(state="normal")
        self.progress.configure(progress_color="#1f6aa5")
        self.progress.set(0.0)
        self.file_progress.set(0.0)
        self.progress_lbl.configure(text="Starting…")
        self._log(f"started: {len(self._files)} item(s), engine="
                  f"{settings.engine_id}, sr={settings.target_sr or 'source'}, "
                  f"fmt={settings.fmt}, workers={settings.workers}")

        self._worker = threading.Thread(
            target=self._worker_run, args=(list(self._files), settings), daemon=True)
        self._worker.start()

    def _worker_run(self, files: list[Path], settings: JobSettings) -> None:
        from ..pipeline import collect_audio_with_roots

        flat, roots = collect_audio_with_roots(files)
        if not flat:
            self._msg_q.put(("error", "No supported audio files found."))
            return

        def cb(i, total, frac, msg, path):
            self._msg_q.put(("progress", i, total, frac, msg, str(path)))

        try:
            summary = run_batch(flat, settings, progress=cb,
                                cancel=self._cancel, roots=roots,
                                force=self._force)
            self._msg_q.put(("done", summary))
        except Exception as exc:
            self._msg_q.put(("error", str(exc)))

    def _cancel_batch(self) -> None:
        self._cancel.set()
        self.progress_lbl.configure(text="Finishing current file…")
        self._log("cancel: finishing current file, then stopping")

    def _force_stop_batch(self) -> None:
        self._force.set()
        self._cancel.set()
        self._files.clear()
        self._refresh_list()
        self.progress_lbl.configure(text="Force stopping now…")
        self._log("FORCE STOP: aborting current file, queue cleared")

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._msg_q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, i, total, frac, text, fname = msg
                    overall = (i + min(max(frac, 0.0), 1.0)) / max(1, total)
                    self.progress.set(overall)
                    self.file_progress.set(frac)
                    now = time.time()
                    if self._eta_prev is not None:
                        d_over = overall - self._eta_prev[0]
                        d_t = now - self._eta_prev[1]
                        if d_over > 1e-6 and d_t > 0.25:
                            sp = d_over / d_t
                            self._eta_speed = (sp if self._eta_speed is None
                                               else 0.7 * self._eta_speed + 0.3 * sp)
                    eta_txt = ""
                    if self._eta_speed and self._eta_speed > 1e-6 and overall < 1.0:
                        rem = int((1.0 - overall) / self._eta_speed)
                        m, s = divmod(rem, 60)
                        h, m = divmod(m, 60)
                        eta_txt = (f"  ·  ETA {h:d}:{m:02d}:{s:02d}" if h
                                   else f"  ·  ETA {m:d}:{s:02d}")
                    el = int(now - self._batch_t0) if self._batch_t0 else 0
                    self._eta_prev = (overall, now)
                    self.progress_lbl.configure(
                        text=f"File {i + 1}/{total} — {text}  ·  this file "
                             f"{frac * 100:.0f}%  ·  total {overall * 100:.0f}%"
                             f"{eta_txt}  ·  elapsed {el // 60:d}:{el % 60:02d}")
                    self.file_lbl.configure(text=fname)
                    if frac >= 1.0:
                        self._log(f"{text.lower()}: {Path(fname).name}")
                elif kind == "done":
                    self._summary = msg[1]
                    self.start_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self.force_btn.configure(state="disabled")
                    self.open_btn.configure(state="normal")
                    ok = self._summary.succeeded
                    tot = len(self._summary.reports)
                    self.progress.set(1.0)
                    self.file_progress.set(0.0)
                    self._failed = [r.source for r in self._summary.reports
                                    if not r.ok and not r.cancelled]
                    skipped = sum(1 for r in self._summary.reports
                                  if r.ok and r.message.startswith("skipped"))
                    if self._summary.cancelled:
                        self.progress.configure(progress_color="#b98900")
                        self.progress_lbl.configure(
                            text=f"Stopped: {ok}/{tot} processed")
                        self.file_lbl.configure(text="")
                        self._log(f"stopped: {ok}/{tot} processed, "
                                  f"{len(self._failed)} failed")
                    else:
                        color = "#8a2b2b" if self._failed else "#2b8a3e"
                        self.progress.configure(progress_color=color)
                        self.progress_lbl.configure(
                            text=f"Done: {ok}/{tot} succeeded"
                                 + (f", {skipped} skipped" if skipped else ""))
                        self.file_lbl.configure(text="")
                        body = f"{ok}/{tot} files processed."
                        if skipped:
                            body += f"\n{skipped} skipped (output existed)."
                        if self._failed:
                            body += "\n\nFailed:\n" + "\n".join(
                                f"  {r.source.name}: {r.message}"
                                for r in self._summary.reports
                                if not r.ok and not r.cancelled)
                        messagebox.showinfo("Batch finished", body)
                        self._log(f"finished: {ok}/{tot} ok"
                                  + (f", {skipped} skipped" if skipped else "")
                                  + (f", {len(self._failed)} failed"
                                     if self._failed else ""))
                    if self.sound_var.get():
                        try:
                            import winsound

                            winsound.MessageBeep(winsound.MB_ICONASTERISK)
                        except Exception:
                            pass
                elif kind == "error":
                    self.start_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self.force_btn.configure(state="disabled")
                    self.progress.configure(progress_color="#8a2b2b")
                    self.progress_lbl.configure(text=f"Error: {msg[1]}")
                    self._log(f"ERROR: {msg[1]}")
                    messagebox.showerror("Error", msg[1])
                elif kind == "preview":
                    self._show_preview(msg[1], msg[2])
                elif kind == "preview_err":
                    messagebox.showerror("Preview failed", msg[1])
                elif kind == "reco":
                    try:
                        self._reco_box.configure(state="normal")
                        self._reco_box.delete("0.0", "end")
                        self._reco_box.insert("0.0", msg[1])
                        self._reco_box.configure(state="disabled")
                    except Exception:
                        pass
                elif kind == "update":
                    _, cur, latest = msg
                    if latest:
                        self.update_lbl.configure(
                            text=f"Update available: v{latest} (you have {cur})",
                            text_color="#e6c07b")
                    else:
                        self.update_lbl.configure(text=f"Up to date (v{cur})",
                                                  text_color="#8fd18b")
                elif kind == "starter":
                    if msg[1]:
                        self.update_lbl.configure(text=f"Model: {Path(msg[1]).name}",
                                                  text_color="#8fd18b")
                        self._refresh_model_list()
                    else:
                        self.update_lbl.configure(text="starter download failed",
                                                  text_color="#e07b7b")
                elif kind == "info":
                    messagebox.showinfo("Info", msg[1])
                elif kind == "models":
                    _, items, mapping = msg
                    self._model_pick_map = mapping
                    self.model_pick.configure(values=items)
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    def _show_recommendations(self) -> None:
        top = ctk.CTkToplevel(self)
        top.title("Model recommendations")
        top.geometry("780x600")
        box = ctk.CTkTextbox(top, wrap="word", font=ctk.CTkFont(size=12))
        box.pack(fill="both", expand=True, padx=10, pady=10)
        box.insert("0.0", "Detecting GPU VRAM…")
        box.configure(state="disabled")
        self._reco_box = box

        def work():
            from ..recommendations import recommendation_report
            self._msg_q.put(("reco", recommendation_report()))

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ preview
    def _preview(self) -> None:
        sel = self.file_listbox.curselection()
        if sel:
            src = self._files[sel[0]]
        elif self._files:
            src = self._files[0]
        else:
            messagebox.showinfo("Preview", "Add a file first.")
            return
        if src.is_dir():
            messagebox.showinfo("Preview", "Select a file (not a folder) to preview.")
            return
        threading.Thread(target=self._preview_worker, args=(src,), daemon=True).start()

    def _preview_worker(self, src: Path) -> None:
        try:
            settings = self._gather_settings()
            if settings is None:
                return
            tmpdir = Path(tempfile.mkdtemp(prefix="au_preview_"))
            engine = create_engine(settings.engine_id, settings.model_path,
                                   settings.device_id,
                                   use_dml=settings.use_dml)
            preset = PRESETS[settings.preset]

            from ..audio_io import read_audio
            x, sr = read_audio(src)
            n = x.shape[-1]
            cap = int(sr * 25)
            if n > cap * 2:
                start = (n - cap) // 2
                x = x[:, start: start + cap]
            result = engine.process(x, sr, settings.target_sr, preset)

            png = tmpdir / "preview.png"
            save_comparison(x, sr, result.audio, result.sr, png,
                            title=src.name)

            wav_out = tmpdir / f"{src.stem}_preview.flac"
            from ..audio_io import write_audio
            write_audio(wav_out, result.audio, result.sr, "FLAC",
                        settings.bit_depth, settings.dither)

            self._msg_q.put(("preview", str(png), str(wav_out)))
        except Exception as exc:
            self._msg_q.put(("preview_err", str(exc)))

    def _show_preview(self, png: str, audio: str) -> None:
        top = ctk.CTkToplevel(self)
        top.title("Preview — before / after")
        top.grab_set()
        try:
            from PIL import Image
            img = Image.open(png)
            maxw, maxh = 1000, 600
            w, h = img.size
            scale = min(maxw / w, maxh / h, 1.0)
            nw, nh = int(w * scale), int(h * scale)
            img_r = img.resize((nw, nh))
            ctk_img = ctk.CTkImage(light_image=img_r, dark_image=img_r,
                                   size=(nw, nh))
            lbl = ctk.CTkLabel(top, image=ctk_img, text="")
            lbl.image = ctk_img
            lbl.pack(padx=10, pady=10)

            btns = ctk.CTkFrame(top, fg_color="transparent")
            btns.pack(pady=(0, 10))
            ctk.CTkButton(btns, text="Play upscaled",
                          command=lambda: os.startfile(audio)).pack(side="left", padx=6)
            ctk.CTkButton(btns, text="Open folder",
                          command=lambda: os.startfile(str(Path(audio).parent))).pack(side="left", padx=6)
        except Exception as exc:
            ctk.CTkLabel(top, text=f"Could not show image: {exc}").pack(padx=20, pady=20)
        top.lift()


def launch() -> None:
    """Create the main window and run the Tk event loop."""
    App().mainloop()


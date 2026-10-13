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
_SR_CHOICES = ["44,100", "48,000", "88,200", "96,000", "176,400", "192,000"]
_LOSSY = {"MP3", "OGG", "OPUS"}


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self._msg_q: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._files: list[Path] = []
        self._formats = available_output_formats()
        self._summary = None
        self._out_dir: Path | None = None

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
        ).grid(row=0, column=0, columnspan=4, padx=8, pady=(8, 4), sticky="w")

        list_holder = ctk.CTkFrame(left, fg_color="transparent")
        list_holder.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=6)
        list_holder.grid_columnconfigure(0, weight=1)
        list_holder.grid_rowconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(
            list_holder, bg="#2b2b2b", fg="#e0e0e0", highlightthickness=0,
            selectbackground="#1f6aa5", activestyle="none",
        )
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
        hw = list_gpus()
        dev_lbl = ctk.CTkLabel(self.model_frame,
                               text="GPU device index (DirectML)")
        dev_lbl.pack(anchor="w", pady=(6, 0))
        self.device_var = tk.StringVar(value="0")
        ctk.CTkEntry(self.model_frame, textvariable=self.device_var, width=60).pack(anchor="w")
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
        ctk.CTkButton(
            self.model_frame, text="Model recommendations (VRAM-aware)",
            command=self._show_recommendations,
        ).pack(anchor="w", pady=(6, 0))

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
        self.preview_btn = ctk.CTkButton(act, text="Preview", width=80,
                                         command=self._preview)
        self.preview_btn.grid(row=0, column=3, padx=4)
        self.open_btn = ctk.CTkButton(act, text="Open output folder", width=130,
                                      command=self._open_output, state="disabled")
        self.open_btn.grid(row=0, column=4, padx=4)

        self.progress = ctk.CTkProgressBar(bottom)
        self.progress.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(2, 2))
        self.progress.set(0.0)
        self.progress_lbl = ctk.CTkLabel(bottom, text="Ready", text_color="#aaa")
        self.progress_lbl.grid(row=3, column=0, columnspan=3, sticky="w", padx=8)
        self.file_lbl = ctk.CTkLabel(bottom, text="", text_color="#888",
                                     font=ctk.CTkFont(size=11))
        self.file_lbl.grid(row=4, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))

        self._on_engine_change(self.engine_label_var.get())
        self._on_fmt_change(None)

    # ------------------------------------------------------------------ helpers
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
        for brace, bare in paths:
            p = brace or bare
            if p:
                self._add_paths([Path(p)])

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

    def _remove_selected(self) -> None:
        sel = list(self.file_listbox.curselection())
        for i in reversed(sel):
            del self._files[i]
        self._refresh_list()

    def _clear_files(self) -> None:
        self._files.clear()
        self.file_listbox.delete(0, "end")

    def _browse_model(self) -> None:
        p = filedialog.askopenfilename(
            title="Select ONNX model", filetypes=[("ONNX model", "*.onnx")])
        if p:
            self.model_var.set(p)

    def _browse_outdir(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.outdir_var.set(d)

    def _open_output(self) -> None:
        if self._out_dir and self._out_dir.is_dir():
            os.startfile(self._out_dir)

    def _gather_settings(self) -> JobSettings | None:
        try:
            sr = int(self.sr_var.get().replace(",", ""))
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
        )

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

        self._cancel.clear()
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0.0)
        self.progress_lbl.configure(text="Starting…")

        self._worker = threading.Thread(
            target=self._worker_run, args=(list(self._files), settings), daemon=True)
        self._worker.start()

    def _worker_run(self, files: list[Path], settings: JobSettings) -> None:
        from ..pipeline import collect_audio_files
        flat = collect_audio_files(files)
        if not flat:
            self._msg_q.put(("error", "No supported audio files found."))
            return

        def cb(i, total, frac, msg, path):
            self._msg_q.put(("progress", i, total, frac, msg))

        try:
            summary = run_batch(flat, settings, progress=cb, cancel=self._cancel)
            self._msg_q.put(("done", summary))
        except Exception as exc:
            self._msg_q.put(("error", str(exc)))

    def _cancel_batch(self) -> None:
        self._cancel.set()
        self.progress_lbl.configure(text="Cancelling after current file…")

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._msg_q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, i, total, frac, text = msg
                    self.progress.set(frac)
                    self.progress_lbl.configure(
                        text=f"File {i + 1}/{total} — {text} ({frac * 100:.0f}%)")
                    self.file_lbl.configure(
                        text=str(self._files[i]) if i < len(self._files) else "")
                elif kind == "done":
                    self._summary = msg[1]
                    self.start_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self.open_btn.configure(state="normal")
                    ok = self._summary.succeeded
                    tot = len(self._summary.reports)
                    self.progress.set(1.0)
                    self.progress_lbl.configure(text=f"Done: {ok}/{tot} succeeded")
                    self.file_lbl.configure(text="")
                    failed = [r for r in self._summary.reports if not r.ok]
                    body = f"{ok}/{tot} files processed."
                    if failed:
                        body += "\n\nFailed:\n" + "\n".join(
                            f"  {r.source.name}: {r.message}" for r in failed)
                    messagebox.showinfo("Batch finished", body)
                elif kind == "error":
                    self.start_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self.progress_lbl.configure(text=f"Error: {msg[1]}")
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
                                   settings.device_id)
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


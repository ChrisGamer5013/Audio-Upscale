"""Reusable text dialogs for the GUI (diagnostics, estimate, update info)."""
from __future__ import annotations

import threading
from pathlib import Path

import customtkinter as ctk


def text_dialog(parent, title: str, text: str, geometry: str = "760x560") -> ctk.CTkToplevel:
    top = ctk.CTkToplevel(parent)
    top.title(title)
    top.geometry(geometry)
    top.grid_columnconfigure(0, weight=1)
    top.grid_rowconfigure(0, weight=1)
    box = ctk.CTkTextbox(top, wrap="word", font=ctk.CTkFont(size=12))
    box.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=(10, 4))
    box.insert("0.0", text)
    box.configure(state="disabled")

    def copy():
        top.clipboard_clear()
        top.clipboard_append(text)

    ctk.CTkButton(top, text="Copy to clipboard", command=copy).grid(
        row=1, column=0, padx=10, pady=(0, 10), sticky="w"
    )
    ctk.CTkButton(top, text="Close", command=top.destroy).grid(
        row=1, column=1, padx=10, pady=(0, 10), sticky="e"
    )
    top.lift()
    return top


def text_dialog_async(parent, title: str, loader, loading_text: str = "Loading…"):
    """Open dialog with placeholder; loader() runs in a thread, then fills in."""
    top = text_dialog(parent, title, loading_text)

    def work():
        try:
            txt = loader()
        except Exception as exc:  # always show something
            txt = f"Failed: {exc}"
        def fill():
            if top.winfo_exists():
                for child in top.winfo_children():
                    if isinstance(child, ctk.CTkTextbox):
                        child.configure(state="normal")
                        child.delete("0.0", "end")
                        child.insert("0.0", txt)
                        child.configure(state="disabled")
        top.after(0, fill)

    threading.Thread(target=work, daemon=True).start()
    return top

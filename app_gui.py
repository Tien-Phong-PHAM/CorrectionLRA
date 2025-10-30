#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CorrectionLRA – simple GUI wrapper for LRA correction.

- Select input: a single TIFF file OR a folder of TIFF files
- Select the CSV file containing LRA coefficients (columns: L_mm,A_R,B_R,A_G,B_G,A_B,B_B)
- Choose settings (DPI, axis, optional offset)
- Outputs are saved into a "Correction_RLA" subfolder next to the input,
  using the same base filename plus the suffix "_CorrectionLRA.tif".

This app reuses the logic from lra_correction.py.
"""

import os
import sys
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Ensure local module is importable
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    import lra_correction as lra
except Exception as e:
    messagebox.showerror("Import error", f"Could not import lra_correction.py:\n{e}")
    raise

# Check optional dependency early to give a clearer error
try:
    import tifffile  # noqa: F401
except Exception:
    # lra_correction will also error, but explain nicely here
    pass

def is_tiff(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CorrectionLRA")
        self.geometry("720x520")
        self.minsize(720, 520)

        # Variables
        self.mode_var = tk.StringVar(value="file")  # 'file' or 'folder'
        self.input_path_var = tk.StringVar(value="")
        self.csv_path_var = tk.StringVar(value="")
        self.dpi_var = tk.StringVar(value="75.0")
        self.axis_var = tk.StringVar(value="columns")  # 'columns' or 'rows'
        self.offset_var = tk.StringVar(value="0.0")
        self.clip_min_var = tk.StringVar(value="0")
        self.clip_max_var = tk.StringVar(value="65535")

        # UI
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Input mode
        frame_mode = ttk.LabelFrame(self, text="Input")
        frame_mode.pack(fill="x", **pad)

        ttk.Radiobutton(frame_mode, text="Single file", variable=self.mode_var, value="file").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Radiobutton(frame_mode, text="Folder", variable=self.mode_var, value="folder").grid(row=0, column=1, sticky="w", padx=8, pady=4)

        # Input path
        frame_in = ttk.Frame(frame_mode)
        frame_in.grid(row=1, column=0, columnspan=3, sticky="ew", padx=4, pady=4)
        frame_in.columnconfigure(1, weight=1)

        ttk.Label(frame_in, text="TIFF file / Folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame_in, textvariable=self.input_path_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frame_in, text="Browse…", command=self.browse_input).grid(row=0, column=2)

        # CSV path
        frame_csv = ttk.Frame(frame_mode)
        frame_csv.grid(row=2, column=0, columnspan=3, sticky="ew", padx=4, pady=4)
        frame_csv.columnconfigure(1, weight=1)

        ttk.Label(frame_csv, text="CSV coefficients:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame_csv, textvariable=self.csv_path_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frame_csv, text="Browse…", command=self.browse_csv).grid(row=0, column=2)

        # Parameters
        frame_params = ttk.LabelFrame(self, text="Parameters")
        frame_params.pack(fill="x", **pad)
        for i in range(4):
            frame_params.columnconfigure(i, weight=1)

        ttk.Label(frame_params, text="DPI").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame_params, textvariable=self.dpi_var).grid(row=1, column=0, sticky="ew", padx=6)

        ttk.Label(frame_params, text="Axis").grid(row=0, column=1, sticky="w")
        ttk.Combobox(frame_params, textvariable=self.axis_var, values=["columns", "rows"], state="readonly").grid(row=1, column=1, sticky="ew", padx=6)

        ttk.Label(frame_params, text="x_offset_px").grid(row=0, column=2, sticky="w")
        ttk.Entry(frame_params, textvariable=self.offset_var).grid(row=1, column=2, sticky="ew", padx=6)

        ttk.Label(frame_params, text="Clip [min, max]").grid(row=0, column=3, sticky="w")
        frame_clip = ttk.Frame(frame_params)
        frame_clip.grid(row=1, column=3, sticky="ew", padx=6)
        ttk.Entry(frame_clip, width=8, textvariable=self.clip_min_var).pack(side="left", fill="x")
        ttk.Label(frame_clip, text=" to ").pack(side="left")
        ttk.Entry(frame_clip, width=8, textvariable=self.clip_max_var).pack(side="left", fill="x")

        # Actions
        frame_actions = ttk.Frame(self)
        frame_actions.pack(fill="x", **pad)

        ttk.Button(frame_actions, text="Run Correction", command=self.run).pack(side="left")
        ttk.Button(frame_actions, text="Open Output Folder", command=self.open_out_folder).pack(side="left", padx=12)

        # Log
        frame_log = ttk.LabelFrame(self, text="Log")
        frame_log.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(frame_log, height=12)
        self.log.pack(fill="both", expand=True)

    def browse_input(self):
        if self.mode_var.get() == "file":
            path = filedialog.askopenfilename(title="Select 48-bit RGB TIFF",
                                              filetypes=[("TIFF images", "*.tif *.tiff"), ("All files", "*.*")])
        else:
            path = filedialog.askdirectory(title="Select folder of TIFF images")
        if path:
            self.input_path_var.set(path)

    def browse_csv(self):
        path = filedialog.askopenfilename(title="Select CSV coefficients",
                                          filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv_path_var.set(path)

    def logln(self, msg):
        self.log.insert("end", str(msg) + "\n")
        self.log.see("end")
        self.update_idletasks()

    def open_out_folder(self):
        outdir = self._guess_out_dir()
        if not outdir:
            messagebox.showinfo("Info", "No output folder yet. Run the correction first or select a valid input.")
            return
        outdir = Path(outdir)
        if not outdir.exists():
            messagebox.showinfo("Info", f"Folder does not exist yet:\n{outdir}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(outdir))  # noqa: PLW1510
            elif sys.platform == "darwin":
                os.system(f'open "{outdir}"')
            else:
                os.system(f'xdg-open "{outdir}"')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder:\n{e}")

    def _guess_out_dir(self):
        inp = self.input_path_var.get().strip()
        if not inp:
            return ""
        p = Path(inp)
        if self.mode_var.get() == "file":
            return str(p.parent / "Correction_RLA")
        else:
            return str(p / "Correction_RLA")

    def _collect_inputs(self):
        mode = self.mode_var.get()
        input_path = Path(self.input_path_var.get().strip())
        csv_path = Path(self.csv_path_var.get().strip())
        # parse numbers
        dpi = float(self.dpi_var.get().strip())
        x_offset_px = float(self.offset_var.get().strip())
        clip_min = float(self.clip_min_var.get().strip())
        clip_max = float(self.clip_max_var.get().strip())
        axis = self.axis_var.get().strip()

        # sanity checks
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if mode == "file":
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            if not is_tiff(input_path):
                raise ValueError(f"Input file must be a TIFF (.tif/.tiff): {input_path}")
            items = [input_path]
            outdir = input_path.parent / "Correction_RLA"
        else:
            if not input_path.exists():
                raise FileNotFoundError(f"Folder not found: {input_path}")
            items = [p for p in input_path.iterdir() if p.is_file() and is_tiff(p)]
            outdir = input_path / "Correction_RLA"
        outdir.mkdir(parents=True, exist_ok=True)

        return {
            "mode": mode,
            "items": items,
            "csv_path": csv_path,
            "dpi": dpi,
            "axis": axis,
            "x_offset_px": x_offset_px,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "outdir": outdir,
        }

    def run(self):
        try:
            cfg = self._collect_inputs()
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self.log.delete("1.0", "end")
        self.logln("Loading coefficients…")
        try:
            L_meas, A_meas, B_meas = lra.load_coeffs(str(cfg["csv_path"]))
        except Exception as e:
            messagebox.showerror("CSV error", f"Failed to read coefficients:\n{e}")
            return
        self.logln(f"Read {len(L_meas)} measured L positions (mm) from CSV.")

        # Process each image
        for idx, img_path in enumerate(cfg["items"], 1):
            try:
                self.logln(f"\n[{idx}/{len(cfg['items'])}] Reading: {img_path.name}")
                img = lra.read_rgb16(str(img_path))
                H, W, _ = img.shape

                if cfg["axis"] == "columns":
                    A_axis, B_axis = lra.build_AB_per_axis(W, cfg["dpi"], L_meas, A_meas, B_meas, x_offset_px=cfg["x_offset_px"])
                else:
                    A_axis, B_axis = lra.build_AB_per_axis(H, cfg["dpi"], L_meas, A_meas, B_meas, x_offset_px=cfg["x_offset_px"])

                self.logln("Applying LRA…")
                out = lra.apply_lra(img, A_axis, B_axis, axis=cfg["axis"],
                                    clip_min=cfg["clip_min"], clip_max=cfg["clip_max"])

                out_name = img_path.stem + "_CorrectionLRA.tif"
                out_path = cfg["outdir"] / out_name

                self.logln(f"Saving to: {out_path}")
                lra.save_rgb16_like(str(img_path), str(out_path), out)

            except Exception as e:
                tb = traceback.format_exc()
                self.logln(f"ERROR processing {img_path.name}: {e}\n{tb}")
                messagebox.showerror("Processing error", f"{img_path.name}\n\n{e}")
                continue

        self.logln("\nDone.")
        messagebox.showinfo("Completed", "All tasks finished.")

if __name__ == "__main__":
    app = App()
    app.mainloop()

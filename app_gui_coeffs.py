#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoeffsAB‑LRA — GUI wrapper to compute A(L), B(L) from RAW RGB16 tables and plot them.

- Input: CSV, XLSX, XLSM (two-row header as required by compute_lra_coeffs_v3.py)
- Output folder: "CoeffsAB_LRA" next to the selected input file
- "Single" methods: OLS, WLS, weightedAvg, median (with R²)
"""

import os
import sys
from pathlib import Path
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import compute_lra_coeffs_v3 as lra
import plot_lra_AB as plotter

SINGLE_NAMES = ["OLS", "WLS", "weightedAvg", "median"]

def default_prefix_for(path: Path) -> str:
    return path.stem.replace(" ", "_")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CoeffsAB-LRA")
        self.geometry("760x620")

        self.input_path_var = tk.StringVar(value="\1")
        self.sheet_var = tk.StringVar(value="\1")
        self.prefix_var = tk.StringVar(value="\1")
        self.single_choice = tk.StringVar(value=SINGLE_NAMES[1])  # WLS by default
        self.auto_plot = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}
        f_in = ttk.LabelFrame(self, text="Input table")
        f_in.pack(fill="x", **pad)
        f_in.columnconfigure(1, weight=1)

        ttk.Label(f_in, text="CSV/XLSX/XLSM:").grid(row=0, column=0, sticky="w")
        ttk.Entry(f_in, textvariable=self.input_path_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(f_in, text="Browse…", command=self.browse_input).grid(row=0, column=2)

        ttk.Label(f_in, text="Excel sheet (optional):").grid(row=1, column=0, sticky="w")
        ttk.Entry(f_in, textvariable=self.sheet_var).grid(row=1, column=1, sticky="ew", padx=6)

        ttk.Label(f_in, text="Output prefix:").grid(row=2, column=0, sticky="w")
        ttk.Entry(f_in, textvariable=self.prefix_var).grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Button(f_in, text="Use input name", command=self.use_input_name).grid(row=2, column=2)

        f_opt = ttk.LabelFrame(self, text="Options")
        f_opt.pack(fill="x", **pad)
        ttk.Label(f_opt, text="Single A,B to plot:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(f_opt, values=SINGLE_NAMES, textvariable=self.single_choice, state="readonly", width=14).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Checkbutton(f_opt, text="Plot after compute", variable=self.auto_plot).grid(row=0, column=2, sticky="w", padx=6)

        f_actions = ttk.Frame(self)
        f_actions.pack(fill="x", **pad)
        ttk.Button(f_actions, text="Compute coefficients", command=self.run_compute).pack(side="left")
        ttk.Button(f_actions, text="Plot selected single A,B", command=self.run_plot_only).pack(side="left", padx=12)
        ttk.Button(f_actions, text="Open output folder", command=self.open_out_folder).pack(side="left", padx=12)

        f_log = ttk.LabelFrame(self, text="Log")
        f_log.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(f_log, height=16)
        self.log.pack(fill="both", expand=True)

    def browse_input(self):
        path = filedialog.askopenfilename(title="Select RAW table (CSV/XLSX/XLSM)",
                                          filetypes=[("Tables", "*.csv *.xlsx *.xlsm *.xls"), ("All files", "*.*")])
        if path:
            self.input_path_var.set(path)
            if not self.prefix_var.get():
                self.use_input_name()

    def use_input_name(self):
        p = Path(self.input_path_var.get().strip())
        if p.name:
            self.prefix_var.set(default_prefix_for(p))

    def logln(self, msg):
        self.log.insert("end", str(msg) + "\n")
        self.log.see("end")
        self.update_idletasks()

    def out_dir(self) -> Path:
        p = Path(self.input_path_var.get().strip())
        return (p.parent / "CoeffsAB_LRA") if p.name else (Path.cwd() / "CoeffsAB_LRA")

    def open_out_folder(self):
        outdir = self.out_dir()
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(outdir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{outdir}"')
            else:
                os.system(f'xdg-open "{outdir}"')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder:\n{e}")

    def run_compute(self):
        self.log.delete("1.0", "end")
        try:
            in_path = Path(self.input_path_var.get().strip())
            if not in_path.exists():
                raise FileNotFoundError(f"Input not found: {in_path}")
            sheet = self.sheet_var.get().strip() or None
            prefix = self.prefix_var.get().strip() or default_prefix_for(in_path)
            outdir = self.out_dir()
            outdir.mkdir(parents=True, exist_ok=True)

            self.logln(f"Reading RAW table: {in_path.name}  (sheet={sheet or 'auto/first'})")
            df_raw = lra.read_raw_table(str(in_path), sheet)
            self.logln(f"→ shape: {df_raw.shape}")

            self.logln("Computing per-dose A,B…")
            A_large, B_large = lra.compute_per_dose_AB(df_raw)
            lra.write_large_like_input(df_raw, A_large, str(outdir / f"{prefix}_AB_per_dose_A.csv"))
            lra.write_large_like_input(df_raw, B_large, str(outdir / f"{prefix}_AB_per_dose_B.csv"))

            self.logln("Writing channel splits…")
            lra.write_channel_splits(A_large, B_large, str(outdir), prefix)

            self.logln("Computing single A,B tables with R²…")
            singles = lra.compute_single_pairs_with_r2(df_raw, B_large)
            for name, tbl in singles.items():
                tbl.to_csv(outdir / f"{prefix}_single_AB_{name}_with_R2.csv", index=False)
                self.logln(f"  wrote: {prefix}_single_AB_{name}_with_R2.csv")

            if self.auto_plot.get():
                self.logln(f"Plotting using '{self.single_choice.get()}'…")
                self._plot_selected(outdir, prefix)

            messagebox.showinfo("Done", f"Outputs saved in:\n{outdir}")
        except Exception as e:
            tb = traceback.format_exc()
            self.logln(f"ERROR: {e}\n{tb}")
            messagebox.showerror("Error", str(e))

    def _plot_selected(self, outdir: Path, prefix: str):
        name = self.single_choice.get()
        csv = outdir / f"{prefix}_single_AB_{name}_with_R2.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Single-AB file not found for '{name}': {csv.name}")
        df = plotter.load_single_ab(str(csv))
        plotter.plot_coefficients(df, str(outdir), f"{prefix} [{name}]")
        self.logln("  wrote: A_vs_L.png, B_vs_L.png")

    def run_plot_only(self):
        try:
            outdir = self.out_dir()
            prefix = self.prefix_var.get().strip() or "LRA"
            self._plot_selected(outdir, prefix)
            messagebox.showinfo("Plots saved", f"Saved A_vs_L.png and B_vs_L.png in:\n{outdir}")
        except Exception as e:
            messagebox.showerror("Plot error", str(e))

if __name__ == "__main__":
    App().mainloop()

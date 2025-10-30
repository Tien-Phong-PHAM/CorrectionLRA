#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute LRA coefficients (A,B) from RAW 16-bit RGB tables and report R².
Now also splits per-dose A and B into 3 channel-specific CSVs each (R/G/B).

Input table (wide, two header rows):
  - Row 1: Dose (cGy) repeated: 0.0,0.0,0.0, 20.1,20.1,20.1, ...
  - Row 2: Channel for each dose: R,G,B, R,G,B, ...
  - First *data* column contains 'Position latérale (mm)' in the first cell,
    followed by lateral positions (mm), including 0 for the scanner center.

Pipeline:
  1) A,B per dose (two-point {0,D})  → two CSVs shaped like the input (A and B),
     + 6 channel-split CSVs: A_R/G/B and B_R/G/B (rows=L_mm, cols=doses).
  2) Single pair (A,B) per position & channel via:
     - OLS (unweighted linear regression)
     - WLS (weighted linear regression, w_D = (x_D - x_0)^2)
     - mean of B(D) across doses (A anchored at D=0)
     - median of B(D) across doses (A anchored at D=0)
     Each single-pair CSV includes R² columns per channel (R2_R, R2_G, R2_B).
"""

import argparse
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd


# ---------- Utils ----------
def _to_float_or_none(x):
    try:
        return float(str(x).replace(",", ".").strip())
    except Exception:
        return None


# ---------- Robust reader for CSV/XLSX/XLSM ----------
def read_raw_table(path: str, sheet: str | None = None) -> pd.DataFrame:
    """
    Read a RAW table with a two-row header and a first column carrying 'Position latérale (mm)'.
    Returns a DataFrame indexed by L_mm and with MultiIndex columns: (Dose (cGy), Channel).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".xlsx", ".xlsm", ".xls"):
        engine = "openpyxl" if ext in (".xlsx", ".xlsm") else None
        if sheet is None:
            xl = pd.ExcelFile(path, engine=engine)
            sheet = xl.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet, header=[0, 1], engine=engine)
    else:
        # CSV with two header rows
        df = pd.read_csv(path, header=[0, 1])

    # First column contains the label + L values; drop the label row if needed
    L_series = df.iloc[:, 0].tolist()
    start_idx = 1 if _to_float_or_none(L_series[0]) is None else 0
    L_vals = [_to_float_or_none(v) for v in L_series[start_idx:]]
    if any(v is None for v in L_vals):
        raise ValueError("Impossible de convertir certaines positions L en float.")

    # Remove that first column and the header-like row
    df = df.drop(columns=df.columns[0]).iloc[start_idx:, :].reset_index(drop=True)

    # Build clean MultiIndex: (Dose float, Channel in {R,G,B})
    doses, chans = [], []
    for a, b in df.columns:
        doses.append(float(str(a).replace(",", ".")))
        chans.append(str(b).strip().upper())
    df.columns = pd.MultiIndex.from_arrays([doses, chans], names=["Dose (cGy)", "Channel"])

    # Set index = L_mm
    df.index = [float(v) for v in L_vals]
    df.index.name = "L_mm"

    # Ensure numeric and sorted by L
    df = df.apply(pd.to_numeric, errors="coerce").sort_index()

    # Quick sanity checks
    if 0.0 not in df.index:
        raise ValueError("L=0 introuvable : la ligne centrale est nécessaire pour ancrer A(0)=0, B(0)=1.")
    expected = {"R", "G", "B"}
    got = set(df.columns.get_level_values("Channel"))
    if not expected.issubset(got):
        raise ValueError(f"Canaux manquants. Attendus: {expected}, trouvés: {got}")

    return df


# ---------- Write helper (preserve 'wide' shape) ----------
def write_large_like_input(input_like: pd.DataFrame, matrix: pd.DataFrame, out_path: str) -> None:
    matrix = matrix.reindex(index=input_like.index, columns=input_like.columns)
    matrix.to_csv(out_path, index=True)


# ---------- Per-dose A,B (two-point {0,D}) ----------
def compute_per_dose_AB(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (Dose D, Channel ch), compute A_D(L), B_D(L) using:
      B = (yD - y0) / (xD - x0),  A = y0 - B*x0
    where yD = R_center(D), y0 = R_center(0), xD = R_lateral(D, L), x0 = R_lateral(0, L).
    Enforce A(0)=0, B(0)=1.
    """
    cols = df_raw.columns
    doses = list(dict.fromkeys([c[0] for c in cols]))  # preserve order
    L0 = 0.0

    A_large = pd.DataFrame(index=df_raw.index, columns=cols, dtype=float)
    B_large = pd.DataFrame(index=df_raw.index, columns=cols, dtype=float)

    for D, ch in cols:
        RL = df_raw[(D, ch)]             # xD(L): lateral at dose D
        RL0 = df_raw[(doses[0], ch)]     # x0(L): lateral at dose 0
        yD = float(df_raw.loc[L0, (D, ch)])         # center at D
        y0 = float(df_raw.loc[L0, (doses[0], ch)])  # center at 0

        denom = RL - RL0
        with np.errstate(divide='ignore', invalid='ignore'):
            B = (yD - y0) / denom
        A = y0 - B * RL0

        # exact center constraints
        A.loc[L0] = 0.0
        B.loc[L0] = 1.0

        A_large[(D, ch)] = A
        B_large[(D, ch)] = B

    return A_large, B_large


# ---------- Channel split helpers (new) ----------
def write_channel_splits(A_large: pd.DataFrame, B_large: pd.DataFrame,
                         out_dir: str, prefix: str) -> None:
    """
    Write 6 CSVs:
      prefix_AB_per_dose_A_R/G/B.csv and prefix_AB_per_dose_B_R/G/B.csv
    Each file: rows = L_mm; columns = doses for the selected channel.
    """
    doses_order = list(dict.fromkeys([c[0] for c in A_large.columns]))
    for ch in ("R", "G", "B"):
        # A per channel: select level "Channel" = ch, columns become the dose index
        A_ch = A_large.xs(ch, axis=1, level="Channel")
        B_ch = B_large.xs(ch, axis=1, level="Channel")
        # Reorder columns by input dose order
        A_ch = A_ch.reindex(columns=doses_order)
        B_ch = B_ch.reindex(columns=doses_order)
        A_path = os.path.join(out_dir, f"{prefix}_AB_per_dose_A_{ch}.csv")
        B_path = os.path.join(out_dir, f"{prefix}_AB_per_dose_B_{ch}.csv")
        A_ch.to_csv(A_path, index=True)
        B_ch.to_csv(B_path, index=True)


# ---------- Goodness-of-fit ----------
def r2_unweighted(y, yhat) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    sst = np.sum((y - np.mean(y))**2)
    sse = np.sum((y - yhat)**2)
    return 1.0 - sse/sst if sst > 0 else 1.0


def r2_weighted(y, yhat, w) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float); w = np.asarray(w, float)
    w = np.clip(w, 0, None)
    if np.allclose(w.sum(), 0): w = np.ones_like(w)
    ybar = np.sum(w*y)/np.sum(w)
    sst = np.sum(w*(y - ybar)**2)
    sse = np.sum(w*(y - yhat)**2)
    return 1.0 - sse/sst if sst > 0 else 1.0


# ---------- Single-pair (A,B) per position & channel ----------
def fit_AB_ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    X = np.c_[np.ones_like(x), x]
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    A, B = float(beta[0]), float(beta[1])
    yhat = X @ beta
    return A, B, r2_unweighted(y, yhat)


def fit_AB_wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float, float]:
    X = np.c_[np.ones_like(x), x]
    W = np.diag(w.astype(float))
    beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
    A, B = float(beta[0]), float(beta[1])
    yhat = X @ beta
    return A, B, r2_weighted(y, yhat, w)


def compute_single_pairs_with_r2(df_raw: pd.DataFrame,
                                 B_per_dose: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build four tables:
      - OLS:      A,B,R² (unweighted)
      - WLS:      A,B,R² (weighted with w = (x-x0)^2)
      - mean B(D) + anchored A, R²
      - median B(D) + anchored A, R²
    """
    doses = list(dict.fromkeys([c[0] for c in df_raw.columns]))
    chans = ["R", "G", "B"]
    Ls = list(df_raw.index)

    def empty():
        return pd.DataFrame({
            "L_mm": Ls,
            "A_R": np.nan, "B_R": np.nan, "R2_R": np.nan,
            "A_G": np.nan, "B_G": np.nan, "R2_G": np.nan,
            "A_B": np.nan, "B_B": np.nan, "R2_B": np.nan,
        })

    tbls = {"OLS": empty(), "WLS": empty(), "weightedAvg": empty(), "median": empty()}

    for L in Ls:
        for ch in chans:
            if L == 0.0:
                for name in tbls:
                    tbls[name].loc[tbls[name]["L_mm"] == L, [f"A_{ch}", f"B_{ch}", f"R2_{ch}"]] = [0.0, 1.0, 1.0]
                continue

            x = np.array([df_raw.loc[L, (D, ch)] for D in doses], float)    # lateral signal
            y = np.array([df_raw.loc[0.0, (D, ch)] for D in doses], float)  # center signal

            # 1) OLS
            A, B, R2 = fit_AB_ols(x, y)
            tbls["OLS"].loc[tbls["OLS"]["L_mm"] == L, [f"A_{ch}", f"B_{ch}", f"R2_{ch}"]] = [A, B, R2]

            # 2) WLS (heavy weight to larger dose differences)
            w = (x - x[0])**2
            if np.allclose(w, 0): w = np.ones_like(w)
            A, B, R2w = fit_AB_wls(x, y, w)
            tbls["WLS"].loc[tbls["WLS"]["L_mm"] == L, [f"A_{ch}", f"B_{ch}", f"R2_{ch}"]] = [A, B, R2w]

            # 3) Mean B(D) (from per-dose table), A anchored at D=0
            Bpd = np.array([B_per_dose.loc[L, (D, ch)] for D in doses if D != doses[0]], float)
            B_mean = float(np.nanmean(Bpd))
            A_mean = float(df_raw.loc[0.0, (doses[0], ch)] - B_mean*df_raw.loc[L, (doses[0], ch)])
            yhat = A_mean + B_mean*x
            R2m = r2_unweighted(y, yhat)
            tbls["weightedAvg"].loc[tbls["weightedAvg"]["L_mm"] == L, [f"A_{ch}", f"B_{ch}", f"R2_{ch}"]] = [A_mean, B_mean, R2m]

            # 4) Median B(D) (robust), A anchored at D=0
            B_med = float(np.nanmedian(Bpd))
            A_med = float(df_raw.loc[0.0, (doses[0], ch)] - B_med*df_raw.loc[L, (doses[0], ch)])
            yhat = A_med + B_med*x
            R2med = r2_unweighted(y, yhat)
            tbls["median"].loc[tbls["median"]["L_mm"] == L, [f"A_{ch}", f"B_{ch}", f"R2_{ch}"]] = [A_med, B_med, R2med]

    return tbls


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Compute LRA coefficients with R² from RAW CSV/XLSX/XLSM.")
    p.add_argument("--input", required=True, help="Path to RAW table (.csv, .xlsx, .xlsm)")
    p.add_argument("--sheet", default=None, help="Sheet name for Excel (optional)")
    p.add_argument("--out_dir", default=".", help="Output directory")
    p.add_argument("--prefix", default="LRA_from_RAW", help="Output file prefix")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_raw = read_raw_table(args.input, args.sheet)

    # Per-dose A,B
    A_large, B_large = compute_per_dose_AB(df_raw)
    path_A = os.path.join(args.out_dir, f"{args.prefix}_AB_per_dose_A.csv")
    path_B = os.path.join(args.out_dir, f"{args.prefix}_AB_per_dose_B.csv")
    write_large_like_input(df_raw, A_large, path_A)
    write_large_like_input(df_raw, B_large, path_B)

    # NEW: split per channel (6 CSVs)
    write_channel_splits(A_large, B_large, args.out_dir, args.prefix)

    # Single pair + R²
    singles = compute_single_pairs_with_r2(df_raw, B_large)
    for name, tbl in singles.items():
        out = os.path.join(args.out_dir, f"{args.prefix}_single_AB_{name}_with_R2.csv")
        tbl.to_csv(out, index=False)

    print("Écrit :")
    print("  ", path_A)
    print("  ", path_B)
    for ch in ("R","G","B"):
        print("  ", os.path.join(args.out_dir, f"{args.prefix}_AB_per_dose_A_{ch}.csv"))
    for ch in ("R","G","B"):
        print("  ", os.path.join(args.out_dir, f"{args.prefix}_AB_per_dose_B_{ch}.csv"))
    for name in singles:
        print("  ", os.path.join(args.out_dir, f"{args.prefix}_single_AB_{name}_with_R2.csv"))


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot LRA coefficients A(L) and B(L) vs lateral position from a CSV.
Expected columns: L_mm, A_R, B_R, A_G, B_G, A_B, B_B
Optional: R2_R, R2_G, R2_B (ignored for plotting).

Usage:
  python plot_lra_AB.py --input path/to/single_AB.csv --out_dir path/to/output --title "WLS coefficients"
"""
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def load_single_ab(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    required = ["L_mm","A_R","B_R","A_G","B_G","A_B","B_B"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    # sort by L
    df = df.sort_values("L_mm").reset_index(drop=True)
    return df

def plot_coefficients(df: pd.DataFrame, out_dir: str, title_prefix: str = "") -> None:
    os.makedirs(out_dir, exist_ok=True)
    # --- Figure A(L) ---
    figA, axA = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    axA.plot(df["L_mm"], df["A_R"], marker="o", color="red",  label="Red")
    axA.plot(df["L_mm"], df["A_G"], marker="o", color="green",label="Green")
    axA.plot(df["L_mm"], df["A_B"], marker="o", color="blue", label="Blue")
    axA.set_xlabel("Lateral position, mm")
    axA.set_ylabel("Correction coefficient A")
    ttl = "Coefficient A vs. Lateral Position"
    if title_prefix:
        ttl = f"{title_prefix} – " + ttl
    axA.set_title(ttl)
    axA.grid(True, alpha=0.3)
    axA.legend()
    outA = os.path.join(out_dir, "A_vs_L.png")
    figA.tight_layout()
    figA.savefig(outA)
    plt.close(figA)

    # --- Figure B(L) ---
    figB, axB = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    axB.plot(df["L_mm"], df["B_R"], marker="o", color="red",  label="Red")
    axB.plot(df["L_mm"], df["B_G"], marker="o", color="green",label="Green")
    axB.plot(df["L_mm"], df["B_B"], marker="o", color="blue", label="Blue")
    axB.set_xlabel("Lateral position, mm")
    axB.set_ylabel("Correction coefficient B")
    ttl = "Coefficient B vs. Lateral Position"
    if title_prefix:
        ttl = f"{title_prefix} – " + ttl
    axB.set_title(ttl)
    axB.grid(True, alpha=0.3)
    axB.legend()
    outB = os.path.join(out_dir, "B_vs_L.png")
    figB.tight_layout()
    figB.savefig(outB)
    plt.close(figB)

    print("Saved:", outA)
    print("Saved:", outB)

def main():
    ap = argparse.ArgumentParser(description="Plot A(L) and B(L) from single_AB CSV.")
    ap.add_argument("--input", required=True, help="CSV with columns L_mm, A_R,B_R,A_G,B_G,A_B,B_B")
    ap.add_argument("--out_dir", required=True, help="Output directory for figures")
    ap.add_argument("--title", default="", help="Optional title prefix")
    args = ap.parse_args()

    df = load_single_ab(args.input)
    plot_coefficients(df, args.out_dir, args.title)

if __name__ == "__main__":
    main()

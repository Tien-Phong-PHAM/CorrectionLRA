#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LRA correction for 48-bit (16-bit per channel) RGB film scans.

Model (per channel X ∈ {R,G,B}, lateral position L in mm):
    R_center = A(L,X) + B(L,X) * R_lateral

Inputs
- RGB 48-bit TIFF (uint16 per channel)
- CSV coefficients with columns: L_mm,A_R,B_R,A_G,B_G,A_B,B_B (measured at discrete L, e.g., -129..+129 mm)

The script:
1) Reads the TIFF and coefficients
2) Interpolates A(L), B(L) along the lateral axis using DPI (pixels ↔ mm)
3) Applies the correction
4) Writes the corrected TIFF **preserving the original metadata** (resolution, unit, compression, ICC, etc.)

Usage (portrait, lateral axis = columns):
  python lra_correction.py \
    --input IN.tif --output OUT_lra.tif \
    --coeffs LRA_coeffs.csv --dpi 75 --axis columns

If your file is rotated (lateral axis along rows):
  --axis rows

If the image is a crop (lateral origin shift):
  --x_offset_px <pixels>

Clamp range (optional):
  --clip_min 0 --clip_max 65535
"""

import argparse
import csv
import sys
import numpy as np

try:
    import tifffile as tiff
except Exception:
    tiff = None


# ---------- utilities ----------

def _require_tifffile():
    if tiff is None:
        raise ImportError("This script requires 'tifffile'. Install with: pip install tifffile")


def _to_float(v):
    """Convert TIFF rational/tuple to float if possible."""
    try:
        # tifffile may return fractions.Fraction or (num,den)
        if isinstance(v, tuple) and len(v) == 2:
            num, den = v
            return float(num) / float(den) if den else float(num)
        return float(v)
    except Exception:
        return None


# ---------- IO: image & metadata ----------

def read_rgb16(path):
    _require_tifffile()
    arr = tiff.imread(path)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {arr.shape}")
    # Coerce to uint16 baseline if needed
    if arr.dtype == np.uint8:
        arr = (arr.astype(np.uint16) << 8)
    elif arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return arr


def read_src_metadata(path):
    """Read a subset of TIFF tags we want to preserve."""
    _require_tifffile()
    meta = {}
    with tiff.TiffFile(path) as tf:
        p = tf.pages[0]
        meta["compression"]   = p.compression        # enum or string acceptable by tifffile.imwrite
        meta["planarconfig"]  = p.planarconfig
        meta["photometric"]   = "rgb"                # we write RGB contiguous
        # Resolution & unit
        xr = p.tags.get("XResolution")
        yr = p.tags.get("YResolution")
        ru = p.tags.get("ResolutionUnit")
        meta["XResolution"]   = _to_float(xr.value) if xr else None
        meta["YResolution"]   = _to_float(yr.value) if yr else None
        meta["ResolutionUnit"] = ru.value if ru else None  # 2=INCH, 3=CM
        # Optional extras
        meta["ICCProfile"]    = (p.tags.get("ICCProfile").value
                                 if p.tags.get("ICCProfile") is not None else None)
        desc = p.tags.get("ImageDescription")
        meta["ImageDescription"] = desc.value if desc else None
        # Some writers add JSON metadata; set None to avoid duplication
        meta["Software"]      = None
    return meta


def save_rgb16_like(src_path, dst_path, arr_uint16,
                    force_resolution=None, force_resolution_unit=None):
    """
    Save arr_uint16 as TIFF preserving important tags from src_path.
    Optionally override resolution and unit with force_resolution=(xres,yres), force_resolution_unit='INCH'/'CM'/2/3.
    """
    _require_tifffile()
    meta = read_src_metadata(src_path)

    # Override resolution if requested or if missing on source
    resolution = None
    resolutionunit = meta["ResolutionUnit"]
    xres = meta["XResolution"]
    yres = meta["YResolution"]

    if force_resolution is not None:
        resolution = tuple(force_resolution)
        resolutionunit = force_resolution_unit if force_resolution_unit is not None else resolutionunit
    elif xres and yres:
        resolution = (xres, yres)  # keep original
    # else: leave resolution=None (no DPI written)

    # Prepare kwargs for tifffile
    kwargs = dict(
        photometric=meta["photometric"],
        compression=meta["compression"],
        planarconfig=meta["planarconfig"],
        metadata=None,  # avoid extra JSON block
    )
    if resolution is not None:
        kwargs["resolution"] = resolution
    if resolutionunit is not None:
        kwargs["resolutionunit"] = resolutionunit
    if meta["ICCProfile"]:
        kwargs["iccprofile"] = meta["ICCProfile"]
    if meta["ImageDescription"]:
        kwargs["description"] = meta["ImageDescription"]

    tiff.imwrite(dst_path,
                 np.clip(arr_uint16, 0, 65535).astype(np.uint16),
                 **kwargs)


# ---------- coefficients & interpolation ----------

def load_coeffs(csv_path):
    """Read CSV with columns: L_mm,A_R,B_R,A_G,B_G,A_B,B_B ; return sorted arrays."""
    L, A_R, B_R, A_G, B_G, A_B, B_B = [], [], [], [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = ["L_mm", "A_R", "B_R", "A_G", "B_G", "A_B", "B_B"]
        miss = [k for k in required if k not in reader.fieldnames]
        if miss:
            raise ValueError(f"CSV is missing columns: {miss}")
        for row in reader:
            L.append(float(row["L_mm"]))
            A_R.append(float(row["A_R"])); B_R.append(float(row["B_R"]))
            A_G.append(float(row["A_G"])); B_G.append(float(row["B_G"]))
            A_B.append(float(row["A_B"])); B_B.append(float(row["B_B"]))
    L = np.asarray(L, dtype=float)
    order = np.argsort(L)
    L = L[order]
    A = {
        "R": np.asarray(A_R, dtype=float)[order],
        "G": np.asarray(A_G, dtype=float)[order],
        "B": np.asarray(A_B, dtype=float)[order],
    }
    B = {
        "R": np.asarray(B_R, dtype=float)[order],
        "G": np.asarray(B_G, dtype=float)[order],
        "B": np.asarray(B_B, dtype=float)[order],
    }
    return L, A, B


def build_AB_per_axis(length_px, dpi, L_meas_mm, A_meas, B_meas, x_offset_px=0.0):
    """
    Build per-index A and B arrays (length = length_px) by linear interpolation over L.
    Coordinate for index i (0-based):
        L_i_mm = ((i + x_offset_px) - (length_px - 1)/2) * (25.4 / dpi)
    x_offset_px shifts the lateral origin if the image is cropped.
    """
    mm_per_px = 25.4 / float(dpi)
    idx = np.arange(length_px, dtype=np.float64)
    Lx = ((idx + float(x_offset_px)) - (length_px - 1) / 2.0) * mm_per_px

    A_axis = {}
    B_axis = {}
    for ch in ("R", "G", "B"):
        A_interp = np.interp(Lx, L_meas_mm, A_meas[ch],
                             left=A_meas[ch][0], right=A_meas[ch][-1])
        B_interp = np.interp(Lx, L_meas_mm, B_meas[ch],
                             left=B_meas[ch][0], right=B_meas[ch][-1])
        A_axis[ch] = A_interp.astype(np.float64)
        B_axis[ch] = B_interp.astype(np.float64)
    return A_axis, B_axis


# ---------- correction ----------

def apply_lra(arr_rgb16, A_axis, B_axis, axis="columns", clip_min=0.0, clip_max=65535.0):
    """
    Apply LRA. If axis='columns', A_axis/B_axis length must be width; if 'rows', length must be height.
    """
    arr = arr_rgb16.astype(np.float64)
    H, W, _ = arr.shape
    out = np.empty_like(arr, dtype=np.float64)

    if axis == "columns":
        # broadcast along rows
        for ci, ch in enumerate(("R", "G", "B")):
            out[:, :, ci] = A_axis[ch][None, :] + B_axis[ch][None, :] * arr[:, :, ci]
    elif axis == "rows":
        # broadcast along columns
        for ci, ch in enumerate(("R", "G", "B")):
            out[:, :, ci] = A_axis[ch][:, None] + B_axis[ch][:, None] * arr[:, :, ci]
    else:
        raise ValueError("axis must be 'columns' or 'rows'")

    return np.clip(out, clip_min, clip_max).astype(np.uint16)


# ---------- CLI ----------

def main(argv=None):
    p = argparse.ArgumentParser(description="Apply LRA correction to 48-bit RGB TIFF scans (preserve metadata).")
    p.add_argument("--input", required=True, help="Input 48-bit RGB TIFF path")
    p.add_argument("--output", required=True, help="Output 48-bit RGB TIFF path")
    p.add_argument("--coeffs", required=True, help="CSV with columns: L_mm,A_R,B_R,A_G,B_G,A_B,B_B")
    p.add_argument("--dpi", type=float, default=75.0, help="Scanner DPI used for pixel↔mm mapping (default: 75)")
    p.add_argument("--axis", choices=["columns", "rows"], default="columns",
                   help="Which image axis is lateral (perpendicular to scan direction) (default: columns)")
    p.add_argument("--x_offset_px", type=float, default=0.0,
                   help="Lateral origin shift in pixels if the image is a crop (default: 0)")
    p.add_argument("--clip_min", type=float, default=0.0, help="Output clamp min (default 0)")
    p.add_argument("--clip_max", type=float, default=65535.0, help="Output clamp max (default 65535)")
    p.add_argument("--force_resolution", type=float, nargs=2, metavar=("XRES", "YRES"),
                   help="Override resolution (DPI or DPCM depending on unit). If omitted, copy from source.")
    p.add_argument("--force_resolution_unit", type=str,
                   help="Override resolution unit: 'INCH' or 'CM' (or 2/3). If omitted, copy from source.")
    args = p.parse_args(argv)

    # 1) read image & coeffs
    img = read_rgb16(args.input)
    H, W, _ = img.shape
    L_meas, A_meas, B_meas = load_coeffs(args.coeffs)

    # 2) interpolate A,B along the lateral axis
    if args.axis == "columns":
        A_axis, B_axis = build_AB_per_axis(W, args.dpi, L_meas, A_meas, B_meas, x_offset_px=args.x_offset_px)
    else:  # rows are lateral
        A_axis, B_axis = build_AB_per_axis(H, args.dpi, L_meas, A_meas, B_meas, x_offset_px=args.x_offset_px)

    # 3) apply correction
    out = apply_lra(img, A_axis, B_axis, axis=args.axis,
                    clip_min=args.clip_min, clip_max=args.clip_max)

    # 4) save copying metadata (optionally override resolution)
    save_rgb16_like(args.input, args.output, out,
                    force_resolution=tuple(args.force_resolution) if args.force_resolution else None,
                    force_resolution_unit=args.force_resolution_unit)

    print(f"[LRA] input:  {args.input}")
    print(f"[LRA] output: {args.output}")
    print(f"[LRA] shape:  {img.shape} | axis={args.axis} | dpi={args.dpi} | x_offset_px={args.x_offset_px}")
    print("[LRA] done.")


if __name__ == "__main__":
    sys.exit(main())


# CorrectionLRA (GUI)

A tiny cross‑platform GUI to apply **LRA correction** to 48‑bit RGB TIFF scans using your CSV of coefficients.

- **Input**: a single TIFF file _or_ a folder containing TIFF images.
- **CSV**: coefficients with columns `L_mm,A_R,B_R,A_G,B_G,A_B,B_B`.
- **Output**: a new subfolder named **`Correction_RLA`** is created **next to** your input.
  Each output file keeps the original basename + suffix **`_CorrectionLRA.tif`**.

This app wraps the provided `lra_correction.py` (unmodified) and preserves TIFF metadata via `tifffile`.

---

## Quick start

### 1) Install Python 3.9+
- Windows: https://www.python.org/downloads/windows/
- macOS: https://www.python.org/downloads/macos/
  - On macOS, prefer the official Python.org installer (bundles a recent Tk).

### 2) Create a virtual environment and install deps

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 3) Run the app

```bash
python app_gui.py
```

---

## Packaging into a clickable app (optional)

We suggest **PyInstaller**. It creates a single EXE on Windows and a `.app` bundle on macOS.

Install it inside your venv:
```bash
pip install pyinstaller
```

### Windows (EXE)
```bash
pyinstaller --noconfirm --windowed --name CorrectionLRA app_gui.py
```
- Your EXE will be in the `dist/` folder as `CorrectionLRA.exe`.
- If TIFFs are large, first run from the terminal to see logs: `dist\CorrectionLRA.exe`

### macOS (.app bundle)
```bash
pyinstaller --noconfirm --windowed --name CorrectionLRA app_gui.py
```
- You’ll get `dist/CorrectionLRA.app`. If macOS blocks it, you may need to right‑click → Open the first time.
- For distribution outside your machine, Apple signing/notarization may be required (beyond this quick guide).

> Note: PyInstaller detects `tifffile` and bundles it. The GUI uses built‑in `tkinter` (ships with Python.org installer).

---

## Notes & tips

- **DPI**: used to map pixels to millimetres when interpolating A(L) and B(L). Default is 75 dpi.
- **Axis**: `"columns"` means the lateral axis is the image width (usual for portrait scans). `"rows"` if lateral is height.
- **x_offset_px**: enter a pixel offset if your image is a crop and the lateral origin is shifted.
- **Clipping**: default `[0, 65535]` for 16‑bit output.

If you need anything tweaked (batch UI, progress bar, CSV preview), it’s easy to extend.

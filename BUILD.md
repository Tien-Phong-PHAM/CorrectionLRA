# Build Cheatsheet

Inside your virtual environment:

```bash
pip install -r requirements.txt
pip install pyinstaller
```

## Windows
```bash
pyinstaller --noconfirm --windowed --name CorrectionLRA app_gui.py
```

## macOS (Intel/Apple Silicon)
```bash
pyinstaller --noconfirm --windowed --name CorrectionLRA app_gui.py
```
The resulting app lives in `dist/`.

### Troubleshooting
- If `tifffile` complains about missing `numpy`, ensure both are installed in the same environment.
- On macOS, if Tk reports issues, prefer Python.org installers rather than Homebrew for GUI apps.
- Large TIFFs can take time. If the app seems unresponsive, run from Terminal to see detailed logs:
  ```bash
  python app_gui.py
  ```

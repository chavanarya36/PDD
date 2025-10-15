# PDD (Plant Disease Detection)

This Streamlit app uses a Keras model to identify plant diseases from leaf images. It features a modern UI, caching for fast inference, and built‑in translation support.

## Quick Start (from ZIP)

Follow these steps when you share the project as a ZIP and want to run it on another device.

Prerequisites:
- Python 3.10 or 3.11
- Internet access (first run installs dependencies; translation calls use online API)
- Model file `trained_model.keras` present in the project root (included in ZIP)

### Windows (PowerShell)
1) Extract the ZIP to a simple path, e.g. `C:\Plant-Disease-Detection`.
2) Open Windows PowerShell in that folder.
3) Create and activate a virtual environment:
	```powershell
	py -3.10 -m venv .venv ; .\.venv\Scripts\Activate.ps1
	```
	- If activation is blocked, temporarily allow scripts for this session:
	  ```powershell
	  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
	  ```
4) Upgrade pip and install dependencies:
	```powershell
	python -m pip install --upgrade pip ; pip install -r requirements.txt
	```
5) Run the app (change port if needed):
	```powershell
	streamlit run mai.py --server.port=8501
	```
6) Open the URL shown in the terminal (usually http://localhost:8501).

### macOS
1) Extract the ZIP and open Terminal in that folder.
2) Create and activate a virtual environment:
	```bash
	python3 -m venv .venv && source .venv/bin/activate
	```
3) Install dependencies and run:
	```bash
	python -m pip install --upgrade pip && pip install -r requirements.txt
	streamlit run mai.py --server.port=8501
	```

### Linux
1) Extract the ZIP and open a shell in that folder.
2) Create and activate a virtual environment:
	```bash
	python3 -m venv .venv && source .venv/bin/activate
	```
3) Install dependencies and run:
	```bash
	python -m pip install --upgrade pip && pip install -r requirements.txt
	streamlit run mai.py --server.port=8501
	```

## Features
- Elegant, modern UI (dark theme, glassmorphism, subtle animations)
- Fast startup and inference using cached translator and model
- Full‑app translation support (English ↔ Hindi) with automatic fallback
- “Preload Model” control in the sidebar for immediate, smooth first prediction

## Troubleshooting
- Model failed to load: Ensure `trained_model.keras` exists in the project root and matches your TensorFlow/Keras versions in `requirements.txt`.
- Port is busy: Run with a different port, e.g. `--server.port=8502`.
- Virtualenv activation blocked (Windows): Use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before running `Activate.ps1`.
- Missing dependency (e.g., `deep_translator`): Run `pip install -r requirements.txt` again after activating the virtualenv.
- Slow first prediction: Click “Preload Model” in the sidebar to load the model upfront.

## Notes
- The app defaults to using the `trained_model.keras` artifact. A legacy `trained_model.h5` may be present but is not used by default.
- For GPU usage, additional drivers/toolkits are required; this setup targets CPU by default.

---

If you want an optional script to validate the model and run a sample prediction automatically, let me know and I’ll add it.

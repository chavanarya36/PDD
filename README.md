# PDD (Plant Disease Detection)

This repository contains a Streamlit app that uses a saved Keras model to identify plant diseases from images.

Quick run (Windows + venv):

```powershell
# activate your virtualenv if needed
& 'C:/Users/chava/Plant-Disease-Detection/venv/Scripts/Activate.ps1'
# install deps (only if not installed)
pip install -r requirements.txt
# run streamlit on the default port (or specify the port with --server.port)
& 'C:/Users/chava/Plant-Disease-Detection/venv/Scripts/streamlit.exe' run mai.py --server.port=8504
```

If the app shows "Model failed to load", check that `trained_model.keras` exists in the project root and is a valid Keras saved model.

If you want me to add a small script to validate the model and run sample predictions automatically, say so and I'll add it.

# PDD (Plant Disease Detection)

## Deploy to Streamlit Community Cloud (fast)

1. Push your latest code to GitHub (public repo recommended for free Streamlit Cloud):

```powershell
git add -A
git commit -m "Prepare for Streamlit Cloud deploy"
git push origin main
```

2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app" → select your repo `chavanarya36/PDD`, branch `main`, and set the file path to `mai.py`.
4. Click "Deploy app" and watch the build logs. When it finishes you'll get a public URL.

Notes:
- If the build fails due to large model file or timeouts, consider hosting `trained_model.keras` externally (S3 / GitHub Release / Google Drive) and downloading it at startup (I can add code for that).
- Check the app logs in the Streamlit dashboard for errors (missing requirements, file paths).

## Quick Procfile (optional)

If you want to deploy on Heroku or Render, add a `Procfile` with this single line (I added one in the repo):

```
web: streamlit run mai.py --server.port $PORT
```

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

# Plant Disease Detection — Interview Prep

Use this as a focused drill guide. It gives you crisp pitches, likely questions with strong answers, realistic debugging stories, and practical next steps. All content is tailored to this repo.

## 30 / 60 / 120‑second pitches

- 30s: I built a Streamlit app that detects plant diseases from leaf photos using a Keras model saved as `trained_model.keras`. The app resizes images to 128×128, feeds raw 0–255 RGB tensors, and returns top‑3 classes plus treatment suggestions. It supports English/Hindi via GoogleTranslator. Validation logs show ~96% val accuracy; in the app I added a confidence threshold and top‑3 to improve UX.
- 60s: The UI is Streamlit (`mai.py`) with lazy, single-load model initialization and a robust relative path for the 94MB model file. Preprocessing is minimal by design—`PIL.Image` → RGB → 128×128 → float32 in 0–255—matching the model’s training. I validated this with a `scripts/preprocess_experiments.py` harness across several test images, comparing 0–255 vs normalized inputs; 0–255 is correct. The app shows top‑3 results and maps the predicted class to a human treatment recommendation with optional translation. For deployment I prepared Streamlit Cloud/Heroku scaffolding but paused per project constraints; the README documents local Windows venv steps.
- 120s: Data flows from user upload to PIL for RGB conversion and resizing to 128×128, then a single Keras inference on a preloaded model (TensorFlow 2.17, Keras 3.5). Predictions are converted to top‑k with confidences; a confidence slider helps decide when to warn users. The class list aligns to training label order (38 classes including Apple/Corn/Potato/Tomato variants). I added stateful navigation, language selection, and treatment guidance. On the engineering side: the model loads once via an absolute path computed relative to `mai.py` to avoid working‑dir pitfalls. Diagnostic scripts inspect the model, evaluate labeled samples, and compare preprocessing regimes. Training history (`training_hist.json`) shows consistent convergence to ~96% validation accuracy. For production hardening, I’d add server logging/monitoring, batch inference, model checksum verification, and optionally a tiny REST API around the model to decouple UI from inference.

## Architecture and data flow

- Frontend: Streamlit app (`mai.py`)
  - Sidebar: language toggle (English/Hindi), page navigation, confidence threshold
  - Pages: Home, About, Disease Recognition (upload → predict)
- Inference: Keras model (`trained_model.keras`, 128×128×3 input)
  - Preprocessing: PIL → RGB, resized to 128×128, converted to float32, no normalization (0–255)
  - Postprocessing: softmax probs → top‑3 classes + confidences; thresholding and treatment mapping
- i18n: `deep_translator.GoogleTranslator` to translate UI strings and treatment text
- Assets: local images under `test/` for sanity checks; model file co‑located with `mai.py`

## Key technical facts (cheat sheet)

- Input shape: 128×128×3 RGB, float32 in 0–255 range
- Output: 38‑way softmax. Top‑3 shown to user
- Libraries: TensorFlow 2.17, Keras 3.5, Streamlit 1.37, Pillow 10.4, deep_translator
- Model file: `trained_model.keras` (absolute path resolved from `mai.py`); backup `trained_model.h5`
- UI extras: Confidence slider, English/Hindi toggle, simple treatment advice per class
- Training signals: `training_hist.json` shows val_accuracy peaking ~0.964, val_loss ~0.12
- Run locally (PowerShell): see README — activate venv, `streamlit run mai.py --server.port 8504`

## Preprocessing and inference details

- Why 0–255? Diagnostics showed the model was trained on raw pixel scale. Normalizing to 0–1 or −1–1 reduced accuracy. Therefore inference uses 0–255 to match training distribution.
- Determinism: `Image.open(...).convert('RGB').resize((128,128))` → `np.array(..., dtype=float32)` → `np.expand_dims` → `model.predict`.
- Top‑k: `np.argsort(probs)[-3:][::-1]`. Confidence is `prob*100`. Slider gates low‑confidence warnings.

## Common interview questions and strong answers

1) What problem does your project solve and what is the target user?
- It helps farmers and gardeners quickly identify crop diseases from leaf photos. Target users are non‑technical end‑users; the UX emphasizes clarity, top‑3 options, and simple treatment advice with translation.

2) How did you choose preprocessing and input scaling?
- I aligned inference to the training distribution. I verified three regimes (0–255, 0–1, −1–1) using `scripts/preprocess_experiments.py` and confirmed 0–255 provided the best top‑1/top‑3 on held‑out test images.

3) How do you ensure the class order matches the model’s training labels?
- I keep a single source of truth for `class_name` aligned with the training dataset index mapping. In the app, the class list mirrors training order, and diagnostics cross‑check indices by running sample predictions and confirming labels.

4) What are potential sources of misclassification?
- Domain shift (lighting, background clutter, camera quality), EXIF rotation issues, motion blur, and disease similarity (e.g., early vs late blight). Also user‑provided images may contain multiple leaves or non‑leaf content.

5) How would you improve accuracy and robustness?
- Add augmentations aligned with field conditions (varying brightness, background, perspective), fine‑tune a modern backbone like EfficientNetV2/ConvNeXt, perform temperature scaling or Platt scaling for calibrated probabilities, and use test‑time augmentation (TTA) selectively. I’d quantify gains using top‑k and AUROC per class.

6) What deployment options did you consider and what constraints exist?
- Streamlit Community Cloud (quick), Heroku/Render, and Dockerized deployments. The 94MB model needs careful handling on free tiers (cold start, storage limits). I use absolute model paths and can add a runtime download with checksum verification if needed.

7) How do you handle performance and scalability?
- Model loads once; subsequent inferences are fast. For scale, a separate inference service (FastAPI) with batching and a GPU‑enabled container would decouple UI from compute. Streamlit can call the API; horizontal scaling then becomes straightforward.

8) How do you test and validate the app?
- Unit: sanity predictions on known images; ensure top‑k indices map to expected classes. Integration: end‑to‑end upload → predict flow. Diagnostics: `preprocess_experiments.py` to compare preprocessing modes. Manual smoke tests on `test/` images.

9) Any security or safety considerations?
- Validate image files (size/type), strip EXIF, limit upload size, and sandbox image processing to avoid resource abuse. For model safety: avoid arbitrary code execution via model deserialization (stick to standard Keras), and keep dependencies patched.

10) How would you explain misprediction cases to a user?
- Show top‑3 with confidences and a low‑confidence warning. Offer guidance: retake photo with better lighting/zoom and consider multiple images. Optionally add Grad‑CAM visualizations to explain focus areas.

11) How is translation handled and what are its limits?
- Via `deep_translator.GoogleTranslator`. It’s best‑effort; for production I’d cache translations and provide curated, domain‑specific phrasing for treatment advice.

12) How would you package this for a coding interview demo?
- A minimal Dockerfile that installs TensorFlow‑CPU, adds the model, and exposes Streamlit on 8501. Provide a one‑liner to run locally, and include a script that runs 2–3 sample predictions in CI for sanity.

## Debugging stories (STAR format)

1) Input scaling mismatch
- Situation: Early tests gave inconsistent predictions across similar images.
- Task: Verify correct preprocessing to match training.
- Action: Built `preprocess_experiments.py` to test 0–255, 0–1, and −1–1 across `test/` images. Compared top‑1/top‑3.
- Result: 0–255 performed best, confirming training distribution; updated app to use raw 0–255 and accuracy stabilized.

2) Model load failures due to working directory
- Situation: Streamlit sometimes launched with a different CWD so the model wasn’t found.
- Task: Make loading robust regardless of where `streamlit run` is invoked.
- Action: Resolved an absolute path relative to `mai.py` (`os.path.dirname(os.path.abspath(__file__))`).
- Result: Eliminated “model not found” issues; model loads exactly once for the session.

3) Prediction discrepancy between web vs local image
- Situation: A web image of Potato Early Blight classified correctly; a local test image didn’t.
- Task: Determine if it’s pipeline or data.
- Action: Confirmed preprocessing is identical, inspected filename‑derived labels, and checked EXIF/rotation and crop quality.
- Result: Concluded it’s likely a domain/quality issue; added a confidence threshold, top‑3 display, and guidance to retake clearer photos.

## Measurable improvements and how I’d validate

- Model backbone: Fine‑tune EfficientNetV2‑S on the dataset → expect +2–4% top‑1; validate via cross‑val and per‑class metrics.
- Data pipeline: Strong augmentations (color jitter, random background, perspective) → improved robustness; measure with stress tests and corruptions benchmarks.
- Calibration: Temperature scaling on val set → better confidence alignment; measure ECE/MCE and reliability plots.
- Inference UX: Add Grad‑CAM overlays → qualitative trust; A/B test with users.
- Performance: Convert to TFLite FP16/INT8 for edge or serverless; benchmark latency and memory.
- MLOps: Separate FastAPI inference service with Prometheus metrics; track latency, throughput, error rates.

## How to talk through the code (mai.py)

- `ensure_model_loaded()`: single‑flight model loader with error capture and absolute path resolution.
- `model_prediction()`: PIL → RGB → 128×128 → float32 0–255 → `model.predict` → probs array; handles (1,N) vs (N,) shapes.
- UI: Sidebar controls (language, pages, threshold), main page handles upload and shows top‑3 + treatment; warns on low confidence.
- `suggest_treatment()`: deterministic mapping from class → advice; optional translation.

## Quick commands (Windows PowerShell)

```powershell
# optional: create venv
python -m venv venv
. .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run mai.py --server.port 8504
```

## Full class list (38)

Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy,
Blueberry___healthy, Cherry_(including_sour)___Powdery_mildew, Cherry_(including_sour)___healthy,
Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot, Corn_(maize)___Common_rust_,
Corn_(maize)___Northern_Leaf_Blight, Corn_(maize)___healthy, Grape___Black_rot,
Grape___Esca_(Black_Measles), Grape___Leaf_blight_(Isariopsis_Leaf_Spot), Grape___healthy,
Orange___Haunglongbing_(Citrus_greening), Peach___Bacterial_spot, Peach___healthy,
Pepper,_bell___Bacterial_spot, Pepper,_bell___healthy, Potato___Early_blight,
Potato___Late_blight, Potato___healthy, Raspberry___healthy, Soybean___healthy,
Squash___Powdery_mildew, Strawberry___Leaf_scorch, Strawberry___healthy,
Tomato___Bacterial_spot, Tomato___Early_blight, Tomato___Late_blight, Tomato___Leaf_Mold,
Tomato___Septoria_leaf_spot, Tomato___Spider_mites Two-spotted_spider_mite, Tomato___Target_Spot,
Tomato___Tomato_Yellow_Leaf_Curl_Virus, Tomato___Tomato_mosaic_virus, Tomato___healthy

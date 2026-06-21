# Early-Stage Diabetic Retinopathy Detection

**Domain: Agriculture (Problem track) · Problem P2**

A computer vision prototype that takes a retinal fundus image and outputs:
1. Whether diabetic retinopathy (DR) is present
2. If present, the severity grade (0–4, APTOS scale)
3. A confidence score, with automatic flagging for manual review when the model is uncertain

Built to run **without a GPU on demo day**: training happens once on Google Colab's
free GPU, the trained model is downloaded, and the actual demo app runs entirely
on a laptop CPU via Streamlit.

---

## 1. Project structure

```
dr-detection/
├── README.md                      <- you are here
├── .gitignore
│
├── notebook/
│   └── train_dr_model.ipynb       <- run this FIRST, in Google Colab (needs GPU)
│
└── app/
    ├── app.py                     <- Streamlit demo app (main deliverable)
    ├── api.py                     <- optional FastAPI endpoint (same model, REST)
    ├── inference.py                <- model loading + prediction + confidence logic
    ├── preprocessing.py           <- Ben Graham preprocessing (shared with notebook)
    ├── requirements.txt
    ├── model/
    │   ├── README.md
    │   └── dr_model_traced.pt     <- YOU put this here after training (not in repo)
    └── sample_images/
        ├── README.md
        └── (optional demo images you add)
```

---

## 2. How it works (architecture)

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│   Google Colab (GPU)     │         │   Your laptop (CPU)           │
│                          │         │                                │
│  train_dr_model.ipynb    │  model  │  app.py (Streamlit)           │
│  - APTOS dataset         │  file   │  api.py (FastAPI, optional)   │
│  - Ben Graham preprocess │ ──────► │  - same Ben Graham preprocess │
│  - EfficientNet-B0       │ (.pt)   │  - loads TorchScript model    │
│    + ordinal regression  │  via    │  - outputs grade + confidence │
│  - WeightedRandomSampler │ Google  │  - flags low-confidence cases │
│  - saves to Drive        │ Drive   │    for manual review          │
└─────────────────────────┘         └──────────────────────────────┘
```

### Key design decisions (and why)

| Decision | What we did | Why |
|---|---|---|
| **Output type** | Ordinal regression (1 scalar), not 5-way softmax classification | DR grades are ordered. Classification treats "predicted 0, true 4" the same as "predicted 3, true 4" — both just "wrong." Regression naturally penalizes far-off predictions more, which matches clinical reality. |
| **Preprocessing** | Ben Graham preprocessing (crop, subtract local average color, clip to circle) | APTOS images come from many different cameras with wildly inconsistent lighting/color. Without this, the model burns capacity learning to normalize cameras instead of lesions. |
| **Class imbalance** | `WeightedRandomSampler` only | APTOS has far more Grade 0 than Grade 3–4 images. Using a weighted sampler AND a class-weighted loss at the same time double-corrects the same problem and can hurt majority-class recall. We use exactly one mechanism. |
| **Confidence score** | Derived from distance-to-nearest-integer-grade, not a generic softmax max | A model that outputs raw value 2.02 and one that outputs 2.49 both round to "Grade 2" — but the second is sitting right on a clinical decision boundary and should be flagged, not reported with false confidence. |
| **Backbone** | EfficientNet-B0, transfer learning from ImageNet | Good accuracy/speed tradeoff; the notebook also supports swapping to MobileNetV2 if you need even faster inference. |
| **Demo environment** | Colab (train) → Streamlit on laptop CPU (demo) | EfficientNet-B0 CPU inference is ~500-900ms/image — too slow to retrain or do heavy live training on stage, but plenty fast for single-image inference in a demo. |

---

## 3. Prerequisites

- A Google account (for Colab + Google Drive)
- A Kaggle account (to download the APTOS 2019 dataset) — free, sign up at kaggle.com
- Python 3.9+ installed locally (for running the Streamlit app)
- ~2 GB free space in Google Drive (for the dataset + saved model)

---

## 4. Step-by-step: Part A — Train the model (Google Colab)

1. Go to [Google Colab](https://colab.research.google.com) and upload
   `notebook/train_dr_model.ipynb` (File → Upload notebook), or open it directly
   from Google Drive if you've already placed it there.

2. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → **T4 GPU** → Save.

3. **Get a Kaggle API token** (only needed once):
   - Go to [kaggle.com](https://www.kaggle.com) → your profile picture → Settings
   - Scroll to "API" → click "Create New Token" — this downloads `kaggle.json`
   - Keep this file handy; you'll upload it when the notebook asks (Section 2, Option A cell)

4. **Run the cells in order, top to bottom**:
   - **Section 1**: Mounts your Google Drive (click "Connect to Google Drive" and authorize when prompted). Creates a `biothon2026-dr-detection` folder in your Drive.
   - **Section 2**: Downloads **only** `train.csv` + `train_images.zip` via the Kaggle API (skips the much larger `test_images.zip`, which isn't needed for training/validation). Upload your `kaggle.json` when prompted. Each image is then resized to a max dimension of 512px on extraction, which keeps the final dataset to roughly **2–4GB** on disk regardless of the original camera resolutions — a cell at the end of this section prints the actual size so you can confirm. If it comes in over 4GB, lower `MAX_DIM` (e.g. to 384) in that cell and re-run.
   - **Section 3**: Defines and visualizes the Ben Graham preprocessing — you'll see a before/after grid across all 5 grades. Sanity-check that the processed images look reasonable (clear circular crop, no weird artifacts).
   - **Section 4**: Builds the PyTorch datasets/dataloaders with the `WeightedRandomSampler`.
   - **Section 5**: Defines the EfficientNet-B0 + ordinal regression model.
   - **Section 6**: Trains for up to 20 epochs with early stopping on validation QWK (Quadratic Weighted Kappa — the standard ordinal metric for this task). Takes roughly 30-60 minutes on a T4 GPU depending on Colab load. Saves the best checkpoint to Drive automatically after every improving epoch — you can close the tab and come back, the best-so-far model is already saved.
   - **Section 7**: Prints a classification report and confusion matrix on the held-out validation set.
   - **Section 8**: Exports a TorchScript-traced version of the model (`dr_model_traced.pt`) to Google Drive — this is the file the local app needs.

5. **Download the model file** from Google Drive:
   - In Google Drive, navigate to `biothon2026-dr-detection/models/`
   - Right-click `dr_model_traced.pt` → Download

---

## 5. Step-by-step: Part B — Run the demo app locally

1. **Clone/copy the project folder** to your laptop (the whole `dr-detection/` folder).

2. **Place the trained model**:
   - Move the `dr_model_traced.pt` you downloaded from Drive into:
     ```
     dr-detection/app/model/dr_model_traced.pt
     ```

3. **(Optional) Add demo sample images**:
   - Grab 2-5 sample fundus images (e.g. one per grade) and drop them into
     `dr-detection/app/sample_images/`
   - This lets you check the "Use a bundled sample image" box during a live
     demo instead of needing to upload a file on stage.

4. **Create a virtual environment and install dependencies**:

   ```bash
   cd dr-detection/app
   python3 -m venv venv

   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate

   pip install -r requirements.txt
   ```

5. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

   This opens automatically in your browser at `http://localhost:8501`.

6. **Use the app**:
   - Upload a retinal fundus image (PNG/JPG), or check "Use a bundled sample image"
   - Click **Run prediction**
   - You'll see: the grade (0–4) with label, confidence %, DR present yes/no,
     a clinical recommendation, per-grade probability bars, and — if confidence
     is below 60% — a clear warning flagging the case for manual review

---

## 6. (Optional) Running the FastAPI endpoint instead of / alongside Streamlit

The Streamlit app is the primary demo deliverable, but a REST API is included
too (useful if you want to show an integration point, e.g. for a future React
frontend, or to satisfy a "working API endpoint" requirement separately from
the UI demo).

```bash
cd dr-detection/app
source venv/bin/activate   # if not already active
uvicorn api:app --reload --port 8000
```

Then in another terminal:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/some_fundus_image.png"
```

Or visit `http://localhost:8000/docs` for an interactive Swagger UI to test
the endpoint by uploading a file directly in the browser.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Model not found at .../dr_model_traced.pt` | You haven't completed Part A (training) and Part B step 2 (placing the file). Re-check the exact path: `app/model/dr_model_traced.pt`. |
| Colab: "No GPU detected" warning in Section 1 | Runtime → Change runtime type → set Hardware accelerator to T4 GPU → Save → re-run from the top. |
| Dataset folder is bigger than 4GB | In Section 2's resize cell, lower `MAX_DIM` from `512` to `384` (or `256`), then re-run the download + extract cells. Smaller images still work fine for EfficientNet-B0 at 224×224 input. |
| Kaggle download fails / 403 error | Make sure you've accepted the competition rules at the [APTOS 2019 Kaggle page](https://www.kaggle.com/c/aptos2019-blindness-detection/rules) — Kaggle blocks API downloads until you've clicked "I Understand and Accept" on that page, even with a valid token. |
| Streamlit app is slow (~1-2s per prediction) | Expected on CPU — EfficientNet-B0 CPU inference is roughly 500-900ms per image plus preprocessing overhead. This is fine for a live single-image demo. If you need it faster, retrain with `BACKBONE = 'mobilenet_v2'` in the notebook (Section 5). |
| `cv2.imread` / preprocessing errors on a specific image | Make sure the uploaded file is a standard PNG/JPG fundus photo, not a PDF, HEIC, or corrupted file. |
| Predictions look wrong / inconsistent with training metrics | Double-check `app/preprocessing.py` hasn't been edited separately from the notebook's Section 3 — they must stay identical, since the model was trained on Ben-Graham-processed images specifically. |

---


---

## 8. Disclaimer

This is a hackathon prototype for demonstration and educational purposes only.
It is **not** a certified or validated medical device, has not undergone clinical
trials, and must not be used for actual patient diagnosis or treatment decisions.
The confidence-based manual-review flagging is a design pattern for responsible
ML UX, not a substitute for clinical judgment.

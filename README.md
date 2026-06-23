# Early-Stage Diabetic Retinopathy Detection

This project is a computer-vision prototype for detecting diabetic retinopathy (DR) from retinal fundus images.

The system takes one eye image and returns:

1. Whether diabetic retinopathy is likely present
2. The predicted DR severity grade from 0 to 4
3. A confidence or uncertainty signal
4. A manual-review warning when the model is unsure

The current project started as an EfficientNet-B0 ordinal-regression baseline and has now been extended with a feature-engineering plan and hybrid-model training code.

Important: this project is for learning, research, and demo purposes only. It is not a certified medical device and must not be used for real patient diagnosis.

---

## What is diabetic retinopathy?

Diabetic retinopathy is eye damage caused by diabetes. High blood sugar can damage small blood vessels in the retina. Over time, this can cause vision problems or blindness.

The model uses the APTOS-style 5-grade scale:

| Grade | Name | Meaning |
|---|---|---|
| 0 | No DR | No visible diabetic retinopathy |
| 1 | Mild NPDR | Small early signs, often microaneurysms |
| 2 | Moderate NPDR | More visible lesions and retinal changes |
| 3 | Severe NPDR | Serious non-proliferative disease |
| 4 | Proliferative DR | Advanced disease with high vision risk |

---

## Project structure

```text
Diabetic Retinopathy Detection/
+-- README.md
+-- SUMMARY.md
+-- DIABETIC_RETINOPATHY_CASE_STUDY.md
+-- Feature_Engineering_Modeling_Report.md
+-- HOW_IT_WORKS.md
+-- feature_engineering.py
+-- train_hybrid_model.py
+-- app/
    +-- app.py
    +-- api.py
    +-- inference.py
    +-- preprocessing.py
    +-- requirements.txt
    +-- model/
    |   +-- dr_model_traced.pt
    +-- sample_images/
```

Key files:

| File | Purpose |
|---|---|
| `README.md` | Main project introduction and setup guide |
| `SUMMARY.md` | Updated simple explanation of how the full system works, including new terms and features |
| `DIABETIC_RETINOPATHY_CASE_STUDY.md` | Problem background, clinical motivation, dataset, method, and expected impact |
| `Feature_Engineering_Modeling_Report.md` | Detailed feature-engineering and modeling strategy |
| `feature_engineering.py` | Extracts handcrafted fundus features from eye images |
| `train_hybrid_model.py` | Trains a CNN + engineered-feature hybrid model |
| `app/app.py` | Streamlit demo interface |
| `app/inference.py` | Loads the trained model and runs predictions |
| `app/preprocessing.py` | Ben Graham preprocessing used before prediction |

---

## Current model pipeline

The original baseline works like this:

```text
Fundus image
    ↓
Ben Graham preprocessing
    ↓
EfficientNet-B0 CNN
    ↓
Single ordinal regression output
    ↓
Grade 0-4 + confidence + manual-review flag
```

The improved hybrid training plan works like this:

```text
Fundus image
    ↓
Ben Graham preprocessing
    ↓
Two parallel paths
    ├── CNN path: EfficientNet-B0 learns deep visual features
    └── Feature path: handcrafted fundus features are extracted
            ↓
CNN features + engineered features are joined
            ↓
Fusion model predicts DR grade
            ↓
QWK-optimized thresholds convert score into grade 0-4
```

---

## New feature-engineering additions

The new `feature_engineering.py` file extracts extra signals from each fundus image. These features are meant to help the model see useful medical-style clues instead of relying only on the CNN.

Feature groups include:

| Feature group | Examples | Why it helps |
|---|---|---|
| Image quality | sharpness, entropy, illumination variation, field-of-view ratio | Detects blurry or poor-quality images |
| Color statistics | RGB mean, standard deviation, percentile values | Captures global color and contrast differences |
| Vessel features | vessel density, vessel component density | Retinal vessel changes can correlate with disease |
| Dark lesion candidates | small dark candidate count, larger dark candidate area | May capture microaneurysm/hemorrhage-like patterns |
| Bright lesion candidates | bright/yellow candidate count and area | May capture exudate-like patterns |
| Texture features | Local Binary Pattern histogram | Captures fine local texture patterns |

These are called candidates, not confirmed lesions. Classical image processing can make mistakes, so the features must be tested through ablation.

---

## Training workflow in Google Colab

Use the notebook first to prepare data and train.

1. Open the training notebook in Google Colab.
2. Turn on GPU: `Runtime -> Change runtime type -> T4 GPU`.
3. Install required packages.
4. Download the APTOS dataset from Kaggle.
5. Create the dataframe `df` with:
   - `id_code`
   - `diagnosis`
   - `img_path`
6. Upload or copy these two files into Colab:
   - `feature_engineering.py`
   - `train_hybrid_model.py`
7. Run feature extraction and save `aptos_features.csv`.
8. Merge extracted features into `df`.
9. Split data into train and validation sets.
10. Train the CNN-only baseline.
11. Train the hybrid CNN + engineered-feature model.
12. Compare results using QWK and accuracy.
13. Keep the hybrid model only if it improves repeatably across seeds/folds.

If you get `NameError: name 'df' is not defined`, it means the notebook does not currently remember the dataframe. Re-run the earlier cell that loads `train.csv` and creates `df["img_path"]`.

---

## Local demo workflow

After training, the app can run locally.

1. Put the trained model file in:

```text
app/model/dr_model_traced.pt
```

2. Install app dependencies:

```bash
cd app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Run Streamlit:

```bash
streamlit run app.py
```

4. Open the browser URL shown by Streamlit.
5. Upload a retinal fundus image.
6. Read the predicted grade, confidence, and manual-review warning.

Note: the current Streamlit app expects the older one-input TorchScript model. The new hybrid model needs image input plus engineered features, so the app inference code must be updated before deploying the hybrid model.

---

## Evaluation metrics

The most important metric is QWK.

| Metric | Meaning |
|---|---|
| Accuracy | Percentage of images where predicted grade equals true grade |
| QWK | Quadratic Weighted Kappa; gives smaller punishment for near misses and larger punishment for far mistakes |
| Confusion matrix | Shows which grades the model confuses |
| Ablation result | Shows whether a new feature group actually helps |

For diabetic retinopathy, QWK is more useful than plain accuracy because Grade 2 predicted as Grade 3 is less severe than Grade 0 predicted as Grade 4.

---

## Responsible-use warning

This project should be presented as a screening-assistance prototype, not a doctor replacement.

The app should always show:

- This is not a diagnosis.
- A trained medical professional must review real patient cases.
- Low-confidence or poor-quality images should be manually reviewed.
- Model results may be wrong, especially outside the APTOS-style dataset.

---

## Recommended next steps

1. Fix the notebook setup issue by making sure `df` is created before feature extraction.
2. Run feature extraction once and cache `aptos_features.csv`.
3. Train CNN-only and hybrid models on the same split.
4. Compare QWK, accuracy, and confusion matrices.
5. Repeat with multiple random seeds.
6. Update `app/inference.py` only after deciding which model is best.
7. Add image-quality warnings to the Streamlit app.
8. Add a visible medical disclaimer on every result screen.

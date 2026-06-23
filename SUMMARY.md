# System Summary: How the Diabetic Retinopathy Detection System Works

This file is the living summary of the project. Keep this file updated whenever we add a new method, model, feature, metric, or important term.

Think of the system like a careful helper:

1. It looks at a retina photo.
2. It cleans the photo.
3. It extracts useful clues.
4. It sends the cleaned image and clues to a model.
5. The model predicts the diabetic retinopathy grade.
6. The app warns the user if the model is unsure.

---

## 1. Big picture

The project predicts diabetic retinopathy severity from retinal fundus images.

The output grade is:

| Grade | Meaning |
|---|---|
| 0 | No diabetic retinopathy |
| 1 | Mild diabetic retinopathy |
| 2 | Moderate diabetic retinopathy |
| 3 | Severe diabetic retinopathy |
| 4 | Proliferative diabetic retinopathy |

The goal is not only to predict a grade, but also to make the system safer by adding:

- image-quality checks
- engineered features
- uncertainty/manual-review warnings
- clear medical disclaimers

---

## 2. Simple flow

```text
Input eye image
    ↓
Preprocessing
    ↓
Feature extraction
    ↓
Model prediction
    ↓
Grade + confidence/uncertainty
    ↓
Manual review warning if needed
```

---

## 3. Preprocessing

Preprocessing means cleaning the image before the model sees it.

The main preprocessing method is Ben Graham preprocessing.

It does three important things:

1. Crops away black borders around the eye.
2. Reduces lighting/color differences between images.
3. Keeps the circular retina area and removes corner artifacts.

Why this matters:

Different fundus cameras produce images with different brightness, color, and borders. Preprocessing makes the images more consistent, so the model can focus on disease signs instead of camera differences.

---

## 4. Baseline model

The baseline model uses EfficientNet-B0.

EfficientNet-B0 is a CNN, which means it learns visual patterns from images.

The baseline is trained as ordinal regression.

That means the model predicts one number, like `2.7`, instead of choosing one class directly. Then the number is converted into a grade.

Why ordinal regression:

DR grades are ordered. Grade 4 is more serious than Grade 3, and Grade 3 is more serious than Grade 2. Ordinal regression respects this order better than normal classification.

---

## 5. New engineered features

The new feature-engineering code extracts extra clues from the image.

These features are not a doctor’s diagnosis. They are computer-calculated signals that may help the model.

| Feature type | What it measures | Why it matters |
|---|---|---|
| Sharpness | Whether the image is blurry | Blurry images are harder to grade |
| Entropy | How much detail/texture exists | Low-detail images may be poor quality |
| Field-of-view ratio | How much of the image is retina | Badly captured images may miss useful retina area |
| Center offset | Whether the retina circle is centered | Off-center images may be unreliable |
| Illumination variation | Uneven lighting | Poor lighting can confuse predictions |
| RGB statistics | Color and brightness patterns | Captures global image appearance |
| Vessel density | Amount of vessel-like structure | Vessel changes may relate to disease |
| Dark candidates | Small/large dark spots | May represent microaneurysm or hemorrhage-like clues |
| Bright candidates | Bright/yellow spots | May represent exudate-like clues |
| LBP texture | Local texture patterns | Captures fine retinal texture |

Important: these are candidate features, not confirmed clinical lesions.

---

## 6. Hybrid model

The improved model is called a hybrid model because it uses two kinds of information:

1. Deep CNN features learned automatically from the image
2. Handcrafted engineered features calculated by image processing

Flow:

```text
Image
    ├── CNN extracts learned features
    └── Feature extractor calculates engineered features
            ↓
Both are joined together
            ↓
Fusion model predicts DR severity
```

The joining step is called feature fusion.

Why fusion helps:

The CNN may learn patterns that humans did not explicitly define. The engineered features add medical-style clues such as image quality, vessel density, and lesion-candidate counts. Together, they may be stronger than either one alone.

---

## 7. Feature cache

Feature extraction can take time, so the extracted features are saved into a CSV file:

```text
aptos_features.csv
```

This is called caching.

Caching means:

- extract features once
- save them
- load them next time instead of recalculating everything

---

## 8. Training

Training means teaching the model using labeled images.

The training data contains:

- image path
- true diagnosis grade
- engineered features

The training process:

1. Load images and labels.
2. Load engineered features.
3. Split data into train and validation sets.
4. Train the model on train data.
5. Check performance on validation data.
6. Save the best model.

---

## 9. Class imbalance

The APTOS dataset has many Grade 0 images and fewer severe Grade 3/4 images.

This is called class imbalance.

The project uses `WeightedRandomSampler` to show rare classes more often during training.

Important rule:

Do not use both heavy class-weighted loss and heavy weighted sampling at the same time unless testing proves it helps. Doing both can over-correct the imbalance.

---

## 10. Evaluation

The main evaluation metric is QWK, short for Quadratic Weighted Kappa.

QWK is useful because it understands distance between grades.

Example:

- True Grade 2, predicted Grade 3: small mistake
- True Grade 0, predicted Grade 4: huge mistake

QWK punishes the huge mistake more.

Other useful checks:

| Check | Purpose |
|---|---|
| Accuracy | Simple percentage correct |
| Confusion matrix | Shows which grades are confused |
| Classification report | Shows precision/recall per grade |
| Ablation study | Tests whether each new feature group helps |

---

## 11. Ablation study

Ablation means testing what happens when we remove or add parts.

Example:

| Experiment | Purpose |
|---|---|
| CNN only | Baseline |
| CNN + all engineered features | Tests full hybrid model |
| CNN + quality features only | Tests image-quality value |
| CNN + lesion-candidate features only | Tests lesion-candidate value |
| CNN + vessel features only | Tests vessel-feature value |

If the hybrid model does not improve QWK across repeated runs, use the CNN-only model for grading and keep engineered features for image-quality warnings.

---

## 12. Confidence and uncertainty

The older app uses a simple confidence heuristic:

- if the raw model score is close to a grade number, confidence is higher
- if it is near the boundary between grades, confidence is lower

Example:

- `2.02` is confidently Grade 2
- `2.49` is uncertain between Grade 2 and Grade 3

Future improvement:

Use MC Dropout for uncertainty. This runs the model many times with dropout active. If predictions vary a lot, the model is unsure.

---

## 13. Deployment

Deployment means making the trained model usable by people.

Current app:

- Streamlit web app for uploading images
- optional FastAPI endpoint
- CPU-friendly inference for demo use

Important deployment note:

The current Streamlit app expects a simple one-input TorchScript model. The new hybrid model uses both image input and engineered-feature input, so `app/inference.py` must be updated before using the hybrid model in the app.

---

## 14. Terms used in this project

| Term | Simple meaning |
|---|---|
| APTOS | Public diabetic retinopathy image dataset used for training/testing |
| Ben Graham preprocessing | Image cleanup method for fundus images |
| CNN | Neural network designed for images |
| EfficientNet-B0 | Lightweight CNN model used as the baseline backbone |
| Feature engineering | Creating useful numeric clues from raw data |
| Fundus image | Photo of the back of the eye |
| Hybrid model | Model using both CNN features and engineered features |
| Ordinal regression | Predicting an ordered number instead of unrelated classes |
| QWK | Metric that rewards close predictions and punishes far mistakes |
| WeightedRandomSampler | Training tool that helps rare classes appear more often |
| Feature fusion | Joining CNN features and engineered features |
| Feature cache | Saved feature file used to avoid recomputing features |
| Ablation study | Testing which parts of the system actually help |
| MC Dropout | Uncertainty method that runs the model many times |
| Manual review | Human expert should check the result |
| TorchScript | Saved PyTorch model format used by the app |
| Streamlit | Python tool used to create the demo web app |
| FastAPI | Python tool used to create an API endpoint |
| Image quality gate | Rule that warns/rejects poor-quality images |

---

## 15. Keep this file updated

Whenever a new method is added, update this file.

Add:

1. What the new thing is
2. Why it was added
3. Where it lives in the code
4. Whether it improved results
5. Any new term in the glossary

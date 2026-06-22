# Feature Engineering & Feature Modeling Report

### Early-Stage Diabetic Retinopathy Detection from Retinal Fundus Images

*A Strategy for Differentiated, Deployable, Kaggle-Submittable Modeling*

**Prepared for:** Project Lead, APTOS DR Detection Deployment Initiative
**Author perspective:** ML/AI Engineering Review
**Date:** June 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State of the Project](#2-current-state-of-the-project)
3. [Issues and Gaps to Address](#3-issues-and-gaps-to-address)
4. [Competitive Landscape Summary](#4-competitive-landscape-summary)
5. [Feature Engineering Strategy](#5-feature-engineering-strategy)
6. [Feature Modeling Strategy](#6-feature-modeling-strategy)
7. [Step-by-Step Implementation Procedure](#7-step-by-step-implementation-procedure)
8. [Risks and Mitigations](#8-risks-and-mitigations)
9. [Summary](#9-summary)

---

## 1. Executive Summary

The hackathon prototype proved the core pipeline works: ordinal-regression EfficientNet-B0 on Ben Graham-preprocessed APTOS images, with a single, non-overlapping class-imbalance correction. That prototype is a reasonable baseline — but a baseline is all it is. A literature scan of recent (2025–2026) published work on this exact dataset confirms that ordinal regression on EfficientNet, transfer-learning backbones, and basic preprocessing are now common, well-trodden approaches, not differentiators. Several published systems report accuracy in the 93–98% range using variations of the same recipe.

This report lays out what would actually make this project stand out for a public Kaggle submission and a real deployed tool: combining the CNN's learned features with engineered, domain-grounded features (lesion counts, vessel statistics, image-quality scores), adding calibrated uncertainty quantification suitable for a tool real users will rely on, and building a feature pipeline that is interpretable enough to explain a prediction, not just produce one.

> **Scope note:** This report covers feature engineering and feature modeling strategy specifically. It assumes the existing training notebook (ordinal EfficientNet-B0, Ben Graham preprocessing, WeightedRandomSampler) as the starting baseline, and proposes what to add on top of it — not a replacement of the existing pipeline.

This document is structured to be read top-to-bottom by someone returning to the project after stepping back from active development: it explains the current state, the competitive landscape, the specific issues with submitting a vanilla version, the proposed feature engineering and modeling strategy, and a concrete step-by-step execution plan.

---

## 2. Current State of the Project

### 2.1 What exists today

- **Data:** APTOS 2019 (3,662 labeled fundus images, grades 0–4), trimmed to a 2–4GB working set via selective extraction and 512px resizing.
- **Preprocessing:** Ben Graham method — crop to eye region, subtract local average color via Gaussian blur subtraction, clip to circular mask.
- **Model:** EfficientNet-B0 (ImageNet-pretrained) with the classification head replaced by a single linear output, trained as an ordinal regressor (Huber/SmoothL1 loss) rather than 5-way softmax classification.
- **Imbalance handling:** `WeightedRandomSampler` only (inverse-class-frequency sampling), deliberately not combined with loss weighting to avoid double-correction.
- **Evaluation:** Quadratic Weighted Kappa (QWK) as the primary metric, matching the original APTOS/EyePACS competition scoring convention.
- **Deployment:** Colab for training (GPU), Streamlit app for local CPU inference, optional FastAPI endpoint, with a heuristic confidence score derived from distance-to-nearest-integer-grade.

### 2.2 Why this is a solid baseline but not yet differentiated

The architecture and training choices in the current notebook are defensible and already better-reasoned than many public APTOS notebooks (most still treat this as plain 5-class classification). But independent literature confirms this exact combination — EfficientNet-B0 plus ordinal regression — has already been published and benchmarked, which means it will not read as novel to a Kaggle audience or to judges who have seen prior APTOS submissions.

> **Key finding from literature review:** A 2023 peer-reviewed study (Vijayan & Venkatakrishnan, *MDPI Diagnostics*) already proposes "viewing the detection of diabetic retinopathy as a regression problem rather than a traditional multi-class classification problem" using EfficientNet-B0 — essentially the same core framing as the current prototype. This needs to be cited as related work, not presented as the project's unique contribution.

---

## 3. Issues and Gaps to Address

Before adding features, it's worth being precise about what's actually missing. These are organized by category.

### 3.1 Differentiation issues

| Issue | Why it matters | What's needed |
|---|---|---|
| Backbone-swapping is saturated | Published work already covers ResNet-18+Swish, DenseNet, EfficientNet-B3, Inception-ResNet-v2, and quantum-hybrid backbones on this exact dataset, often at 93–98% accuracy. | Stop competing on backbone choice. Differentiate on what's fused with the backbone's output. |
| Ordinal regression alone is not novel | Already published specifically on EfficientNet-B0 for APTOS. | Use ordinal regression as the foundation, but add engineered features and uncertainty quantification on top — that combination is comparatively rare. |
| No lesion-level interpretability | A model that outputs only "Grade 2" gives no clinical reasoning. Reviewers and real users will ask "why." | Add engineered features tied to actual DR pathology (microaneurysms, hemorrhages, exudates, vessel patterns) that can be reported alongside the grade. |

### 3.2 Modeling and feature issues

- Current pipeline relies entirely on CNN-learned features — no explicit feature engineering layer exists. The CNN learns useful representations implicitly, but with only 3,662 training images, that representation can be noisy and is not interpretable.
- No image-quality gating: a blurry, poorly-illuminated, or off-center fundus photo is fed to the model the same as a high-quality one, and the model has no mechanism to flag "this image is unsuitable for grading" — a real issue once real users start uploading photos of unknown quality.
- Confidence scoring is a geometric heuristic (distance from the raw regression output to the nearest integer), not a statistically grounded uncertainty estimate. It is a reasonable placeholder but won't hold up to scrutiny in a published Kaggle notebook or a real deployment with liability implications.
- No feature-level explainability (e.g., which regions of the image drove the prediction) — important both for trust in a deployed tool and as a differentiator most competing notebooks skip.

### 3.3 Deployment issues (network-accessible, real users)

Moving from a hackathon demo to a tool real users can test against introduces new constraints the current design doesn't yet handle:

- **Input validation:** real users will upload non-fundus images, screenshots, or corrupted files. The current Streamlit app only checks decode success, not whether the image plausibly is a fundus photo at all.
- **Medical liability and framing:** a publicly accessible tool that returns a DR grade reads, to a layperson, as a diagnosis. This needs explicit, persistent, hard-to-miss disclaimer language, not a one-line caption.
- **Latency and scaling:** CPU inference at ~500–900ms per image is fine for a single local demo user but needs a request queue or async handling once multiple network users hit it concurrently.
- **No logging/monitoring layer** to detect drift (e.g., if uploaded images start looking systematically different from APTOS training data — different camera types, different populations).

---

## 4. Competitive Landscape Summary

Drawn from a review of recent peer-reviewed and preprint work specifically on the APTOS 2019 dataset and closely related fundus datasets (EyePACS, Messidor).

| Approach | Reported Result | Where it leaves a gap |
|---|---|---|
| ResNet-18 + Swish activation | 93.51% accuracy, validated on APTOS + real hospital data | Pure architecture tweak; no interpretability or uncertainty layer |
| DenseNet (Efficient DenseNet + KNN) | 98.40% test accuracy on 13,000 fundus images | High accuracy but a black box; not deployment-uncertainty-aware |
| EfficientNet-B0, regression framing | Published 2023, MDPI Diagnostics | Same core idea as current baseline — already not novel on its own |
| Hybrid Feature Fusion Network (deep + handcrafted features) + multi-stage uncertainty classifier | 2025 arXiv preprint, explicit focus on clinical trust | Closest match to the recommended direction in this report — validates the strategy but is itself very recent, meaning few public Kaggle notebooks have adopted it yet |
| Quantum transfer learning hybrids (ResNet+VQC) | 97% accuracy with ResNet-18 backbone | Novel but impractical for the stated deployment goal (laptop-CPU Streamlit app, no quantum hardware) |

> **Strategic takeaway:** The clearest, most defensible whitespace is the combination of (a) engineered, pathology-grounded features fused with CNN features, and (b) rigorous, calibrated uncertainty quantification built for deployment — not just raw accuracy chasing on a backbone swap.

---

## 5. Feature Engineering Strategy

Feature engineering here means producing additional, explicit numerical features from each fundus image — grounded in known DR pathology — to fuse with the CNN's learned feature vector, rather than relying on the CNN alone to implicitly discover everything from raw pixels.

### 5.1 Why add engineered features at all, given a CNN already exists

CNNs learn features end-to-end, but with only ~3,662 labeled training images, the network has limited data from which to discover subtle pathological patterns reliably on its own. Explicit, domain-grounded features act as a strong prior — information clinicians already know is diagnostic — injected directly into the model rather than left to chance discovery. This is exactly the rationale behind the published Hybrid Feature Fusion Network approach referenced in Section 4: combining deep learning embeddings with handcrafted features to improve generalization on limited data.

### 5.2 Proposed engineered feature groups

**Group A — Lesion-density features**
- **Microaneurysm candidate count:** small, round, dark dot-like structures, the earliest DR sign. Detectable via morphological top-hat filtering on the green color channel (microaneurysms have highest contrast there) followed by candidate-region counting.
- **Hemorrhage area ratio:** larger, irregular dark blotches. Detectable via adaptive thresholding plus connected-component area filtering, expressed as a fraction of total retinal area.
- **Hard/soft exudate area ratio:** bright, yellowish lesions. Detectable via thresholding on the contrast-enhanced green/luminance channel, distinguished from the optic disc by location and shape exclusion.

**Group B — Vascular features**
- **Vessel density:** fraction of retinal area occupied by blood vessels, extracted via a vessel-segmentation filter (e.g., a simple Frangi vesselness filter, fast enough to run as a preprocessing feature rather than a full segmentation network).
- **Vessel tortuosity index:** a classical, well-published handcrafted feature measuring how curved/twisted vessels are — increased tortuosity correlates with disease severity in prior DR literature.

**Group C — Image-quality features (also serve as input gating)**
- **Sharpness score** (e.g., variance of Laplacian) — flags blurry uploads.
- **Illumination uniformity** — flags poorly lit or unevenly exposed photos.
- **Field-of-view / centration check** — flags photos where the optic disc/macula aren't reasonably centered, a common real-world capture error.

**Group D — Global statistical features**
- **Color channel histogram statistics** (mean, variance, skew per RGB channel) post-Ben-Graham-preprocessing — cheap to compute, captures residual global tint/contrast patterns.
- **Local Binary Pattern (LBP) texture histograms** — a classical texture descriptor that complements CNN features by capturing fine local texture statistics in a rotation-invariant way.

### 5.3 How these get used (not bolted on naively)

The risk with handcrafted features is treating them as an afterthought concatenated onto CNN output with no real integration. The recommended approach:

1. Extract each engineered feature group as a fixed-length numeric vector per image (Groups A–D above), independent of the CNN.
2. Pass the CNN's penultimate-layer embedding (the feature vector right before the final regression head, typically ~1280-dim for EfficientNet-B0) through a small projection layer to reduce it to a manageable size.
3. Concatenate the projected CNN embedding with the normalized engineered feature vector.
4. Feed the concatenated vector through a small fusion MLP (1–2 hidden layers) ending in the same single ordinal-regression output used today — this preserves the ordinal regression framing while adding the fused features as additional signal.
5. Train end-to-end, but monitor whether engineered features are actually contributing (see Section 6.4, ablation) rather than assuming they help by default.

---

## 6. Feature Modeling Strategy

### 6.1 Model architecture (Hybrid Feature Fusion)

Building directly on the existing ordinal-regression EfficientNet-B0 baseline:

| Component | Role |
|---|---|
| EfficientNet-B0 backbone (existing) | Produces a learned, high-dimensional image embedding — unchanged from current baseline |
| Engineered feature extractor (new) | Runs Groups A–D (Section 5.2) on each image during preprocessing, producing a fixed-length numeric vector |
| Projection layer (new) | Small linear + ReLU layer reducing CNN embedding to ~128 dims, keeping the fusion layer from being dominated by CNN dimensionality |
| Feature normalization (new) | Z-score normalize engineered features using training-set statistics, so lesion-count features (which can range widely) don't dominate or vanish next to bounded CNN features |
| Fusion MLP (new) | Concatenates projected CNN embedding + normalized engineered features, passes through 1–2 hidden layers (e.g., 256 → 64) with dropout |
| Ordinal regression head (existing) | Single linear output, Huber loss — unchanged framing from current baseline |

### 6.2 Uncertainty quantification (the deployment-critical addition)

This is the single highest-leverage addition for a real, network-deployed tool. The current heuristic confidence score (distance from raw regression output to nearest integer) is a reasonable placeholder but is not statistically calibrated — it has no guarantee of actually tracking real error rates.

> **Recommended method: Monte Carlo (MC) Dropout.** Recent published work (OpenReview, 2026) specifically studied where to place dropout for the most reliable uncertainty calibration, finding that applying dropout in the penultimate layer consistently gives the most monotonic, actionable uncertainty — validated on a real clinical triage task (mammography). The current model already has a `Dropout(0.3)` layer immediately before the final regression output, which is precisely the recommended placement — this is a low-cost upgrade, not a redesign.

How it works in practice: instead of disabling dropout at inference time (the normal behavior), keep it active and run the same image through the model multiple times (e.g., 20–30 forward passes). Each pass gives a slightly different prediction due to the random dropout mask. The spread (standard deviation) of these predictions becomes the uncertainty estimate — a tight cluster of predictions means high confidence, a wide spread means the model is genuinely unsure, not just "near a rounding boundary."

- This directly replaces the current distance-to-boundary heuristic with a statistically grounded epistemic uncertainty estimate.
- Published results show this approach detects a meaningful share of a model's actual mispredictions — not just predictions that happen to round awkwardly — making it a genuinely more trustworthy basis for the existing "flag for manual review" feature.
- Computational cost: ~20–30x inference passes per image instead of 1. On CPU at ~500–900ms per single pass, this pushes a single prediction to several seconds — acceptable for a deployed single-image-at-a-time tool, but should be made async/queued in the web deployment rather than blocking.

### 6.3 Ordinal-aware fusion: keep what works

The existing decision to frame this as ordinal regression rather than 5-way classification remains correct and should not be abandoned for the "hybrid" model — it should simply receive richer input features. QWK remains the right evaluation metric for the same reasons as the current baseline.

### 6.4 Validation: feature ablation study

To make a credible Kaggle submission, it isn't enough to claim engineered features help — this needs to be demonstrated quantitatively. Run a controlled ablation:

| Configuration | Purpose | Compare against |
|---|---|---|
| CNN only (current baseline) | Establishes the floor — exactly the current notebook's performance | — |
| CNN + engineered features (all groups) | Tests whether fusion improves QWK/accuracy over baseline | vs. CNN only |
| CNN + individual feature groups (A, B, C, D separately) | Identifies which feature group actually contributes — avoids shipping dead weight | vs. CNN + all groups |

This ablation table, reported directly in the Kaggle notebook, is itself a differentiator — most public APTOS notebooks report only a single final accuracy number with no analysis of what's actually driving performance.

---

## 7. Step-by-Step Implementation Procedure

Organized as sequential phases. Each phase produces something runnable/testable before moving to the next — don't build the full hybrid pipeline before confirming each piece in isolation.

### Phase 1 — Engineered feature extraction module

1. Write a standalone Python module (e.g., `feature_extraction.py`) implementing Groups A–D from Section 5.2, each as an independent function taking a Ben-Graham-preprocessed image and returning a fixed-length numeric vector.
2. Run it across the full APTOS training set once, caching the resulting feature vectors to disk (e.g., a Parquet/CSV file keyed by image ID) — these are deterministic given the image, so there's no need to recompute them every training run.
3. Sanity-check feature distributions per grade (e.g., box plots of microaneurysm count by grade) — confirm the engineered features actually correlate with severity before trusting them in the model. If a feature shows no separation across grades, reconsider or drop it.

### Phase 2 — Fusion model architecture

1. Extend the existing `DRModel` class (or create `DRHybridModel`) to accept two inputs: the image tensor (for the CNN backbone) and the precomputed engineered feature vector.
2. Add the projection layer, normalization, and fusion MLP as described in Section 6.1.
3. Verify the model runs end-to-end on a single batch (forward pass only) before attempting full training — catch shape/dimension mismatches early.

### Phase 3 — Training and ablation

1. Train the CNN-only baseline first (this is close to the existing notebook — confirms the starting point hasn't regressed).
2. Train the full hybrid model (CNN + all engineered feature groups).
3. Train per-group ablations (CNN + Group A only, CNN + Group B only, etc.) to isolate which features matter, per Section 6.4.
4. Record QWK, accuracy, and confusion matrices for every configuration in a single comparison table — this table becomes a core section of the Kaggle write-up.

### Phase 4 — Uncertainty quantification

1. Confirm the existing `Dropout(0.3)` layer sits immediately before the final regression output (penultimate-layer placement) — already true in the current architecture.
2. Implement MC Dropout inference: keep dropout active at test time, run N=20–30 forward passes per image, compute mean prediction and standard deviation.
3. Replace the current distance-to-boundary confidence heuristic with the MC Dropout standard deviation, recalibrating the existing 60% manual-review threshold against the new uncertainty scale (the two are not on the same numeric scale, so the threshold value itself will need to be re-tuned, not copied directly).
4. Validate calibration: bucket predictions by uncertainty level and confirm error rate actually increases as uncertainty increases (an "accuracy-uncertainty curve") — this is the check that distinguishes genuine calibration from a cosmetic confidence number.

### Phase 5 — Deployment hardening for network/public users

1. Add an image-quality gate using Group C features (Section 5.2): reject or warn on images below a sharpness/illumination threshold before they reach the model at all.
2. Add a basic "is this plausibly a fundus photo" check (e.g., expected circular structure, expected color profile) to reject obviously wrong uploads (screenshots, random photos) gracefully rather than producing a nonsense grade.
3. Move inference to an async task queue (e.g., FastAPI background tasks or a lightweight job queue) if deploying for concurrent network users, given MC Dropout's added latency from Phase 4.
4. Add prominent, persistent medical disclaimer UI (not just a caption) — every results screen should restate that this is not a diagnostic device.
5. Add basic request logging (image-quality scores, predicted grade, uncertainty, timestamp — not the image itself, for privacy) to monitor for data drift once real users start uploading.

### Phase 6 — Kaggle submission packaging

1. Structure the Kaggle notebook narrative around the differentiation story: start from the published "ordinal regression on EfficientNet" baseline (cite it), then show the ablation study proving the engineered-feature fusion adds measurable value, then show the calibrated uncertainty quantification with its accuracy-uncertainty curve.
2. Include the confusion matrix and QWK comparison table (Section 6.4) directly in the notebook — reviewers reward demonstrated rigor over a single headline accuracy number.
3. Cite related work explicitly (the regression-framing paper, the hybrid feature fusion paper, the MC Dropout calibration paper) — this signals awareness of the field rather than presenting known techniques as novel claims.
4. Include a clear "limitations" section (small dataset size, single-dataset training, not clinically validated) — this reads as more credible to experienced reviewers than an unqualified accuracy claim.

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Engineered features turn out not to help (ablation shows no gain) | Wasted engineering effort if not caught early | Phase 1's correlation sanity-check and Phase 3's per-group ablation are designed specifically to catch this before full investment |
| MC Dropout latency makes the deployed app feel slow | Poor user experience for network users | Make inference async (Phase 5); consider reducing N (forward pass count) if 20–30 proves too slow, trading some calibration precision for speed |
| Public deployment of a health-adjacent tool draws scrutiny over medical claims | Reputational/liability concern | Explicit, persistent disclaimers (Phase 5); frame consistently as a screening-assistance prototype, never as diagnostic |
| Small dataset (3,662 images) limits how much a hybrid model can improve over baseline | Capped ceiling on achievable performance | Be honest about this in the Kaggle write-up's limitations section — credibility matters more than an inflated claim |

---

## 9. Summary

The existing prototype's core decisions — ordinal regression, single imbalance-correction mechanism, Ben Graham preprocessing — remain sound and should not be discarded. What's missing for a genuinely differentiated, deployment-ready, Kaggle-credible submission is:

1. Explicit, pathology-grounded engineered features fused with the CNN's learned representation, validated through ablation rather than assumed.
2. Statistically calibrated uncertainty quantification via MC Dropout, replacing the current heuristic, directly enabling safer manual-review flagging for real users.

Both additions build on architecture already in place (the existing Dropout layer, the existing ordinal regression head) rather than requiring a redesign — this is an extension of the current work, not a restart.

---

## 10. Code Delivered and Corrections to the Current Notebook

Two implementation files now accompany this report:

- `feature_engineering.py`: deterministic, 512px fundus feature extraction using OpenCV and NumPy.
- `train_hybrid_model.py`: EfficientNet-B0 feature fusion, train-only normalization, differential learning rates, mixed precision, gradient clipping, early stopping, and optimized QWK thresholds.

### 10.1 Corrections found in `train_dr_model (1).ipynb`

1. Add `import json` before `json.dump()` in the new Kaggle-token path.
2. Do not extract lesion candidates from the 224x224 CNN input. Extract deterministic features at 512x512 and use 300x300 for the CNN.
3. Fit engineered-feature mean and standard deviation on `train_df` only to prevent validation leakage.
4. Replace fixed `np.round()` cut points with four monotonic thresholds optimized on validation QWK. Save and reuse them at inference.
5. Use a lower learning rate for pretrained backbone weights and a higher rate for new fusion/head weights.
6. One 85/15 split is noisy for 3,662 images. Report at least three stratified seeds or five-fold cross-validation.
7. The value in `app/inference.py` is a boundary heuristic, not calibrated confidence.
8. The Gaussian grade bars are display scores, not probabilities.

### 10.2 Precompute and cache engineered features

Run this after creating `df['img_path']`:

```python
from pathlib import Path
import cv2
import pandas as pd
from tqdm.auto import tqdm
from feature_engineering import FundusFeatureExtractor

FEATURE_CACHE = f"{DRIVE_PROJECT_DIR}/features/aptos_features.csv"
Path(FEATURE_CACHE).parent.mkdir(parents=True, exist_ok=True)

if Path(FEATURE_CACHE).exists():
    feature_df = pd.read_csv(FEATURE_CACHE)
else:
    extractor = FundusFeatureExtractor(feature_size=512)
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        image = cv2.imread(row.img_path)
        try:
            values, names = extractor.extract(image)
            rows.append({"id_code": row.id_code, **dict(zip(names, values))})
        except Exception as exc:
            print(f"Feature extraction failed for {row.id_code}: {exc}")
    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(FEATURE_CACHE, index=False)

df = df.merge(feature_df, on="id_code", how="inner", validate="one_to_one")
feature_columns = [column for column in feature_df.columns if column != "id_code"]
print(f"Usable images: {len(df)}; engineered features: {len(feature_columns)}")
```

The extractor intentionally calls outputs `dark_*` and `bright_candidate_*`, not definitive lesion diagnoses. Classical morphology can confuse normal anatomy with pathology; ablation must determine whether each group helps.

### 10.3 Train the hybrid model

```python
from sklearn.model_selection import train_test_split
import torch
from train_hybrid_model import (
    DRHybridModel, CNNOnlyModel, make_loaders, save_inference_metadata,
    seed_everything, train_model,
)

seed_everything(42)
train_df, val_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["diagnosis"]
)
train_loader, val_loader, feature_mean, feature_std = make_loaders(
    train_df, val_df, feature_columns, image_size=300, batch_size=24, num_workers=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hybrid = DRHybridModel(engineered_dim=len(feature_columns))
result = train_model(
    hybrid, train_loader, val_loader, device,
    f"{DRIVE_PROJECT_DIR}/models/dr_hybrid_best.pt",
    epochs=25, patience=6,
)
save_inference_metadata(
    f"{DRIVE_PROJECT_DIR}/models/dr_hybrid_metadata.json",
    feature_columns, feature_mean, feature_std, result["thresholds"], image_size=300,
)
print("Best validation QWK:", result["best_qwk"])
```

### 10.4 Fair CNN-only ablation

```python
cnn_only = CNNOnlyModel()
cnn_result = train_model(
    cnn_only, train_loader, val_loader, device,
    f"{DRIVE_PROJECT_DIR}/models/dr_cnn_only_best.pt",
    epochs=25, patience=6,
)
print({
    "cnn_only_qwk": cnn_result["best_qwk"],
    "hybrid_qwk": result["best_qwk"],
    "absolute_gain": result["best_qwk"] - cnn_result["best_qwk"],
})
```

If the gain is not repeatable across seeds or folds, retain quality features for upload gating but use the CNN-only model for grading.

### 10.5 Recommended accuracy-improvement order

1. Increase CNN input from 224 to 300 while retaining 512px source images.
2. Optimize and save QWK thresholds instead of fixed rounding.
3. Use differential learning rates and stronger but plausible augmentation.
4. Measure three seeds or five folds; select changes by mean QWK.
5. Add engineered features and retain only groups that win ablation.
6. If compute permits, average raw outputs from fold models before applying thresholds.

No code change can guarantee higher unseen-test accuracy. This procedure makes improvements measurable and prevents leakage or unfair comparisons.

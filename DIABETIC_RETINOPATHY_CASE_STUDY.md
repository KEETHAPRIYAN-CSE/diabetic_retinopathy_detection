# Case Study: Diabetic Retinopathy Detection Using Fundus Images

## 1. Problem overview

Diabetic retinopathy is a diabetes-related eye disease. It damages the blood vessels in the retina, which is the light-sensitive layer at the back of the eye.

If diabetic retinopathy is not found early, it can lead to serious vision loss or blindness.

The challenge is that many people with early diabetic retinopathy may not notice symptoms. This makes regular screening very important.

---

## 2. Why this problem matters

Diabetes is common worldwide, and diabetic retinopathy is one of the major preventable causes of blindness.

Early detection can help doctors:

- monitor patients earlier
- recommend follow-up care
- prevent disease progression
- reduce avoidable vision loss

However, screening every patient manually can be difficult when there are many patients and limited eye specialists.

This creates an opportunity for AI-assisted screening.

The goal is not to replace doctors. The goal is to help prioritize cases and flag images that may need expert review.

---

## 3. Medical background

Diabetic retinopathy progresses in stages.

| Grade | Stage | Description |
|---|---|---|
| 0 | No DR | No visible diabetic retinopathy |
| 1 | Mild NPDR | Small early abnormalities, often microaneurysms |
| 2 | Moderate NPDR | More visible lesions and retinal changes |
| 3 | Severe NPDR | Serious vessel damage and high risk of progression |
| 4 | Proliferative DR | Advanced disease with abnormal new blood vessels |

Common visual signs can include:

- microaneurysms
- hemorrhages
- hard exudates
- soft exudates/cotton-wool spots
- abnormal vessel patterns
- poor retinal blood flow
- neovascularization in advanced cases

The AI model does not truly “understand” the eye like a clinician. It learns visual patterns that correlate with these grades.

---

## 4. Dataset used

The project is based on the APTOS 2019 diabetic retinopathy dataset.

The dataset contains retinal fundus images labeled from Grade 0 to Grade 4.

Each training row includes:

- `id_code`: image identifier
- `diagnosis`: true DR grade
- image file: retinal fundus photo

Main dataset challenge:

APTOS images are not perfectly uniform. They can vary in:

- brightness
- color tone
- camera quality
- blur
- image size
- black border size
- retina position

This is why preprocessing and image-quality features are important.

---

## 5. Baseline solution

The baseline system uses a CNN model called EfficientNet-B0.

The image is first cleaned using Ben Graham preprocessing.

Then the CNN predicts a single number representing severity.

This is called ordinal regression.

Why ordinal regression is useful:

The grades have an order. Grade 4 is more severe than Grade 3. Grade 3 is more severe than Grade 2. A single-number prediction respects this order better than treating all grades as unrelated categories.

Baseline flow:

```text
Fundus image
    ↓
Ben Graham preprocessing
    ↓
EfficientNet-B0
    ↓
Ordinal regression score
    ↓
Predicted grade 0-4
```

---

## 6. Improved feature-engineering solution

The improved project adds handcrafted features.

These features are calculated directly from the fundus image and then combined with the CNN’s learned features.

This creates a hybrid model.

Feature groups:

| Feature group | Example signals |
|---|---|
| Quality features | sharpness, entropy, illumination variation |
| Color features | RGB mean, standard deviation, percentile values |
| Vessel features | vessel density and vessel-like component density |
| Dark candidate features | possible microaneurysm/hemorrhage-like dark regions |
| Bright candidate features | possible exudate-like bright/yellow regions |
| Texture features | Local Binary Pattern histogram |

Hybrid model flow:

```text
Fundus image
    ├── CNN learns deep visual features
    └── Feature extractor calculates engineered features
            ↓
Feature fusion
            ↓
Final DR grade prediction
```

---

## 7. Why engineered features help

CNNs are powerful, but the dataset is relatively small for medical deep learning.

Engineered features give the model extra structured clues.

For example:

- sharpness can warn that an image may be too blurry
- vessel density may capture vascular pattern changes
- dark candidate regions may capture microaneurysm or hemorrhage-like patterns
- bright candidate regions may capture exudate-like patterns
- texture features may capture subtle retinal surface changes

These features also make the system more explainable than a pure black-box CNN.

Important limitation:

The engineered features are not certified lesion detectors. They are approximate computer-vision clues and must be validated by experiments.

---

## 8. Evaluation strategy

The main metric is QWK: Quadratic Weighted Kappa.

QWK is well-suited because diabetic retinopathy grades are ordered.

A model should be punished more for predicting Grade 0 when the truth is Grade 4 than for predicting Grade 2 when the truth is Grade 3.

The project should report:

- QWK
- accuracy
- confusion matrix
- classification report
- ablation results

Recommended experiments:

| Experiment | Purpose |
|---|---|
| CNN only | Establish baseline |
| CNN + all engineered features | Test full hybrid model |
| CNN + quality features | Check quality feature value |
| CNN + dark/bright candidate features | Check lesion-candidate value |
| CNN + vessel features | Check vessel feature value |

---

## 9. Deployment plan

The system can be demonstrated through a Streamlit app.

User workflow:

1. User uploads a fundus image.
2. App checks/loads the image.
3. Image is preprocessed.
4. Model predicts DR grade.
5. App displays:
   - predicted grade
   - DR present/not present
   - confidence or uncertainty
   - manual-review warning if needed
   - medical disclaimer

For the hybrid model, the deployment code must also extract engineered features during inference.

That means the app must pass both:

- image tensor
- engineered feature vector

to the model.

---

## 10. Risks and limitations

This project has important limitations.

| Risk | Explanation |
|---|---|
| Not clinically validated | The model has not passed medical-device testing |
| Dataset limitation | APTOS may not represent all cameras, populations, or clinics |
| Image quality variation | Blurry or badly captured images may produce unreliable results |
| Feature approximation | Engineered features are candidates, not confirmed lesions |
| False positives | Healthy images may be incorrectly flagged |
| False negatives | Diseased images may be missed |

Because of these risks, every real-world case must be reviewed by a qualified medical professional.

---

## 11. Expected impact

If developed responsibly, a system like this could help:

- support early screening
- reduce workload for specialists
- prioritize high-risk images
- warn when image quality is poor
- provide educational demonstrations of medical AI

The safest framing is:

AI-assisted screening support, not automated diagnosis.

---

## 12. Conclusion

This case study shows how a diabetic retinopathy detection system can be built using retinal images, deep learning, preprocessing, engineered features, and uncertainty-aware design.

The strongest version of the project is not just a model that predicts a grade. It is a full pipeline that:

1. cleans the image
2. checks image quality
3. extracts medically meaningful candidate features
4. predicts severity
5. reports uncertainty
6. asks for manual review when needed
7. clearly explains that the tool is not a medical diagnosis

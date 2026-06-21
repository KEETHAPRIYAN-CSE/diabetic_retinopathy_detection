"""
Streamlit app: Early-Stage Diabetic Retinopathy Detection.

Biothon 2026 — Problem P2.

Run with:  streamlit run app.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from inference import DRPredictor, UNCERTAINTY_THRESHOLD

MODEL_PATH = Path(__file__).parent / "model" / "dr_model_traced.pt"

st.set_page_config(
    page_title="DR Detection — Biothon 2026",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_predictor():
    return DRPredictor(model_path=str(MODEL_PATH), img_size=224, device="cpu")


def grade_color(grade: int) -> str:
    return {
        0: "#2e7d32",  # green
        1: "#9e9d24",  # yellow-green
        2: "#f9a825",  # amber
        3: "#ef6c00",  # orange
        4: "#c62828",  # red
    }[grade]


st.title("🩺 Diabetic Retinopathy Detection")
st.caption(
    "Biothon 2026 · Problem P2 — Early-Stage Diabetic Retinopathy Detection · "
    "Agriculture-track team, Python/ML stack"
)

if not MODEL_PATH.exists():
    st.error(
        f"**Model file not found** at `{MODEL_PATH}`.\n\n"
        "1. Run `notebook/train_dr_model.ipynb` in Google Colab.\n"
        "2. After training finishes, it saves `dr_model_traced.pt` to your Google Drive "
        "under `biothon2026-dr-detection/models/`.\n"
        "3. Download that file and place it at `app/model/dr_model_traced.pt` "
        "in this project folder.\n"
        "4. Restart this app."
    )
    st.stop()

with st.spinner("Loading model..."):
    predictor = load_predictor()

st.success("Model loaded. Ready for inference.", icon="✅")

with st.expander("ℹ️ How this works / methodology notes", expanded=False):
    st.markdown(
        """
- **Model**: EfficientNet-B0 backbone (ImageNet pretrained), fine-tuned on APTOS 2019
  as an **ordinal regressor** (single continuous output, not 5-way softmax classification)
  — DR grades are ordered, so regression + rounding better respects "near misses"
  than treating every wrong class as equally wrong.
- **Preprocessing**: Ben Graham preprocessing (crop, subtract local average color,
  clip to circular mask) is applied to every image before inference, exactly matching
  what the model was trained on.
- **Class imbalance**: handled once, in training, via `WeightedRandomSampler`
  (not combined with class-weighted loss — that would double-correct).
- **Confidence score** is derived from how close the raw regression output is to an
  integer grade boundary, not a generic softmax max-probability.
- **This is a hackathon prototype, not a diagnostic device.** It is not validated
  for clinical use. Predictions below the confidence threshold are explicitly
  flagged for manual review rather than presented as reliable.
        """
    )

st.divider()

uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["png", "jpg", "jpeg"],
    help="Use a standard fundus photograph, e.g. from the APTOS dataset or a fundus camera.",
)

col_a, col_b = st.columns(2)
with col_a:
    use_sample = st.checkbox("Use a bundled sample image instead", value=False)

sample_dir = Path(__file__).parent / "sample_images"
sample_choice = None
if use_sample:
    samples = sorted(sample_dir.glob("*.png")) + sorted(sample_dir.glob("*.jpg"))
    if samples:
        sample_choice = st.selectbox("Sample image", samples, format_func=lambda p: p.name)
    else:
        st.warning(
            f"No sample images found in `{sample_dir}`. Add a few APTOS images there, "
            "or uncheck this box and upload your own."
        )

image_bytes = None
if use_sample and sample_choice is not None:
    image_bytes = sample_choice.read_bytes()
elif uploaded_file is not None:
    image_bytes = uploaded_file.read()

if image_bytes is None:
    st.info("Upload an image or select a sample to run a prediction.")
    st.stop()

file_array = np.frombuffer(image_bytes, dtype=np.uint8)
bgr_img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

if bgr_img is None:
    st.error("Couldn't decode that file as an image. Please upload a valid PNG/JPG.")
    st.stop()

st.image(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), caption="Input image", width=350)

run_clicked = st.button("Run prediction", type="primary")

if run_clicked:
    with st.spinner("Running inference... (CPU inference, ~0.5-1s per image)"):
        start = time.time()
        result = predictor.predict(bgr_img)
        elapsed = time.time() - start

    st.divider()
    st.subheader("Result")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f"<h2 style='color:{grade_color(result.grade)}'>"
            f"Grade {result.grade} — {result.grade_label}</h2>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{result.confidence * 100:.1f}%")
        st.metric("DR present", "Yes" if result.dr_present else "No")
    with col2:
        st.caption(f"Raw regression output: `{result.raw_score:.3f}`")
        st.caption(f"Inference time: `{elapsed * 1000:.0f} ms` (CPU)")

    if result.needs_manual_review:
        st.warning(
            f"⚠️ **Confidence ({result.confidence * 100:.1f}%) is below the "
            f"{UNCERTAINTY_THRESHOLD * 100:.0f}% threshold.** This prediction sits "
            f"near a grade boundary — flagging for manual clinical review rather than "
            f"presenting it as reliable.",
            icon="⚠️",
        )
    else:
        st.info(
            f"Confidence is above the {UNCERTAINTY_THRESHOLD * 100:.0f}% review threshold.",
            icon="✅",
        )

    st.markdown("**Clinical recommendation:**")
    st.write(result.clinical_recommendation)

    st.markdown("**Grade probability distribution** (visualization of model certainty across grades):")
    for g in range(5):
        p = result.probability_bars[g]
        st.write(f"Grade {g} ({['No DR','Mild','Moderate','Severe','Proliferative'][g]})")
        st.progress(p, text=f"{p * 100:.1f}%")

    st.caption(
        "Disclaimer: This is a hackathon prototype for demonstration purposes only. "
        "It is not a certified medical device and must not be used for real clinical "
        "diagnosis or treatment decisions."
    )

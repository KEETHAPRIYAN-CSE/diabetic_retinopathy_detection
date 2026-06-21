"""
Inference wrapper around the trained DR detection model.

Key design point: the model is trained as an ORDINAL REGRESSOR (single scalar
output, not 5-way softmax). That means we don't get a natural softmax
probability vector "for free." We derive:

  1. `grade` — the rounded, clipped regression output (0-4).
  2. `confidence` — a calibrated confidence score derived from how far the raw
     regression output sits from the nearest integer grade boundary. A raw
     output of 2.02 is confidently Grade 2; a raw output of 2.49 is right on
     the boundary with Grade 2/3 and should report LOW confidence, not a fake
     high number.
  3. `probability_bars` — a soft pseudo-distribution over the 5 grades for the
     UI's probability bars, built from a Gaussian centered on the raw
     regression output. This is for visualization only; the model does not
     literally output a softmax distribution.

This distinction matters: a model outputting "Grade 2" from a raw value of
1.95 and one outputting "Grade 2" from a raw value of 2.49 are NOT equally
trustworthy, even though they round to the same grade. Collapsing that
distinction into a single number is exactly the bug this module avoids.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from preprocessing import ben_graham_preprocess

GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

CLINICAL_RECOMMENDATIONS = {
    0: "No signs of diabetic retinopathy detected. Routine annual screening recommended.",
    1: "Mild non-proliferative DR detected. Recommend re-screening in 9-12 months.",
    2: "Moderate non-proliferative DR detected. Recommend ophthalmologist referral within 3-6 months.",
    3: "Severe non-proliferative DR detected. Recommend prompt ophthalmologist referral within 1 month.",
    4: "Proliferative DR detected. Recommend urgent ophthalmologist referral (within days).",
}

UNCERTAINTY_THRESHOLD = 0.60  # below this confidence, flag for manual review

# Below this raw-output distance from an integer grade, we consider the
# prediction "on a boundary" and cap confidence accordingly.
BOUNDARY_ZONE = 0.5


@dataclass
class PredictionResult:
    grade: int
    grade_label: str
    raw_score: float
    confidence: float
    probability_bars: dict
    needs_manual_review: bool
    clinical_recommendation: str
    dr_present: bool


def _confidence_from_raw_score(raw_score: float) -> float:
    """Distance-to-nearest-integer based confidence, in [0, 1].

    raw_score exactly on an integer (e.g. 2.00) -> confidence 1.0
    raw_score exactly on a boundary (e.g. 2.50) -> confidence ~0.0 (floor below)
    """
    nearest_int = round(raw_score)
    distance = abs(raw_score - nearest_int)  # in [0, 0.5]
    # Map distance 0 -> confidence 1.0, distance 0.5 -> confidence ~0.35 floor
    # (a floor, not 0, because being on a boundary between two adjacent
    # clinically-similar grades is uncertain but not meaningless).
    confidence = 1.0 - (distance / BOUNDARY_ZONE) * 0.65
    return float(np.clip(confidence, 0.0, 1.0))


def _soft_probability_bars(raw_score: float, spread: float = 0.6) -> dict:
    """Gaussian pseudo-distribution over grades 0-4, centered on raw_score.

    This is purely for the UI's probability bars — it visualizes how close
    the continuous prediction sits to neighboring grades. It is NOT a
    calibrated softmax output (the model has no classification head).
    """
    grades = np.arange(5)
    weights = np.exp(-0.5 * ((grades - raw_score) / spread) ** 2)
    weights = weights / weights.sum()
    return {int(g): float(w) for g, w in zip(grades, weights)}


class DRPredictor:
    def __init__(self, model_path: str, img_size: int = 224, device: str = "cpu"):
        self.device = torch.device(device)
        self.img_size = img_size
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Train it in the Colab notebook "
                f"(notebook/train_dr_model.ipynb), download dr_model_traced.pt from "
                f"your Google Drive, and place it at app/model/dr_model_traced.pt."
            )
        self.model = torch.jit.load(str(path), map_location=self.device)
        self.model.eval()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _to_tensor(self, rgb_img: np.ndarray) -> torch.Tensor:
        img = rgb_img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(self, bgr_img: np.ndarray) -> PredictionResult:
        """Run full pipeline: preprocess -> model -> structured result.

        Args:
            bgr_img: raw image as read by cv2.imread (BGR channel order).
        """
        processed = ben_graham_preprocess(bgr_img, target_size=self.img_size)
        tensor = self._to_tensor(processed)

        with torch.no_grad():
            raw_output = self.model(tensor)
            raw_score = float(raw_output.squeeze().cpu().numpy())

        grade = int(np.clip(round(raw_score), 0, 4))
        confidence = _confidence_from_raw_score(raw_score)
        prob_bars = _soft_probability_bars(raw_score)
        needs_review = confidence < UNCERTAINTY_THRESHOLD

        return PredictionResult(
            grade=grade,
            grade_label=GRADE_LABELS[grade],
            raw_score=raw_score,
            confidence=confidence,
            probability_bars=prob_bars,
            needs_manual_review=needs_review,
            clinical_recommendation=CLINICAL_RECOMMENDATIONS[grade],
            dr_present=grade > 0,
        )

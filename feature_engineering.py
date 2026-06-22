"""Deterministic handcrafted features for retinal fundus images.

These are candidate/summary signals, not clinically validated lesion masks.
Extract them at high resolution (default 512 px), cache them once, and fit
normalization statistics on the training split only.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

EPS = 1e-6


def _crop_fundus(img: np.ndarray, tol: int = 7) -> np.ndarray:
    if img is None or img.size == 0:
        raise ValueError("Empty image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    foreground = gray > tol
    if not foreground.any():
        return img
    rows = np.where(foreground.any(axis=1))[0]
    cols = np.where(foreground.any(axis=0))[0]
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _retina_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask.astype(bool)
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def _stats(values: np.ndarray) -> Tuple[float, float, float, float, float]:
    values = values.astype(np.float32)
    mean, std = float(values.mean()), float(values.std())
    skew = float(np.mean(((values - mean) / (std + EPS)) ** 3))
    p10, p90 = np.percentile(values, [10, 90])
    return mean, std, skew, float(p10), float(p90)


def _entropy(values: np.ndarray) -> float:
    hist = cv2.calcHist([values.astype(np.uint8)], [0], None, [64], [0, 256]).ravel()
    p = hist / (hist.sum() + EPS)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _component_features(binary: np.ndarray, low: int, high: int) -> Tuple[int, float, float]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if n <= 1:
        return 0, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas[(areas >= low) & (areas <= high)]
    if not keep.size:
        return 0, 0.0, 0.0
    return int(keep.size), float(keep.sum()), float(keep.mean())


def _line_kernel(length: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), np.uint8)
    center = length // 2
    radians = np.deg2rad(angle)
    dx, dy = int(np.cos(radians) * center), int(np.sin(radians) * center)
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)
    return kernel


def _vessel_mask(green: np.ndarray, retina: np.ndarray) -> np.ndarray:
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(green)
    response = np.zeros_like(enhanced)
    for length in (9, 15):
        for angle in (0, 30, 60, 90, 120, 150):
            blackhat = cv2.morphologyEx(
                enhanced, cv2.MORPH_BLACKHAT, _line_kernel(length, angle)
            )
            response = np.maximum(response, blackhat)
    threshold = np.percentile(response[retina], 82)
    vessels = ((response >= threshold) & retina).astype(np.uint8)
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return vessels.astype(bool)


def _optic_disc_exclusion(rgb: np.ndarray, retina: np.ndarray) -> np.ndarray:
    lightness = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
    score = cv2.GaussianBlur(lightness, (0, 0), 13)
    score[~retina] = 0
    _, _, _, max_loc = cv2.minMaxLoc(score)
    exclusion = np.zeros(retina.shape, np.uint8)
    cv2.circle(exclusion, max_loc, max(12, rgb.shape[0] // 14), 1, -1)
    return exclusion.astype(bool)


def _lbp_histogram(gray: np.ndarray, retina: np.ndarray, bins: int = 16) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    lbp = np.zeros_like(center, np.uint8)
    neighbors = (
        gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:],
        gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2],
    )
    for bit, neighbor in enumerate(neighbors):
        lbp |= (neighbor >= center).astype(np.uint8) << bit
    hist, _ = np.histogram(lbp[retina[1:-1, 1:-1]], bins=bins, range=(0, 256))
    return hist.astype(np.float32) / (hist.sum() + EPS)


@dataclass(frozen=True)
class FundusFeatureExtractor:
    feature_size: int = 512

    def extract_named(self, bgr_img: np.ndarray) -> Dict[str, float]:
        cropped = _crop_fundus(bgr_img)
        interpolation = cv2.INTER_AREA if max(cropped.shape[:2]) > self.feature_size else cv2.INTER_CUBIC
        bgr = cv2.resize(cropped, (self.feature_size, self.feature_size), interpolation=interpolation)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        retina = _retina_mask(bgr)
        if retina.sum() < 0.25 * retina.size:
            raise ValueError("Insufficient fundus field")
        area = float(retina.sum())
        features: Dict[str, float] = {}

        features["quality_log_laplacian_var"] = float(
            np.log1p(cv2.Laplacian(gray, cv2.CV_32F)[retina].var())
        )
        features["quality_entropy"] = _entropy(gray[retina])
        features["quality_fov_ratio"] = area / retina.size
        ys, xs = np.where(retina)
        features["quality_center_offset"] = float(
            np.hypot(xs.mean() - self.feature_size / 2, ys.mean() - self.feature_size / 2)
            / self.feature_size
        )
        illumination = cv2.GaussianBlur(gray, (0, 0), self.feature_size / 16)
        features["quality_illumination_cv"] = float(
            illumination[retina].std() / (illumination[retina].mean() + EPS)
        )

        for channel, name in enumerate(("red", "green", "blue")):
            values = _stats(rgb[:, :, channel][retina])
            for suffix, value in zip(("mean", "std", "skew", "p10", "p90"), values):
                features[f"color_{name}_{suffix}"] = value if suffix == "skew" else value / 255.0

        green = rgb[:, :, 1]
        enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(green)
        vessels = _vessel_mask(green, retina)
        features["vessel_density"] = float(vessels.sum() / area)
        features["vessel_component_density"] = float(
            max(cv2.connectedComponents(vessels.astype(np.uint8), 8)[0] - 1, 0) / area
        )

        small_response = cv2.morphologyEx(
            enhanced, cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        )
        dark = (small_response >= np.percentile(small_response[retina], 94)) & retina & ~vessels
        dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        count, candidate_area, mean_area = _component_features(dark, 3, 90)
        features["dark_small_count_log"] = float(np.log1p(count))
        features["dark_small_area_ratio"] = candidate_area / area
        features["dark_small_mean_area"] = mean_area / area

        large_response = cv2.morphologyEx(
            enhanced, cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
        )
        large_dark = (large_response >= np.percentile(large_response[retina], 96)) & retina & ~vessels
        count, candidate_area, _ = _component_features(large_dark, 60, 3500)
        features["dark_large_count_log"] = float(np.log1p(count))
        features["dark_large_area_ratio"] = candidate_area / area

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        top_hat = cv2.morphologyEx(
            enhanced, cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        )
        yellowish = (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 45) & (hsv[:, :, 1] >= 25)
        bright = (
            (top_hat >= np.percentile(top_hat[retina], 94))
            & yellowish & retina & ~_optic_disc_exclusion(rgb, retina)
        )
        count, candidate_area, _ = _component_features(bright, 4, 1800)
        features["bright_candidate_count_log"] = float(np.log1p(count))
        features["bright_candidate_area_ratio"] = candidate_area / area

        for i, value in enumerate(_lbp_histogram(gray, retina)):
            features[f"lbp_{i:02d}"] = float(value)
        return features

    def extract(self, bgr_img: np.ndarray) -> Tuple[np.ndarray, Tuple[str, ...]]:
        named = self.extract_named(bgr_img)
        names = tuple(named)
        return np.asarray([named[name] for name in names], np.float32), names

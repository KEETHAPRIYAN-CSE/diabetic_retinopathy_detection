"""
Ben Graham preprocessing for retinal fundus images.

This MUST stay identical to the preprocessing used in the training notebook
(notebook/train_dr_model.ipynb, Section 3). If this drifts from what the model
was trained on, predictions will be silently wrong — the model has learned
features on Ben-Graham-processed images, not raw fundus photos.
"""

import cv2
import numpy as np


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop away the black border around the fundus image."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray_img > tol
    if mask.sum() == 0:
        return img
    check_shape = img[np.ix_(mask.any(1), mask.any(0))].shape[0]
    if check_shape == 0:
        return img
    img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    return np.stack([img1, img2, img3], axis=-1)


def ben_graham_preprocess(img: np.ndarray, sigma_x: int = 10, target_size: int = 224) -> np.ndarray:
    """Ben Graham preprocessing: crop, subtract local average color, clip to circle.

    Args:
        img: BGR numpy array (as read by cv2.imread).
        sigma_x: Gaussian blur sigma used for local-average-color subtraction.
        target_size: output image is target_size x target_size.

    Returns:
        RGB numpy array, target_size x target_size, uint8.
    """
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (target_size, target_size))

    # Subtract local average color (removes per-image lighting/camera bias)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma_x)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)

    # Circular mask to zero out corners (no retinal tissue there)
    mask = np.zeros(img.shape, dtype=np.uint8)
    center = (target_size // 2, target_size // 2)
    radius = int(target_size * 0.46)
    cv2.circle(mask, center, radius, (1, 1, 1), -1)
    img = img * mask + 128 * (1 - mask)
    img = img.astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

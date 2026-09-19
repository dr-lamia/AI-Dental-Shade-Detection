"""Session-level color calibration for rendered clinical photographs.

This module supports a pragmatic affine correction in CIE XYZ using a photographed
multi-patch reference target. It is intended for research QC of JPEG/sRGB images.

Important:
- A single neutral patch can correct white balance, but cannot fully characterize
  camera color response.
- Prefer >= 18 well-exposed reference patches spanning the tooth-color region and
  broader color space.
- Fit a separate matrix for each locked camera/flash/white-balance session.
- The correction must be frozen before evaluating the Rayplicker validation set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color

from colorimetry import LabResult, pil_to_rgb01, delta_e00


REQUIRED_PATCH_COLUMNS = ["patch", "x0", "y0", "x1", "y1", "L", "a", "b"]


@dataclass(frozen=True)
class CalibrationFit:
    matrix: np.ndarray
    qc: pd.DataFrame
    mean_pre_delta_e00: float
    mean_post_delta_e00: float
    median_post_delta_e00: float
    max_post_delta_e00: float


def _validate_patch_table(spec: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_PATCH_COLUMNS if c not in spec.columns]
    if missing:
        raise ValueError("Missing calibration columns: " + ", ".join(missing))

    out = spec.copy()
    numeric = ["x0", "y0", "x1", "y1", "L", "a", "b"]
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=numeric)

    if len(out) < 6:
        raise ValueError("At least 6 valid patches are required; >=18 is preferred.")

    if ((out[["x0", "y0", "x1", "y1"]] < 0).any().any()
            or (out[["x0", "y0", "x1", "y1"]] > 1).any().any()):
        raise ValueError("Patch coordinates must be normalized to the range 0–1.")

    if (out["x1"] <= out["x0"]).any() or (out["y1"] <= out["y0"]).any():
        raise ValueError("Each patch must have x1>x0 and y1>y0.")

    return out


def _patch_median_xyz(image: Image.Image, row: pd.Series) -> np.ndarray:
    rgb = pil_to_rgb01(image)
    h, w = rgb.shape[:2]

    x0 = max(0, min(w - 1, int(round(float(row["x0"]) * w))))
    y0 = max(0, min(h - 1, int(round(float(row["y0"]) * h))))
    x1 = max(x0 + 1, min(w, int(round(float(row["x1"]) * w))))
    y1 = max(y0 + 1, min(h, int(round(float(row["y1"]) * h))))

    patch_rgb = rgb[y0:y1, x0:x1]
    patch_xyz = color.rgb2xyz(patch_rgb)
    return np.median(patch_xyz.reshape(-1, 3), axis=0)


def fit_xyz_affine(image: Image.Image, patch_spec: pd.DataFrame) -> CalibrationFit:
    """Fit observed JPEG XYZ -> reference XYZ with an affine 4x3 matrix."""
    spec = _validate_patch_table(patch_spec)

    observed_xyz = np.vstack([_patch_median_xyz(image, row) for _, row in spec.iterrows()])
    reference_lab = spec[["L", "a", "b"]].to_numpy(dtype=float)
    reference_xyz = color.lab2xyz(reference_lab.reshape(-1, 1, 3)).reshape(-1, 3)

    design = np.column_stack([observed_xyz, np.ones(len(observed_xyz))])
    matrix, *_ = np.linalg.lstsq(design, reference_xyz, rcond=None)

    predicted_xyz = design @ matrix
    observed_lab = color.xyz2lab(observed_xyz.reshape(-1, 1, 3)).reshape(-1, 3)
    predicted_lab = color.xyz2lab(predicted_xyz.reshape(-1, 1, 3)).reshape(-1, 3)

    pre = np.array([delta_e00(a, b) for a, b in zip(observed_lab, reference_lab)], dtype=float)
    post = np.array([delta_e00(a, b) for a, b in zip(predicted_lab, reference_lab)], dtype=float)

    qc = spec[["patch", "L", "a", "b"]].copy()
    qc["observed_L"] = observed_lab[:, 0]
    qc["observed_a"] = observed_lab[:, 1]
    qc["observed_b"] = observed_lab[:, 2]
    qc["corrected_L"] = predicted_lab[:, 0]
    qc["corrected_a"] = predicted_lab[:, 1]
    qc["corrected_b"] = predicted_lab[:, 2]
    qc["pre_delta_e00"] = pre
    qc["post_delta_e00"] = post

    return CalibrationFit(
        matrix=matrix,
        qc=qc,
        mean_pre_delta_e00=float(pre.mean()),
        mean_post_delta_e00=float(post.mean()),
        median_post_delta_e00=float(np.median(post)),
        max_post_delta_e00=float(post.max()),
    )


def matrix_to_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 3):
        raise ValueError("Calibration matrix must have shape 4x3.")
    return pd.DataFrame(
        m,
        index=["X_in", "Y_in", "Z_in", "bias"],
        columns=["X_out", "Y_out", "Z_out"],
    ).reset_index(names="component")


def dataframe_to_matrix(df: pd.DataFrame) -> np.ndarray:
    needed = ["component", "X_out", "Y_out", "Z_out"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("Calibration matrix CSV missing: " + ", ".join(missing))
    work = df.set_index("component").loc[["X_in", "Y_in", "Z_in", "bias"]]
    return work[["X_out", "Y_out", "Z_out"]].to_numpy(dtype=float)


def calibrated_lab_image(image: Image.Image, matrix: np.ndarray) -> np.ndarray:
    rgb = pil_to_rgb01(image)
    xyz = color.rgb2xyz(rgb)
    flat = xyz.reshape(-1, 3)
    design = np.column_stack([flat, np.ones(len(flat))])
    corrected_xyz = design @ np.asarray(matrix, dtype=float)
    corrected_xyz = np.clip(corrected_xyz, 0.0, None)
    lab = color.xyz2lab(corrected_xyz.reshape(xyz.shape))
    return lab


def robust_lab_calibrated(
    image: Image.Image,
    matrix: np.ndarray,
    trim_percent: float = 5.0,
    min_L: float = 15.0,
    max_L: float = 97.0,
) -> LabResult:
    lab = calibrated_lab_image(image, matrix).reshape(-1, 3)
    mask = (lab[:, 0] >= min_L) & (lab[:, 0] <= max_L)
    usable = lab[mask]
    if usable.shape[0] < 25:
        usable = lab

    trim = float(np.clip(trim_percent, 0.0, 20.0))
    if trim > 0 and usable.shape[0] >= 50:
        lo = np.percentile(usable, trim, axis=0)
        hi = np.percentile(usable, 100.0 - trim, axis=0)
        keep = np.all((usable >= lo) & (usable <= hi), axis=1)
        if keep.sum() >= 25:
            usable = usable[keep]

    mean_lab = usable.mean(axis=0)
    return LabResult(
        L=float(mean_lab[0]),
        a=float(mean_lab[1]),
        b=float(mean_lab[2]),
        n_pixels=int(usable.shape[0]),
        method=f"session_calibrated_trimmed_mean_{trim:g}pct",
    )

"""Colorimetry utilities for the research-validation workflow.

This module intentionally avoids hard-coded VITA reference values. Shade-guide
LAB coordinates depend on the physical guide, device, illuminant, observer, and
measurement geometry. For publishable shade matching, provide a measured
reference table collected under the same standardized setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from PIL import Image
from skimage import color


@dataclass(frozen=True)
class LabResult:
    L: float
    a: float
    b: float
    n_pixels: int
    method: str

    def as_array(self) -> np.ndarray:
        return np.array([self.L, self.a, self.b], dtype=float)


def pil_to_rgb01(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return np.clip(arr, 0.0, 1.0)


def crop_fraction(
    image: Image.Image,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> Image.Image:
    """Crop by normalized image coordinates in [0, 1]."""
    w, h = image.size
    left = int(np.clip(x0, 0, 1) * w)
    top = int(np.clip(y0, 0, 1) * h)
    right = int(np.clip(x1, 0, 1) * w)
    bottom = int(np.clip(y1, 0, 1) * h)
    if right <= left or bottom <= top:
        raise ValueError("ROI has zero or negative area.")
    return image.crop((left, top, right, bottom))


def center_crop(image: Image.Image, fraction: float = 0.70) -> Image.Image:
    fraction = float(np.clip(fraction, 0.10, 1.00))
    margin = (1.0 - fraction) / 2.0
    return crop_fraction(image, margin, margin, 1.0 - margin, 1.0 - margin)


def neutral_reference_balance(
    image: Image.Image,
    patch_box: tuple[float, float, float, float],
) -> Image.Image:
    """Simple neutral-patch channel balancing.

    This is a practical normalization aid, not a substitute for a full
    color-calibration profile. Use only when a known neutral reference patch is
    present in the same image.
    """
    rgb = pil_to_rgb01(image)
    patch = pil_to_rgb01(crop_fraction(image, *patch_box))
    means = patch.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    scale = np.divide(target, means, out=np.ones_like(means), where=means > 1e-8)
    balanced = np.clip(rgb * scale[None, None, :], 0.0, 1.0)
    return Image.fromarray(np.round(balanced * 255).astype(np.uint8), mode="RGB")


def robust_lab(
    image: Image.Image,
    trim_percent: float = 5.0,
    min_L: float = 15.0,
    max_L: float = 97.0,
) -> LabResult:
    """Estimate CIELAB using a trimmed mean after excluding very dark/specular pixels."""
    rgb = pil_to_rgb01(image)
    lab = color.rgb2lab(rgb)
    flat = lab.reshape(-1, 3)

    mask = (flat[:, 0] >= min_L) & (flat[:, 0] <= max_L)
    usable = flat[mask]
    if usable.shape[0] < 25:
        usable = flat

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
        method=f"trimmed_mean_{trim:g}pct",
    )


def delta_e00(lab1: Iterable[float], lab2: Iterable[float]) -> float:
    a = np.asarray(list(lab1), dtype=float).reshape(1, 1, 3)
    b = np.asarray(list(lab2), dtype=float).reshape(1, 1, 3)
    return float(color.deltaE_ciede2000(a, b)[0, 0])


def nearest_shades(
    lab: Iterable[float],
    reference_rows: Iterable[dict],
    top_k: int = 3,
) -> list[dict]:
    """Return nearest shade references ranked by CIEDE2000.

    Each reference row must contain: shade, L, a, b.
    """
    query = np.asarray(list(lab), dtype=float)
    ranked = []
    for row in reference_rows:
        ref = np.array([float(row["L"]), float(row["a"]), float(row["b"])])
        ranked.append(
            {
                "shade": str(row["shade"]),
                "L": float(row["L"]),
                "a": float(row["a"]),
                "b": float(row["b"]),
                "delta_e00": delta_e00(query, ref),
            }
        )
    ranked.sort(key=lambda x: x["delta_e00"])
    return ranked[: max(1, int(top_k))]


def thirds_analysis(
    image: Image.Image,
    incisal_at_top: bool = False,
    trim_percent: float = 5.0,
) -> dict[str, LabResult]:
    """Analyze cervical, middle, and incisal thirds of an already cropped tooth ROI."""
    w, h = image.size
    cuts = [0, h // 3, (2 * h) // 3, h]
    parts = [
        image.crop((0, cuts[0], w, cuts[1])),
        image.crop((0, cuts[1], w, cuts[2])),
        image.crop((0, cuts[2], w, cuts[3])),
    ]
    names = ["incisal", "middle", "cervical"] if incisal_at_top else ["cervical", "middle", "incisal"]
    return {
        name: robust_lab(part, trim_percent=trim_percent)
        for name, part in zip(names, parts)
    }

"""Deterministic shade engine for the upgraded Streamlit app.

No tooth-shade classifier is trained here. The pipeline is:
photograph -> optional color calibration -> CIELAB -> CIEDE2000 ->
nearest measured VITA 3D-Master reference -> regional shade map.

True optical polarization must happen at image acquisition. For ordinary clinical
photographs, this module can only suppress likely specular highlights; it cannot
reconstruct a genuinely cross-polarized image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color


@dataclass(frozen=True)
class LabResult:
    L: float
    a: float
    b: float
    n_pixels: int
    excluded_fraction: float

    def array(self) -> np.ndarray:
        return np.array([self.L, self.a, self.b], dtype=float)


def pil_to_rgb01(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def crop_fraction(image: Image.Image, x0: float, y0: float, x1: float, y1: float) -> Image.Image:
    w, h = image.size
    left = int(np.clip(x0, 0, 1) * w)
    top = int(np.clip(y0, 0, 1) * h)
    right = int(np.clip(x1, 0, 1) * w)
    bottom = int(np.clip(y1, 0, 1) * h)
    if right <= left or bottom <= top:
        raise ValueError("ROI has zero or negative area.")
    return image.crop((left, top, right, bottom))


def neutral_reference_balance(
    image: Image.Image,
    patch_box: tuple[float, float, float, float],
) -> Image.Image:
    """Simple neutral-patch RGB balance. This is not a full color profile."""
    rgb = pil_to_rgb01(image)
    patch = pil_to_rgb01(crop_fraction(image, *patch_box))
    means = patch.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    scale = np.divide(target, means, out=np.ones_like(means), where=means > 1e-8)
    corrected = np.clip(rgb * scale[None, None, :], 0.0, 1.0)
    return Image.fromarray(np.round(corrected * 255).astype(np.uint8), mode="RGB")


def dataframe_to_matrix(df: pd.DataFrame) -> np.ndarray:
    required = ["component", "X_out", "Y_out", "Z_out"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Calibration matrix CSV missing: " + ", ".join(missing))
    work = df.set_index("component").loc[["X_in", "Y_in", "Z_in", "bias"]]
    return work[["X_out", "Y_out", "Z_out"]].to_numpy(dtype=float)


def matrix_to_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 3):
        raise ValueError("Calibration matrix must be 4x3.")
    return pd.DataFrame(
        matrix,
        index=["X_in", "Y_in", "Z_in", "bias"],
        columns=["X_out", "Y_out", "Z_out"],
    ).reset_index(names="component")


def fit_xyz_affine(image: Image.Image, spec: pd.DataFrame):
    """Fit a 4x3 affine XYZ correction from a photographed color target."""
    required = ["patch", "x0", "y0", "x1", "y1", "L", "a", "b"]
    missing = [c for c in required if c not in spec.columns]
    if missing:
        raise ValueError("Patch CSV missing: " + ", ".join(missing))

    work = spec.copy()
    for c in ["x0", "y0", "x1", "y1", "L", "a", "b"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=required[1:])
    if len(work) < 6:
        raise ValueError("At least 6 valid target patches are required; >=18 is preferred.")

    rgb = pil_to_rgb01(image)
    h, w = rgb.shape[:2]
    obs_xyz, ref_lab = [], []

    for _, row in work.iterrows():
        vals = [float(row[c]) for c in ["x0", "y0", "x1", "y1"]]
        if any(v < 0 or v > 1 for v in vals):
            raise ValueError("Patch coordinates must be normalized to 0–1.")
        x0, y0, x1, y1 = vals
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Patch boxes require x1>x0 and y1>y0.")

        xa, ya = int(x0*w), int(y0*h)
        xb, yb = max(xa+1, int(x1*w)), max(ya+1, int(y1*h))
        patch = rgb[ya:yb, xa:xb]
        xyz = color.rgb2xyz(patch).reshape(-1, 3)
        obs_xyz.append(np.median(xyz, axis=0))
        ref_lab.append([float(row["L"]), float(row["a"]), float(row["b"])])

    obs_xyz = np.vstack(obs_xyz)
    ref_lab = np.asarray(ref_lab, dtype=float)
    ref_xyz = color.lab2xyz(ref_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    design = np.column_stack([obs_xyz, np.ones(len(obs_xyz))])
    matrix, *_ = np.linalg.lstsq(design, ref_xyz, rcond=None)

    pred_xyz = np.clip(design @ matrix, 0.0, None)
    obs_lab = color.xyz2lab(obs_xyz.reshape(-1,1,3)).reshape(-1,3)
    pred_lab = color.xyz2lab(pred_xyz.reshape(-1,1,3)).reshape(-1,3)

    qc = work[["patch","L","a","b"]].copy()
    qc["pre_delta_e00"] = [delta_e00(a,b) for a,b in zip(obs_lab, ref_lab)]
    qc["post_delta_e00"] = [delta_e00(a,b) for a,b in zip(pred_lab, ref_lab)]
    return matrix, qc


def calibrated_lab_image(image: Image.Image, matrix: np.ndarray | None = None) -> np.ndarray:
    rgb = pil_to_rgb01(image)
    if matrix is None:
        return color.rgb2lab(rgb)

    xyz = color.rgb2xyz(rgb)
    flat = xyz.reshape(-1,3)
    design = np.column_stack([flat, np.ones(len(flat))])
    corrected_xyz = np.clip(design @ np.asarray(matrix, dtype=float), 0.0, None)
    return color.xyz2lab(corrected_xyz.reshape(xyz.shape))


def valid_tooth_mask(
    image: Image.Image,
    lab_img: np.ndarray,
    cross_polarized: bool = False,
) -> np.ndarray:
    """Exclude dark pixels and likely glare.

    With true cross-polarized capture, only extreme highlights are removed.
    With ordinary photography, high-value low-saturation pixels are also removed.
    """
    rgb = pil_to_rgb01(image)
    hsv = color.rgb2hsv(rgb)
    L = lab_img[...,0]
    mask = (L >= 15.0) & (L <= 97.5)

    if cross_polarized:
        glare = (hsv[...,2] > 0.985) & (hsv[...,1] < 0.08)
    else:
        glare = (hsv[...,2] > 0.92) & (hsv[...,1] < 0.22)

    return mask & (~glare)


def robust_lab(
    image: Image.Image,
    matrix: np.ndarray | None = None,
    cross_polarized: bool = False,
    trim_percent: float = 5.0,
) -> LabResult:
    lab_img = calibrated_lab_image(image, matrix)
    mask = valid_tooth_mask(image, lab_img, cross_polarized=cross_polarized)
    flat = lab_img[mask]
    total = lab_img.shape[0] * lab_img.shape[1]

    if flat.shape[0] < 25:
        flat = lab_img.reshape(-1,3)

    trim = float(np.clip(trim_percent, 0.0, 20.0))
    if trim > 0 and len(flat) >= 50:
        lo = np.percentile(flat, trim, axis=0)
        hi = np.percentile(flat, 100-trim, axis=0)
        keep = np.all((flat >= lo) & (flat <= hi), axis=1)
        if keep.sum() >= 25:
            flat = flat[keep]

    mean = flat.mean(axis=0)
    return LabResult(
        L=float(mean[0]), a=float(mean[1]), b=float(mean[2]),
        n_pixels=int(len(flat)),
        excluded_fraction=float(1.0 - min(1.0, len(flat)/max(1,total))),
    )


def delta_e00(lab1: Iterable[float], lab2: Iterable[float]) -> float:
    a = np.asarray(list(lab1), dtype=float).reshape(1,1,3)
    b = np.asarray(list(lab2), dtype=float).reshape(1,1,3)
    return float(color.deltaE_ciede2000(a,b)[0,0])


def validate_reference_table(df: pd.DataFrame) -> pd.DataFrame:
    required = ["shade","L","a","b"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Reference CSV must contain: shade, L, a, b")
    out = df.copy()
    out["shade"] = out["shade"].astype(str).str.strip()
    for c in ["L","a","b"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=required).drop_duplicates(subset=["shade"], keep="last")
    if out.empty:
        raise ValueError("No valid shade references were found.")
    return out


def nearest_shades(lab: Iterable[float], refs: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    refs = validate_reference_table(refs)
    q = np.asarray(list(lab), dtype=float)
    rows = []
    for _, r in refs.iterrows():
        ref = [float(r["L"]), float(r["a"]), float(r["b"])]
        rows.append({
            "shade": str(r["shade"]),
            "L": ref[0], "a": ref[1], "b": ref[2],
            "delta_e00": delta_e00(q, ref),
        })
    return pd.DataFrame(rows).sort_values("delta_e00").head(int(top_k)).reset_index(drop=True)


def build_reference_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Build an exploratory deterministic reference table from Rayplicker observations.

    This is NOT a trained classifier. Do not evaluate categorical accuracy on the same
    observations used to build these centroids.
    """
    candidates = [
        ("ray_shade","ray_L","ray_a","ray_b"),
        ("shade","L","a","b"),
        ("Shade","L","a","b"),
    ]
    cols = None
    for cand in candidates:
        if all(c in df.columns for c in cand):
            cols = cand
            break
    if cols is None:
        raise ValueError(
            "Need either ray_shade,ray_L,ray_a,ray_b or shade,L,a,b columns."
        )
    s,L,a,b = cols
    work = df[[s,L,a,b]].copy()
    work.columns = ["shade","L","a","b"]
    work["shade"] = work["shade"].astype(str).str.strip()
    for c in ["L","a","b"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()
    out = (
        work.groupby("shade", as_index=False)
        .agg(L=("L","median"), a=("a","median"), b=("b","median"), n=("shade","size"))
        .sort_values("shade")
    )
    return out


def region_grid(mode: str):
    if mode == "3 zones":
        return 3, 1
    if mode == "9 zones":
        return 3, 3
    if mode == "Detailed":
        return 8, 8
    raise ValueError("Unknown map mode.")


def shade_map_table(
    image: Image.Image,
    refs: pd.DataFrame,
    mode: str,
    matrix: np.ndarray | None = None,
    cross_polarized: bool = False,
    incisal_at_top: bool = False,
) -> pd.DataFrame:
    rows, cols = region_grid(mode)
    w,h = image.size
    output = []

    for rr in range(rows):
        for cc in range(cols):
            x0, x1 = cc/cols, (cc+1)/cols
            y0, y1 = rr/rows, (rr+1)/rows
            cell = crop_fraction(image,x0,y0,x1,y1)
            result = robust_lab(
                cell, matrix=matrix, cross_polarized=cross_polarized, trim_percent=5.0
            )
            best = nearest_shades(result.array(), refs, top_k=1).iloc[0]

            if rows == 3:
                names = ["incisal","middle","cervical"] if incisal_at_top else ["cervical","middle","incisal"]
                region = names[rr] if cols == 1 else f"{names[rr]}-{cc+1}"
            else:
                region = f"R{rr+1}C{cc+1}"

            output.append({
                "row": rr, "col": cc, "region": region,
                "L": result.L, "a": result.a, "b": result.b,
                "shade": best["shade"], "shade_delta_e00": float(best["delta_e00"]),
                "ref_L": float(best["L"]), "ref_a": float(best["a"]), "ref_b": float(best["b"]),
            })

    return pd.DataFrame(output)


def lab_to_rgb01(lab: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(lab), dtype=float).reshape(1,1,3)
    return np.clip(color.lab2rgb(arr)[0,0], 0.0, 1.0)

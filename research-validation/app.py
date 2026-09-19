from __future__ import annotations

import io
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from colorimetry import (
    center_crop,
    crop_fraction,
    delta_e00,
    nearest_shades,
    neutral_reference_balance,
    robust_lab,
    thirds_analysis,
)
from calibration import (
    dataframe_to_matrix,
    fit_xyz_affine,
    matrix_to_dataframe,
    robust_lab_calibrated,
    thirds_analysis_calibrated,
)
from validation_stats import bland_altman, concordance_correlation_coefficient, mae, rmse


ALGORITHM_VERSION = "shade-validation-v1.1.0"
REQUIRED_REFERENCE_COLUMNS = ["patient_id", "tooth_fdi", "timepoint", "ray_L", "ray_a", "ray_b"]


st.set_page_config(page_title="Shade AI Research Validation", page_icon="🦷", layout="wide")
st.title("🦷 Shade AI Research Validation")
st.caption(
    "A research workbench for paired photographic colorimetry versus Rayplicker reference measurements."
)

st.warning(
    "Research use only. This branch does not upload or store the thesis patient dataset in GitHub. "
    "Use de-identified IDs (for example P1/P2) and keep clinical images in approved private storage."
)


def load_shade_reference(uploaded):
    if uploaded is None:
        return []
    df = pd.read_csv(uploaded, comment="#")
    needed = {"shade", "L", "a", "b"}
    if not needed.issubset(df.columns):
        raise ValueError("Shade reference CSV must contain: shade, L, a, b")
    return df.dropna(subset=["shade", "L", "a", "b"]).to_dict("records")


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["patient_id"] = out["patient_id"].astype(str).str.strip().str.upper()
    out["tooth_fdi"] = out["tooth_fdi"].astype(str).str.extract(r"(\d{2})", expand=False)
    out["timepoint"] = out["timepoint"].astype(str).str.strip().str.upper()
    return out


def parse_image_key(filename: str):
    stem = filename.rsplit(".", 1)[0]
    match = re.search(r"(P\d+)[-_ ]+(\d{2})[-_ ]+(T[0-4])", stem, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper(), match.group(2), match.group(3).upper()


def lab_table(result):
    return pd.DataFrame(
        [{"L*": result.L, "a*": result.a, "b*": result.b, "usable pixels": result.n_pixels}]
    )


def plot_bland_altman(reference, estimate, label):
    ba = bland_altman(reference, estimate)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ba["means"], ba["diffs"], alpha=0.75)
    ax.axhline(ba["bias"], linestyle="--")
    ax.axhline(ba["loa_low"], linestyle=":")
    ax.axhline(ba["loa_high"], linestyle=":")
    ax.set_xlabel(f"Mean {label}")
    ax.set_ylabel(f"Photographic − Rayplicker {label}")
    ax.set_title(f"Bland–Altman: {label}")
    return fig, ba


def single_image_page():
    st.header("1. Single-image analysis")
    st.write(
        "Use this page to inspect one photograph, define a tooth ROI, and verify the colorimetry "
        "before locking the batch protocol."
    )

    uploaded = st.file_uploader("Clinical photograph", type=["jpg", "jpeg", "png"], key="single")
    ref_file = st.file_uploader(
        "Optional measured shade-guide reference CSV",
        type=["csv"],
        key="shade_ref_single",
        help="Columns: shade, L, a, b. Use measured values from the study setup; do not use invented generic LAB values.",
    )
    calibration_file = st.file_uploader(
        "Optional session calibration matrix",
        type=["csv"],
        key="cal_matrix_single",
        help="Use a matrix generated on the Session calibration QC page from the same acquisition session.",
    )

    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption=uploaded.name, use_container_width=True)

    st.subheader("Tooth ROI")
    x_range = st.slider("Horizontal ROI (%)", 0, 100, (20, 80), key="sx")
    y_range = st.slider("Vertical ROI (%)", 0, 100, (15, 90), key="sy")
    roi = crop_fraction(
        image,
        x_range[0] / 100,
        y_range[0] / 100,
        x_range[1] / 100,
        y_range[1] / 100,
    )

    use_neutral = st.checkbox(
        "Apply neutral-reference correction",
        value=False,
        help="Only enable when a known neutral gray/white reference patch is visible in the same photograph.",
    )
    working_image = image
    if use_neutral:
        nx = st.slider("Neutral patch horizontal (%)", 0, 100, (0, 15), key="nx")
        ny = st.slider("Neutral patch vertical (%)", 0, 100, (0, 15), key="ny")
        working_image = neutral_reference_balance(
            image,
            (nx[0] / 100, ny[0] / 100, nx[1] / 100, ny[1] / 100),
        )
        roi = crop_fraction(
            working_image,
            x_range[0] / 100,
            y_range[0] / 100,
            x_range[1] / 100,
            y_range[1] / 100,
        )

    trim = st.slider("Trim extreme LAB pixels (%)", 0.0, 15.0, 5.0, 0.5)
    calibration_matrix = None
    if calibration_file is not None:
        try:
            calibration_matrix = dataframe_to_matrix(pd.read_csv(calibration_file))
        except Exception as exc:
            st.error(f"Calibration matrix could not be applied: {exc}")
            return

    result = (
        robust_lab_calibrated(roi, calibration_matrix, trim_percent=trim)
        if calibration_matrix is not None
        else robust_lab(roi, trim_percent=trim)
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(roi, caption="Analyzed tooth ROI", use_container_width=True)
    with c2:
        st.dataframe(lab_table(result), use_container_width=True)
        st.caption(f"Algorithm: {ALGORITHM_VERSION} · {result.method}")

    st.subheader("Regional analysis")
    incisal_top = st.radio(
        "Incisal edge position",
        ["Bottom", "Top"],
        horizontal=True,
        help="Needed to label cervical / middle / incisal thirds correctly.",
    ) == "Top"
    thirds = (
        thirds_analysis_calibrated(
            roi, calibration_matrix, incisal_at_top=incisal_top, trim_percent=trim
        )
        if calibration_matrix is not None
        else thirds_analysis(roi, incisal_at_top=incisal_top, trim_percent=trim)
    )
    thirds_df = pd.DataFrame(
        [
            {"region": name, "L*": r.L, "a*": r.a, "b*": r.b}
            for name, r in thirds.items()
        ]
    )
    st.dataframe(thirds_df, use_container_width=True)

    if ref_file is not None:
        try:
            refs = load_shade_reference(ref_file)
            ranked = nearest_shades([result.L, result.a, result.b], refs, top_k=3)
            st.subheader("Nearest measured shade references")
            st.dataframe(pd.DataFrame(ranked), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))
    else:
        st.info(
            "No shade code is reported without a measured shade-reference library. "
            "This avoids the hard-coded A2 and incomplete 3D-Master table in the original prototype."
        )


def batch_validation_page():
    st.header("2. Paired Rayplicker validation")
    st.write(
        "Upload a de-identified Rayplicker reference CSV plus pre-cropped tooth images. "
        "The app matches images to rows by filename."
    )

    with st.expander("Required naming and CSV schema", expanded=True):
        st.code("Image: P1_11_T0.jpg\nImage: P1_11_T4.jpg")
        st.code(",".join(REQUIRED_REFERENCE_COLUMNS) + ",ray_shade(optional)")

    gt_file = st.file_uploader("Rayplicker reference CSV", type=["csv"], key="gt")
    images = st.file_uploader(
        "Pre-cropped tooth images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_images",
    )
    shade_ref_file = st.file_uploader(
        "Optional measured shade-guide reference CSV",
        type=["csv"],
        key="shade_ref_batch",
    )
    calibration_file = st.file_uploader(
        "Optional session calibration matrix",
        type=["csv"],
        key="cal_matrix_batch",
        help="Apply only when every uploaded image belongs to the same calibrated camera/flash session.",
    )

    center_fraction = st.slider(
        "Central fraction analyzed inside each pre-cropped tooth image",
        0.40,
        1.00,
        0.75,
        0.05,
    )
    trim = st.slider("Batch trim extreme LAB pixels (%)", 0.0, 15.0, 5.0, 0.5, key="batch_trim")

    if gt_file is None or not images:
        return

    try:
        gt = pd.read_csv(gt_file)
    except Exception as exc:
        st.error(f"Could not read reference CSV: {exc}")
        return

    missing = [c for c in REQUIRED_REFERENCE_COLUMNS if c not in gt.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    gt = normalize_key_columns(gt)

    shade_refs = []
    if shade_ref_file is not None:
        try:
            shade_refs = load_shade_reference(shade_ref_file)
        except Exception as exc:
            st.error(str(exc))
            return

    calibration_matrix = None
    if calibration_file is not None:
        try:
            calibration_matrix = dataframe_to_matrix(pd.read_csv(calibration_file))
        except Exception as exc:
            st.error(f"Calibration matrix could not be read: {exc}")
            return

    image_rows = []
    rejected = []
    for uploaded in images:
        key = parse_image_key(uploaded.name)
        if key is None:
            rejected.append(uploaded.name)
            continue
        patient_id, tooth_fdi, timepoint = key
        img = Image.open(uploaded).convert("RGB")
        roi = center_crop(img, center_fraction)
        result = (
            robust_lab_calibrated(roi, calibration_matrix, trim_percent=trim)
            if calibration_matrix is not None
            else robust_lab(roi, trim_percent=trim)
        )

        row = {
            "patient_id": patient_id,
            "tooth_fdi": tooth_fdi,
            "timepoint": timepoint,
            "image_file": uploaded.name,
            "ai_L": result.L,
            "ai_a": result.a,
            "ai_b": result.b,
            "ai_usable_pixels": result.n_pixels,
            "algorithm_version": ALGORITHM_VERSION,
        }
        if shade_refs:
            top3 = nearest_shades([result.L, result.a, result.b], shade_refs, top_k=3)
            row["ai_shade"] = top3[0]["shade"]
            row["ai_shade_delta_e00"] = top3[0]["delta_e00"]
            row["ai_top3_shades"] = "|".join(x["shade"] for x in top3)
        image_rows.append(row)

    if rejected:
        st.warning("Skipped files that did not match the naming rule: " + ", ".join(rejected))

    if not image_rows:
        st.error("No image filenames matched the required naming format.")
        return

    ai = pd.DataFrame(image_rows)
    merged = gt.merge(ai, on=["patient_id", "tooth_fdi", "timepoint"], how="inner")

    if merged.empty:
        st.error("No image keys matched the reference CSV.")
        return

    for c in ["ray_L", "ray_a", "ray_b"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = merged.dropna(subset=["ray_L", "ray_a", "ray_b"])

    merged["delta_e00_ai_vs_ray"] = merged.apply(
        lambda r: delta_e00(
            [r["ai_L"], r["ai_a"], r["ai_b"]],
            [r["ray_L"], r["ray_a"], r["ray_b"]],
        ),
        axis=1,
    )
    merged["error_L"] = merged["ai_L"] - merged["ray_L"]
    merged["error_a"] = merged["ai_a"] - merged["ray_a"]
    merged["error_b"] = merged["ai_b"] - merged["ray_b"]

    st.subheader("Paired results")
    st.dataframe(merged, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paired observations", len(merged))
    c2.metric("Mean ΔE00", f'{merged["delta_e00_ai_vs_ray"].mean():.2f}')
    c3.metric("Median ΔE00", f'{merged["delta_e00_ai_vs_ray"].median():.2f}')
    c4.metric("Max ΔE00", f'{merged["delta_e00_ai_vs_ray"].max():.2f}')

    metric_rows = []
    for label, ref_col, ai_col in [
        ("L*", "ray_L", "ai_L"),
        ("a*", "ray_a", "ai_a"),
        ("b*", "ray_b", "ai_b"),
    ]:
        metric_rows.append(
            {
                "coordinate": label,
                "MAE": mae(merged[ref_col], merged[ai_col]),
                "RMSE": rmse(merged[ref_col], merged[ai_col]),
                "Lin CCC": concordance_correlation_coefficient(merged[ref_col], merged[ai_col]),
            }
        )
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

    st.subheader("Negative-control benchmark")
    patient_ids = sorted(merged["patient_id"].dropna().unique())
    if len(patient_ids) >= 2:
        baseline_rows = []
        for patient_id in patient_ids:
            train = merged[merged["patient_id"] != patient_id]
            test = merged[merged["patient_id"] == patient_id]
            if train.empty or test.empty:
                continue
            mean_ref = train[["ray_L", "ray_a", "ray_b"]].mean().to_numpy(dtype=float)
            for idx, row in test.iterrows():
                baseline_rows.append(
                    {
                        "index": idx,
                        "baseline_delta_e00": delta_e00(
                            mean_ref,
                            [row["ray_L"], row["ray_a"], row["ray_b"]],
                        ),
                    }
                )
        if baseline_rows:
            baseline_df = pd.DataFrame(baseline_rows).set_index("index")
            merged = merged.join(baseline_df)
            ai_mean = float(merged["delta_e00_ai_vs_ray"].mean())
            baseline_mean = float(merged["baseline_delta_e00"].mean())
            b1, b2 = st.columns(2)
            b1.metric("Photographic mean ΔE00", f"{ai_mean:.2f}")
            b2.metric("Other-patient mean-color baseline ΔE00", f"{baseline_mean:.2f}")
            st.caption(
                "The negative control predicts each patient's teeth using only the mean Rayplicker LAB "
                "from the other patient. A useful imaging method should outperform this trivial benchmark."
            )
            if ai_mean >= baseline_mean:
                st.warning(
                    "This batch does not outperform the trivial mean-color benchmark. Do not interpret "
                    "low calibrated error alone as evidence that the photographs contain useful shade signal."
                )
    else:
        st.info("Negative-control benchmarking requires at least two patient IDs.")

    if shade_refs and "ray_shade" in merged.columns:
        valid = merged["ray_shade"].notna()
        if valid.any():
            exact = (
                merged.loc[valid, "ai_shade"].astype(str).str.upper()
                == merged.loc[valid, "ray_shade"].astype(str).str.upper()
            ).mean()
            top3 = merged.loc[valid].apply(
                lambda r: str(r["ray_shade"]).upper()
                in str(r.get("ai_top3_shades", "")).upper().split("|"),
                axis=1,
            ).mean()
            s1, s2 = st.columns(2)
            s1.metric("Exact shade agreement", f"{100*exact:.1f}%")
            s2.metric("Reference shade in AI top-3", f"{100*top3:.1f}%")

    st.subheader("Agreement plots")
    for label, ref_col, ai_col in [
        ("L*", "ray_L", "ai_L"),
        ("a*", "ray_a", "ai_a"),
        ("b*", "ray_b", "ai_b"),
    ]:
        fig, ba = plot_bland_altman(merged[ref_col], merged[ai_col], label)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"{label}: bias={ba['bias']:.2f}, 95% limits of agreement "
            f"{ba['loa_low']:.2f} to {ba['loa_high']:.2f}"
        )

    st.subheader("T0 → T4 longitudinal agreement")
    longitudinal = []
    for (patient_id, tooth_fdi), group in merged.groupby(["patient_id", "tooth_fdi"]):
        by_tp = group.set_index("timepoint")
        if "T0" in by_tp.index and "T4" in by_tp.index:
            t0 = by_tp.loc["T0"]
            t4 = by_tp.loc["T4"]
            if isinstance(t0, pd.DataFrame) or isinstance(t4, pd.DataFrame):
                continue
            ai_change = delta_e00(
                [t0["ai_L"], t0["ai_a"], t0["ai_b"]],
                [t4["ai_L"], t4["ai_a"], t4["ai_b"]],
            )
            ray_change = delta_e00(
                [t0["ray_L"], t0["ray_a"], t0["ray_b"]],
                [t4["ray_L"], t4["ray_a"], t4["ray_b"]],
            )
            longitudinal.append(
                {
                    "patient_id": patient_id,
                    "tooth_fdi": tooth_fdi,
                    "AI ΔE00 T0→T4": ai_change,
                    "Rayplicker ΔE00 T0→T4": ray_change,
                    "absolute difference": abs(ai_change - ray_change),
                }
            )

    if longitudinal:
        long_df = pd.DataFrame(longitudinal)
        st.dataframe(long_df, use_container_width=True)
        st.metric(
            "MAE of longitudinal ΔE00",
            f'{long_df["absolute difference"].mean():.2f}',
        )
    else:
        st.info("No matched T0/T4 tooth pairs were present in the uploaded batch.")

    st.info(
        "Statistical caution: multiple teeth are nested within the same patients. "
        "Treat these as repeated/clustered observations. With the current thesis material, "
        "use agreement-focused descriptive analysis rather than pretending each tooth is an independent patient."
    )

    csv_bytes = merged.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download paired validation results",
        data=csv_bytes,
        file_name="shade_ai_paired_validation_results.csv",
        mime="text/csv",
    )




def calibration_page():
    st.header("3. Session calibration QC")
    st.write(
        "Fit a session-specific color correction from a photographed multi-patch reference target. "
        "Use this before analyzing clinical teeth from the same locked camera/flash/white-balance session."
    )
    st.info(
        "The existing thesis photographs reviewed so far do not show a gray card or multi-patch target. "
        "This page is therefore mainly for prospective/repeat acquisitions or any historical session "
        "where a calibration-target photograph can be recovered."
    )

    target = st.file_uploader(
        "Calibration-target photograph",
        type=["jpg", "jpeg", "png"],
        key="cal_target",
    )
    spec_file = st.file_uploader(
        "Patch specification CSV",
        type=["csv"],
        key="cal_spec",
        help="Columns: patch,x0,y0,x1,y1,L,a,b. Coordinates are normalized 0–1.",
    )

    st.code("patch,x0,y0,x1,y1,L,a,b\nP01,0.10,0.10,0.15,0.15,50.0,0.0,0.0")

    if target is None or spec_file is None:
        return

    image = Image.open(target).convert("RGB")
    st.image(image, caption=target.name, use_container_width=True)

    try:
        spec = pd.read_csv(spec_file)
        fit = fit_xyz_affine(image, spec)
    except Exception as exc:
        st.error(str(exc))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean ΔE00 before", f"{fit.mean_pre_delta_e00:.2f}")
    c2.metric("Mean ΔE00 after", f"{fit.mean_post_delta_e00:.2f}")
    c3.metric("Median ΔE00 after", f"{fit.median_post_delta_e00:.2f}")
    c4.metric("Max ΔE00 after", f"{fit.max_post_delta_e00:.2f}")

    st.dataframe(fit.qc, use_container_width=True)

    if fit.mean_post_delta_e00 >= fit.mean_pre_delta_e00:
        st.warning(
            "Calibration did not improve the target-patch error. Do not use this matrix for validation."
        )
    elif fit.mean_post_delta_e00 > 2.0:
        st.warning(
            "Residual calibration error is still high. Check exposure, glare, patch coordinates, "
            "target reference values, and session consistency before using this matrix."
        )
    else:
        st.success(
            "Target-patch fit improved. Freeze this matrix and verify it on independent check patches "
            "before applying it to the Rayplicker validation set."
        )

    matrix_df = matrix_to_dataframe(fit.matrix)
    st.dataframe(matrix_df, use_container_width=True)
    st.download_button(
        "Download session calibration matrix",
        data=matrix_df.to_csv(index=False).encode("utf-8"),
        file_name="session_calibration_matrix.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download calibration QC table",
        data=fit.qc.to_csv(index=False).encode("utf-8"),
        file_name="session_calibration_qc.csv",
        mime="text/csv",
    )


def protocol_page():
    st.header("4. Frozen research protocol")
    st.markdown(
        """
**Primary reference:** averaged Rayplicker CIELAB measurements.

**Primary endpoint:** CIEDE2000 ΔE00 between photographic estimate and Rayplicker for the same tooth/timepoint.

**Initial cohort:** the currently available de-identified full-veneer table contains 17 teeth. Begin with the post-cementation/T0 photographic set only after image-to-timepoint provenance is verified. Add T4 only when matching 12-month clinical photographs are confirmed.

**Image provenance:** confirm that each clinical photograph was acquired at the same study timepoint as the Rayplicker reference before treating it as a paired observation. Folder names such as "after" are not sufficient by themselves.

**Do not train on these two patients.** The validation cohort must remain independent of model training or parameter tuning.

**Shade-code analysis:** only after a measured VITA reference table is supplied. The old hard-coded A2 function and incomplete seven-shade 3D-Master lookup are intentionally not used.

**TRIOS:** keep as a planned secondary comparator until actual TRIOS shade/LAB outputs are recovered. A TRIOS scan by itself is not a shade-result dataset.

**Sectional veneers:** add only after the original averaged sectional measurements are located and linked to tooth IDs. Do not back-calculate study means from screenshots unless the acquisition identities and averaging procedure are verified.
"""
    )
    st.caption(f"Frozen algorithm label: {ALGORITHM_VERSION}")


page = st.sidebar.radio(
    "Workspace",
    ["Single image", "Paired validation", "Calibration QC", "Protocol"],
)

if page == "Single image":
    single_image_page()
elif page == "Paired validation":
    batch_validation_page()
elif page == "Calibration QC":
    calibration_page()
else:
    protocol_page()

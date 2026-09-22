from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from batch_research_page import render_batch_research_page
from vlm_lab_research_page import render_vlm_lab_research_page
from gpt_vision import (
    ALLOWED_3D_MASTER_SHADES,
    estimate_visual_shade,
    normalize_shade_name,
)
from shade_engine import (
    build_reference_centroids,
    crop_fraction,
    dataframe_to_matrix,
    delta_e00,
    fit_xyz_affine,
    lab_to_rgb01,
    matrix_to_dataframe,
    nearest_shades,
    neutral_reference_balance,
    region_grid,
    robust_lab,
    shade_map_table,
    validate_reference_table,
)

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


APP_VERSION = "ShadeGPT calibrated 3D-Master v2.0"
st.set_page_config(page_title="ShadeGPT – Calibrated Tooth Shade", page_icon="🦷", layout="wide")

st.title("🦷 ShadeGPT – Calibrated Tooth Shade & 3D-Master Mapping")
st.caption(
    "Same Streamlit shade workflow, upgraded to calibration → CIELAB → CIEDE2000 → "
    "VITA 3D-Master mapping. No tooth-shade classifier is trained."
)

with st.sidebar:
    st.markdown(f"**{APP_VERSION}**")
    page = st.radio(
        "Workspace",
        ["Analyze tooth", "Build calibration", "Build shade reference", "Batch research benchmark", "VLM Lab benchmark", "Method"],
    )
    st.divider()
    st.caption("Research/educational use. Clinical deployment requires independent validation.")


def load_reference(uploaded):
    if uploaded is None:
        return None
    return validate_reference_table(pd.read_csv(uploaded))


def load_matrix(uploaded):
    if uploaded is None:
        return None
    return dataframe_to_matrix(pd.read_csv(uploaded))


def make_shade_map_figure(roi: Image.Image, table: pd.DataFrame, mode: str):
    rows, cols = region_grid(mode)
    w, h = roi.size
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(roi)

    for _, r in table.iterrows():
        rr, cc = int(r["row"]), int(r["col"])
        x = cc * w / cols
        y = rr * h / rows
        cw, ch = w / cols, h / rows
        rgb = lab_to_rgb01([r["ref_L"], r["ref_a"], r["ref_b"]])
        rect = Rectangle((x, y), cw, ch, facecolor=rgb, alpha=0.42, edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        if mode != "Detailed" or (rows <= 8 and cols <= 8):
            ax.text(
                x + cw/2,
                y + ch/2,
                str(r["shade"]),
                ha="center",
                va="center",
                fontsize=8 if mode == "Detailed" else 11,
                weight="bold",
                color="black",
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1),
            )

    ax.set_title(f"VITA 3D-Master shade map – {mode}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def generate_pdf_report(
    overall,
    best_shade,
    top3,
    map_table,
    calibration_mode,
    polarization_mode,
    delta_e=None,
    gpt_result=None,
):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    y = 810

    c.setFont("Helvetica-Bold", 16)
    c.drawString(55, y, "ShadeGPT Calibrated Tooth Shade Report")
    y -= 28
    c.setFont("Helvetica", 10)
    c.drawString(55, y, f"Version: {APP_VERSION}")
    y -= 18
    c.drawString(55, y, f"Calibration: {calibration_mode}")
    y -= 18
    c.drawString(55, y, f"Optical input: {polarization_mode}")
    y -= 28

    c.setFont("Helvetica-Bold", 12)
    c.drawString(55, y, "Overall CIELAB")
    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(70, y, f"L* {overall.L:.2f}    a* {overall.a:.2f}    b* {overall.b:.2f}")
    y -= 20

    if best_shade is not None:
        c.drawString(70, y, f"Nearest VITA 3D-Master: {best_shade}")
        y -= 20

    if delta_e is not None:
        c.drawString(70, y, f"Color change vs saved previous image, DeltaE00: {delta_e:.2f}")
        y -= 22

    if top3 is not None and not top3.empty:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(55, y, "Top shade references")
        y -= 18
        c.setFont("Helvetica", 10)
        for _, r in top3.iterrows():
            c.drawString(70, y, f'{r["shade"]}: DeltaE00 {r["delta_e00"]:.2f}')
            y -= 16

    if gpt_result is not None:
        y -= 8
        c.setFont("Helvetica-Bold", 12)
        c.drawString(55, y, "GPT Vision independent visual estimate")
        y -= 18
        c.setFont("Helvetica", 10)
        c.drawString(70, y, f'Predicted shade: {gpt_result.get("predicted_shade", "N/A")}')
        y -= 16
        confidence = gpt_result.get("confidence_percent")
        confidence_text = f"{confidence}%" if confidence is not None else "N/A"
        c.drawString(70, y, f"Model-reported confidence: {confidence_text}")
        y -= 16
        c.drawString(70, y, f'Image quality: {gpt_result.get("image_quality", "N/A")}')
        y -= 16
        c.drawString(70, y, f'Glare: {gpt_result.get("glare", "N/A")}')
        y -= 16
        c.drawString(70, y, f'Blur: {gpt_result.get("blur", "N/A")}')
        y -= 16
        c.drawString(70, y, f'Exposure: {gpt_result.get("exposure", "N/A")}')
        y -= 16
        if gpt_result.get("agreement_with_calibrated") is not None:
            agreement_text = "Yes" if gpt_result["agreement_with_calibrated"] else "No"
            c.drawString(70, y, f"Agreement with calibrated shade: {agreement_text}")
            y -= 16
        if gpt_result.get("gpt_tab_vs_calibrated_delta_e00") is not None:
            c.drawString(
                70,
                y,
                "DeltaE00, GPT-selected VITA tab vs calibrated tooth Lab: "
                f'{gpt_result["gpt_tab_vs_calibrated_delta_e00"]:.2f}',
            )
            y -= 16

    if map_table is not None and not map_table.empty:
        y -= 8
        c.setFont("Helvetica-Bold", 12)
        c.drawString(55, y, "Regional shade map summary")
        y -= 18
        c.setFont("Helvetica", 9)
        for _, r in map_table.head(24).iterrows():
            c.drawString(
                70,
                y,
                f'{r["region"]}: {r["shade"]}  '
                f'(L* {r["L"]:.1f}, a* {r["a"]:.1f}, b* {r["b"]:.1f}, '
                f'DeltaE00-to-reference {r["shade_delta_e00"]:.2f})',
            )
            y -= 14
            if y < 80:
                c.showPage()
                y = 810

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


def gpt_explain(overall, top3, map_table, audience, language, calibration_mode, polarization_mode):
    if OpenAI is None:
        return None, "The OpenAI Python package is not installed."
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None, "Add OPENAI_API_KEY to Streamlit secrets to enable the optional ChatGPT explanation."

    try:
        model = st.secrets.get("OPENAI_MODEL", "gpt-5.6-luna")
    except Exception:
        model = "gpt-5.6-luna"

    shade_text = "No validated shade-reference library supplied."
    if top3 is not None and not top3.empty:
        shade_text = "; ".join(
            f'{r["shade"]} (DeltaE00 {r["delta_e00"]:.2f})'
            for _, r in top3.iterrows()
        )

    regional = "No regional map available."
    if map_table is not None and not map_table.empty:
        counts = map_table["shade"].value_counts().head(5)
        regional = ", ".join(f"{shade}: {count} regions" for shade, count in counts.items())

    prompt = f"""
You are explaining the output of a dental shade-analysis application.
Do not calculate a new shade and do not override the deterministic colorimetry.
Do not diagnose disease or claim clinical-grade accuracy.

Audience: {audience}
Language: {language}
Calibration mode: {calibration_mode}
Input polarization status: {polarization_mode}
Overall CIELAB: L*={overall.L:.2f}, a*={overall.a:.2f}, b*={overall.b:.2f}
Nearest measured VITA 3D-Master references: {shade_text}
Regional distribution: {regional}

Explain the result briefly. State that shade assignment comes from calibrated CIELAB
nearest-neighbor matching with CIEDE2000, not from ChatGPT. Mention any relevant
limitation of the stated calibration/polarization mode.
"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=model, input=prompt)
        return response.output_text, None
    except Exception as exc:
        return None, str(exc)


def analyze_page():
    st.header("Analyze tooth")
    st.write(
        "Upload the clinical photograph, calibrate it first when a valid reference is available, "
        "select the tooth ROI, then compute CIELAB and the VITA 3D-Master map."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        uploaded = st.file_uploader("Clinical photograph", type=["jpg", "jpeg", "png"], key="clinical")
    with col_b:
        ref_upload = st.file_uploader(
            "VITA 3D-Master reference CSV",
            type=["csv"],
            key="vita_ref",
            help="Columns: shade,L,a,b. Prefer measurements of the actual physical guide under a defined setup.",
        )

    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")

    refs = None
    if ref_upload is not None:
        try:
            refs = load_reference(ref_upload)
            st.session_state["vita_refs"] = refs
        except Exception as exc:
            st.error(str(exc))
            return
    elif "vita_refs" in st.session_state:
        refs = st.session_state["vita_refs"]

    st.subheader("1. Optical input")
    polarization_mode = st.radio(
        "How was this photograph acquired?",
        ["Standard clinical photograph", "True cross-polarized photograph"],
        horizontal=True,
    )
    cross_polarized = polarization_mode == "True cross-polarized photograph"
    if not cross_polarized:
        st.caption(
            "The app will exclude likely specular-highlight pixels. This is glare suppression, "
            "not synthetic polarization."
        )

    st.subheader("2. Color calibration before CIELAB")
    calibration_mode = st.radio(
        "Calibration",
        ["None / exploratory", "Neutral patch in this image", "Session calibration matrix"],
        horizontal=True,
    )

    working = image
    matrix = None

    if calibration_mode == "Neutral patch in this image":
        nx = st.slider("Neutral patch horizontal (%)", 0, 100, (2, 12), key="neutral_x")
        ny = st.slider("Neutral patch vertical (%)", 0, 100, (2, 12), key="neutral_y")
        try:
            working = neutral_reference_balance(
                image,
                (nx[0]/100, ny[0]/100, nx[1]/100, ny[1]/100),
            )
        except Exception as exc:
            st.error(str(exc))
            return
        st.warning("A neutral patch corrects channel balance only; it is not a complete camera color profile.")

    elif calibration_mode == "Session calibration matrix":
        matrix_upload = st.file_uploader(
            "Upload session calibration matrix CSV",
            type=["csv"],
            key="matrix_analyze",
        )
        if matrix_upload is not None:
            try:
                matrix = load_matrix(matrix_upload)
            except Exception as exc:
                st.error(str(exc))
                return
        elif "session_matrix" in st.session_state:
            matrix = st.session_state["session_matrix"]
        else:
            st.info("Build or upload a session matrix before analysis.")
            return

    st.subheader("3. Tooth ROI")
    st.image(working, caption="Working image", use_container_width=True)
    x_range = st.slider("Horizontal tooth ROI (%)", 0, 100, (25, 75), key="roi_x")
    y_range = st.slider("Vertical tooth ROI (%)", 0, 100, (15, 90), key="roi_y")

    try:
        roi = crop_fraction(
            working,
            x_range[0]/100,
            y_range[0]/100,
            x_range[1]/100,
            y_range[1]/100,
        )
    except Exception as exc:
        st.error(str(exc))
        return

    trim = st.slider("Trim extreme CIELAB pixels (%)", 0.0, 15.0, 5.0, 0.5)
    overall = robust_lab(
        roi,
        matrix=matrix,
        cross_polarized=cross_polarized,
        trim_percent=trim,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(roi, caption="Analyzed tooth ROI", use_container_width=True)
    with c2:
        st.metric("L*", f"{overall.L:.2f}")
        st.metric("a*", f"{overall.a:.2f}")
        st.metric("b*", f"{overall.b:.2f}")
        st.caption(f"Usable pixels: {overall.n_pixels:,}")

    top3 = None
    best_shade = None
    if refs is not None:
        top3 = nearest_shades(overall.array(), refs, top_k=3)
        best_shade = str(top3.iloc[0]["shade"])
        st.success(f"Nearest VITA 3D-Master shade: **{best_shade}**")
        st.dataframe(
            top3.rename(columns={"delta_e00": "ΔE00 to reference"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "CIELAB is available, but shade naming is disabled until a measured VITA 3D-Master "
            "reference table is supplied. The app will not invent shade-guide LAB values."
        )

    map_table = None
    if refs is not None:
        st.subheader("4. Rayplicker-style regional shade map")
        map_mode = st.radio("Mapping resolution", ["3 zones", "9 zones", "Detailed"], horizontal=True)
        incisal_at_top = st.radio(
            "Incisal edge location in the crop",
            ["Top", "Bottom"],
            horizontal=True,
        ) == "Top"

        map_table = shade_map_table(
            roi,
            refs,
            map_mode,
            matrix=matrix,
            cross_polarized=cross_polarized,
            incisal_at_top=incisal_at_top,
        )
        fig = make_shade_map_figure(roi, map_table, map_mode)
        st.pyplot(fig)
        plt.close(fig)

        if map_mode != "Detailed":
            st.dataframe(
                map_table[["region","L","a","b","shade","shade_delta_e00"]]
                .rename(columns={"shade_delta_e00":"ΔE00 to shade reference"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Detailed map uses dense local blocks; export the table for the full pixel-region record.")
            st.download_button(
                "Download detailed shade map CSV",
                data=map_table.to_csv(index=False).encode("utf-8"),
                file_name="detailed_vita_3d_master_map.csv",
                mime="text/csv",
            )

    st.subheader("5. Color-change comparison")
    delta_prev = None
    if st.button("Save current CIELAB as previous"):
        st.session_state["prev_lab"] = overall.array().tolist()
        st.success("Saved.")

    if st.session_state.get("prev_lab") is not None:
        delta_prev = delta_e00(st.session_state["prev_lab"], overall.array())
        st.metric("ΔE00 vs saved previous image", f"{delta_prev:.2f}")

    st.subheader("6. GPT Vision independent visual estimate")
    st.caption(
        "This is an additional multimodal-AI comparison arm. GPT receives the tooth ROI only; "
        "it does not receive the calibrated CIELAB result, CIEDE2000 result, Rayplicker result, "
        "or predictions from the other models."
    )

    gpt_language = st.selectbox(
        "GPT Vision output language",
        ["English", "Arabic"],
        key="gpt_vision_language",
    )

    roi_key = (
        uploaded.name,
        image.size,
        tuple(x_range),
        tuple(y_range),
        calibration_mode,
        polarization_mode,
    )

    if st.button("Run GPT visual shade estimate", key="run_gpt_visual"):
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            vision_model = st.secrets.get(
                "OPENAI_VISION_MODEL",
                st.secrets.get("OPENAI_MODEL", "gpt-5.6-luna"),
            )
            allowed_shades = (
                refs["shade"].astype(str).tolist()
                if refs is not None
                else ALLOWED_3D_MASTER_SHADES
            )
            with st.spinner("Running independent GPT Vision estimate..."):
                result = estimate_visual_shade(
                    image=roi,
                    api_key=api_key,
                    model=vision_model,
                    allowed_shades=allowed_shades,
                    language=gpt_language,
                )
            st.session_state["gpt_visual_result"] = result
            st.session_state["gpt_visual_roi_key"] = roi_key
        except Exception as exc:
            st.session_state["gpt_visual_result"] = None
            st.session_state["gpt_visual_roi_key"] = None
            st.warning(str(exc))

    gpt_result = None
    if st.session_state.get("gpt_visual_roi_key") == roi_key:
        gpt_result = st.session_state.get("gpt_visual_result")

    if gpt_result is not None:
        gpt_result = dict(gpt_result)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("GPT visual shade", gpt_result["predicted_shade"])
        with c2:
            confidence = gpt_result.get("confidence_percent")
            st.metric(
                "Model-reported confidence",
                f"{confidence}%" if confidence is not None else "N/A",
            )

        st.caption(
            f'Vision model: {gpt_result.get("model", "N/A")} · '
            "Confidence is model-reported and is not a calibrated probability."
        )

        gpt_tab_vs_calibrated_delta_e00 = None
        agreement_with_calibrated = None

        if best_shade is not None:
            agreement_with_calibrated = (
                normalize_shade_name(gpt_result["predicted_shade"])
                == normalize_shade_name(best_shade)
            )
            if agreement_with_calibrated:
                st.success(
                    "GPT visual estimate agrees with the calibrated deterministic shade: "
                    f"**{best_shade}**."
                )
            else:
                st.warning(
                    "GPT visual estimate differs from the calibrated deterministic shade: "
                    f"calibrated **{best_shade}** vs GPT **{gpt_result['predicted_shade']}**."
                )

        if refs is not None:
            refs_for_gpt = refs.copy()
            refs_for_gpt["shade_norm"] = refs_for_gpt["shade"].astype(str).apply(
                normalize_shade_name
            )
            selected = refs_for_gpt.loc[
                refs_for_gpt["shade_norm"]
                == normalize_shade_name(gpt_result["predicted_shade"])
            ]
            if not selected.empty:
                ref_lab = selected.iloc[0][["L", "a", "b"]].to_numpy(dtype=float)
                gpt_tab_vs_calibrated_delta_e00 = delta_e00(
                    ref_lab,
                    overall.array(),
                )
                st.metric(
                    "ΔE00: GPT-selected VITA tab vs calibrated tooth Lab",
                    f"{gpt_tab_vs_calibrated_delta_e00:.2f}",
                )
                st.caption(
                    "This ΔE00 represents the published/measured CIELAB coordinates of the "
                    "VITA tab selected by GPT versus the calibrated tooth CIELAB. It does not "
                    "mean GPT directly measured L*, a*, or b*."
                )

        gpt_result["agreement_with_calibrated"] = agreement_with_calibrated
        gpt_result["gpt_tab_vs_calibrated_delta_e00"] = (
            gpt_tab_vs_calibrated_delta_e00
        )

        gpt_display = pd.DataFrame(
            [
                {
                    "GPT shade": gpt_result["predicted_shade"],
                    "Model-reported confidence (%)": gpt_result.get(
                        "confidence_percent"
                    ),
                    "Image quality": gpt_result.get("image_quality"),
                    "Glare": gpt_result.get("glare"),
                    "Blur": gpt_result.get("blur"),
                    "Exposure": gpt_result.get("exposure"),
                    "Cervical/middle/incisal variation": gpt_result.get(
                        "cervical_middle_incisal_variation"
                    ),
                    "Notes": gpt_result.get("notes"),
                }
            ]
        )
        st.dataframe(gpt_display, use_container_width=True, hide_index=True)

        comparison_row = pd.DataFrame(
            [
                {
                    "calibrated_vita_shade": best_shade,
                    "calibrated_L": overall.L,
                    "calibrated_a": overall.a,
                    "calibrated_b": overall.b,
                    "gpt_model": gpt_result.get("model"),
                    "gpt_visual_shade": gpt_result["predicted_shade"],
                    "gpt_confidence_percent": gpt_result.get(
                        "confidence_percent"
                    ),
                    "gpt_agrees_with_calibrated": agreement_with_calibrated,
                    "gpt_tab_vs_calibrated_delta_e00": (
                        gpt_tab_vs_calibrated_delta_e00
                    ),
                    "image_quality": gpt_result.get("image_quality"),
                    "glare": gpt_result.get("glare"),
                    "blur": gpt_result.get("blur"),
                    "exposure": gpt_result.get("exposure"),
                }
            ]
        )
        st.download_button(
            "Download GPT comparison row",
            data=comparison_row.to_csv(index=False).encode("utf-8"),
            file_name="shadegpt_gpt_vision_comparison.csv",
            mime="text/csv",
        )

        with st.expander("Show raw GPT JSON response"):
            st.code(gpt_result.get("raw_response", ""), language="json")

    st.subheader("7. Optional ChatGPT explanation")
    audience = st.selectbox("Audience", ["Dentist", "Patient"])
    language = st.selectbox("Language", ["English", "Arabic"])
    if st.button("Explain this result with ChatGPT"):
        text, err = gpt_explain(
            overall,
            top3,
            map_table,
            audience,
            language,
            calibration_mode,
            polarization_mode,
        )
        if err:
            st.warning(err)
        else:
            st.write(text)

    st.subheader("8. Report")
    pdf = generate_pdf_report(
        overall,
        best_shade,
        top3,
        map_table,
        calibration_mode,
        polarization_mode,
        delta_prev,
        gpt_result,
    )
    st.download_button(
        "📄 Download PDF report",
        data=pdf,
        file_name="ShadeGPT_calibrated_report.pdf",
        mime="application/pdf",
    )


def calibration_page():
    st.header("Build session calibration")
    st.write(
        "Photograph a multi-patch reference target under the same camera, flash, exposure, "
        "white balance, distance and magnification as the dental photograph."
    )

    target = st.file_uploader("Calibration-target photograph", type=["jpg","jpeg","png"], key="target")
    spec_upload = st.file_uploader(
        "Patch specification CSV",
        type=["csv"],
        key="patch_spec",
        help="Columns: patch,x0,y0,x1,y1,L,a,b using normalized 0–1 image coordinates.",
    )

    st.code("patch,x0,y0,x1,y1,L,a,b\nP01,0.10,0.10,0.15,0.15,50.0,0.0,0.0")

    if target is None or spec_upload is None:
        return

    image = Image.open(target).convert("RGB")
    st.image(image, caption=target.name, use_container_width=True)

    try:
        spec = pd.read_csv(spec_upload)
        matrix, qc = fit_xyz_affine(image, spec)
    except Exception as exc:
        st.error(str(exc))
        return

    pre = float(qc["pre_delta_e00"].mean())
    post = float(qc["post_delta_e00"].mean())
    median_post = float(qc["post_delta_e00"].median())
    max_post = float(qc["post_delta_e00"].max())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean ΔE00 before", f"{pre:.2f}")
    c2.metric("Mean ΔE00 after", f"{post:.2f}")
    c3.metric("Median after", f"{median_post:.2f}")
    c4.metric("Max after", f"{max_post:.2f}")

    st.dataframe(qc, use_container_width=True, hide_index=True)
    if post >= pre:
        st.error("Calibration did not improve the reference-target error. Do not use this matrix.")
        return

    st.session_state["session_matrix"] = matrix
    matrix_df = matrix_to_dataframe(matrix)
    st.success("Calibration matrix stored in this session.")
    st.download_button(
        "Download calibration matrix",
        data=matrix_df.to_csv(index=False).encode("utf-8"),
        file_name="shade_session_calibration_matrix.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download calibration QC",
        data=qc.to_csv(index=False).encode("utf-8"),
        file_name="shade_session_calibration_qc.csv",
        mime="text/csv",
    )


def reference_page():
    st.header("Build VITA 3D-Master reference library")
    st.write(
        "The shade engine does not train a classifier. It maps measured CIELAB values to the "
        "nearest VITA 3D-Master reference using CIEDE2000."
    )

    st.markdown(
        "**Preferred:** measure the physical VITA 3D-Master tabs with the same reference device/setup "
        "and upload one L*, a*, b* value per shade."
    )

    direct = st.file_uploader(
        "Measured shade-guide table (shade,L,a,b)",
        type=["csv"],
        key="direct_ref",
    )
    if direct is not None:
        try:
            refs = validate_reference_table(pd.read_csv(direct))
            st.session_state["vita_refs"] = refs
            st.dataframe(refs, use_container_width=True, hide_index=True)
            st.success("Measured reference library loaded for this session.")
            st.download_button(
                "Download normalized reference library",
                data=refs.to_csv(index=False).encode("utf-8"),
                file_name="vita_3d_master_reference.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.markdown("**Exploratory fallback:** derive per-shade medians from Rayplicker observations.")
    ray = st.file_uploader(
        "Rayplicker observation table",
        type=["csv"],
        key="ray_ref_builder",
        help="Accepted columns: ray_shade,ray_L,ray_a,ray_b or shade,L,a,b.",
    )
    if ray is not None:
        try:
            centroids = build_reference_centroids(pd.read_csv(ray))
            st.dataframe(centroids, use_container_width=True, hide_index=True)
            st.warning(
                "This is a deterministic centroid reference, not ML training. However, do not test exact "
                "shade accuracy on the same observations used to build it because that would be circular."
            )
            if st.button("Use exploratory centroids in this session"):
                st.session_state["vita_refs"] = centroids[["shade","L","a","b"]].copy()
                st.success("Exploratory reference loaded.")
            st.download_button(
                "Download exploratory centroid library",
                data=centroids.to_csv(index=False).encode("utf-8"),
                file_name="rayplicker_derived_3d_master_centroids.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(str(exc))


def method_page():
    st.header("Method")
    st.markdown(
        """
### Detection pipeline

**Photograph → optical-status check → color calibration → tooth ROI → glare exclusion →
CIELAB → CIEDE2000 → VITA 3D-Master reference matching → regional shade map**

There is **no trained tooth-shade classifier** in this version.

### Polarization

If the photograph was truly cross-polarized at acquisition, select that option. If it was not,
the app excludes likely specular highlights, but software cannot turn an ordinary photograph
into a genuinely polarized acquisition.

### VITA 3D-Master mapping

The app does not use the old seven-shade hard-coded table. It requires a measured reference
library and assigns the nearest shade by **CIEDE2000**. It can produce three-zone, nine-zone,
or dense regional maps.

### ChatGPT

ChatGPT now has two clearly separated optional roles:

1. **Independent visual comparator:** GPT receives only the tooth ROI and must choose one
   VITA 3D-Master shade from the allowed list. It does not receive calibrated CIELAB,
   CIEDE2000, Rayplicker, or other-model results. This is a visual estimate rather than a
   spectrophotometric measurement.
2. **Explanation layer:** GPT may explain the already-computed deterministic result for a
   dentist or patient.

The calibrated CIELAB → CIEDE2000 → VITA mapping remains the primary deterministic shade
engine. GPT does not modify L*, a*, b*, ΔE00, or the regional map.

### Validation

Rayplicker remains the reference for the thesis comparison. Compare the app's continuous
CIELAB values and ΔE00 with the Rayplicker values before emphasizing exact categorical shade
agreement.
"""
    )


if page == "Analyze tooth":
    analyze_page()
elif page == "Build calibration":
    calibration_page()
elif page == "Build shade reference":
    reference_page()
elif page == "Batch research benchmark":
    render_batch_research_page()
elif page == "VLM Lab benchmark":
    render_vlm_lab_research_page()
else:
    method_page()

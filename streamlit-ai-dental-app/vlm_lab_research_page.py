from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from gpt_vision import estimate_visual_lab, normalize_shade_name
from shade_engine import delta_e00, nearest_shades, validate_reference_table


APP_DIR = Path(__file__).resolve().parent
DEFAULT_VITA_REFERENCE = APP_DIR / "vita_3d_master_reference_published.csv"


def _read_secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def _require_research_access() -> bool:
    expected = _read_secret("RESEARCH_BENCHMARK_PASSWORD")
    if not expected:
        st.error(
            "Research benchmark access is disabled. Add RESEARCH_BENCHMARK_PASSWORD "
            "to Streamlit secrets before using this page."
        )
        return False
    entered = st.text_input(
        "Research benchmark passphrase",
        type="password",
        key="vlm_lab_passphrase",
    )
    if not entered:
        st.info("Enter the private research passphrase to unlock this workspace.")
        return False
    if entered != expected:
        st.error("Incorrect passphrase.")
        return False
    return True


def _index_zip_images(zip_bytes: bytes) -> dict[str, bytes]:
    indexed = {}
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            data = zf.read(name)
            indexed[name] = data
            indexed[Path(name).name] = data
    return indexed


def _resolve_image(indexed: dict[str, bytes], image_path: str) -> bytes:
    raw = str(image_path).replace("\\", "/")
    if raw in indexed:
        return indexed[raw]
    base = Path(raw).name
    if base in indexed:
        return indexed[base]
    matches = [
        data for name, data in indexed.items()
        if name.replace("\\", "/").endswith(raw)
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Image '{image_path}' was not found uniquely in the uploaded ZIP."
    )


def _validate_manifest(df: pd.DataFrame) -> pd.DataFrame:
    required = ["case_id", "image_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    out = df.copy()
    out["case_id"] = out["case_id"].astype(str)
    if out["case_id"].duplicated().any():
        raise ValueError("case_id must be unique in the manifest.")
    return out


def _load_reference(uploaded_reference):
    if uploaded_reference is not None:
        return validate_reference_table(pd.read_csv(uploaded_reference))
    if DEFAULT_VITA_REFERENCE.exists():
        return validate_reference_table(pd.read_csv(DEFAULT_VITA_REFERENCE))
    return None


def _run_key(case_id: str, run_number: int, model: str) -> str:
    return f"{model}::{case_id}::{run_number}"


def _pairwise_delta_e00(arrays: list[np.ndarray]) -> float:
    vals = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            vals.append(delta_e00(arrays[i], arrays[j]))
    return float(np.mean(vals)) if vals else np.nan


def _case_summary(
    manifest: pd.DataFrame,
    runs: pd.DataFrame,
    refs: pd.DataFrame | None,
) -> pd.DataFrame:
    successful = runs.loc[runs["status"] == "ok"].copy()
    rows = []

    for _, case in manifest.iterrows():
        cid = str(case["case_id"])
        sub = successful.loc[successful["case_id"].astype(str) == cid].copy()
        for c in ["pred_L", "pred_a", "pred_b"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna(subset=["pred_L", "pred_a", "pred_b"])

        mean_lab = None
        if not sub.empty:
            mean_lab = sub[["pred_L", "pred_a", "pred_b"]].mean().to_numpy(dtype=float)

        ray_lab = None
        if all(c in manifest.columns for c in ["ray_L", "ray_a", "ray_b"]):
            vals = [case.get("ray_L"), case.get("ray_a"), case.get("ray_b")]
            if all(pd.notna(v) for v in vals):
                ray_lab = np.asarray(vals, dtype=float)

        derived_shade = None
        derived_shade_de = np.nan
        if refs is not None and mean_lab is not None:
            nearest = nearest_shades(mean_lab, refs, top_k=1)
            if not nearest.empty:
                derived_shade = normalize_shade_name(str(nearest.iloc[0]["shade"]))
                derived_shade_de = float(nearest.iloc[0]["delta_e00"])

        ray_shade = None
        if pd.notna(case.get("ray_shade")) and str(case.get("ray_shade")).strip():
            ray_shade = normalize_shade_name(str(case.get("ray_shade")))

        exact = (
            derived_shade == ray_shade
            if derived_shade is not None and ray_shade is not None
            else np.nan
        )

        direct_de = (
            delta_e00(mean_lab, ray_lab)
            if mean_lab is not None and ray_lab is not None
            else np.nan
        )

        run_des = []
        if ray_lab is not None:
            for _, rr in sub.iterrows():
                run_des.append(
                    delta_e00(
                        np.asarray([rr["pred_L"], rr["pred_a"], rr["pred_b"]], dtype=float),
                        ray_lab,
                    )
                )

        pred_arrays = [
            np.asarray([r.pred_L, r.pred_a, r.pred_b], dtype=float)
            for r in sub.itertuples()
        ]

        rows.append({
            "case_id": cid,
            "image_path": case["image_path"],
            "ray_shade": ray_shade,
            "ray_L": case.get("ray_L"),
            "ray_a": case.get("ray_a"),
            "ray_b": case.get("ray_b"),
            "successful_runs": int(len(sub)),
            "mean_pred_L": float(mean_lab[0]) if mean_lab is not None else np.nan,
            "mean_pred_a": float(mean_lab[1]) if mean_lab is not None else np.nan,
            "mean_pred_b": float(mean_lab[2]) if mean_lab is not None else np.nan,
            "sd_pred_L": float(sub["pred_L"].std(ddof=1)) if len(sub) > 1 else np.nan,
            "sd_pred_a": float(sub["pred_a"].std(ddof=1)) if len(sub) > 1 else np.nan,
            "sd_pred_b": float(sub["pred_b"].std(ddof=1)) if len(sub) > 1 else np.nan,
            "within_case_mean_pairwise_delta_e00": _pairwise_delta_e00(pred_arrays),
            "case_mean_lab_vs_ray_delta_e00": direct_de,
            "mean_run_level_lab_vs_ray_delta_e00": float(np.mean(run_des)) if run_des else np.nan,
            "derived_vita_shade_from_mean_lab": derived_shade,
            "derived_vita_reference_delta_e00": derived_shade_de,
            "derived_shade_exact_match_rayplicker": exact,
            "mean_confidence_percent": pd.to_numeric(
                sub.get("confidence_percent"), errors="coerce"
            ).mean() if not sub.empty else np.nan,
        })

    return pd.DataFrame(rows)


def _overall_summary(case_summary: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    ok = runs.loc[runs["status"] == "ok"].copy()
    case_de = pd.to_numeric(case_summary["case_mean_lab_vs_ray_delta_e00"], errors="coerce")
    pair_de = pd.to_numeric(
        case_summary["within_case_mean_pairwise_delta_e00"], errors="coerce"
    )
    exact = pd.to_numeric(
        case_summary["derived_shade_exact_match_rayplicker"], errors="coerce"
    )

    # Coordinate-wise MAE from the averaged 3-run case prediction.
    evaluable = case_summary.dropna(
        subset=["mean_pred_L", "mean_pred_a", "mean_pred_b", "ray_L", "ray_a", "ray_b"]
    ).copy()
    if not evaluable.empty:
        L_mae = float(np.mean(np.abs(evaluable["mean_pred_L"] - evaluable["ray_L"])))
        a_mae = float(np.mean(np.abs(evaluable["mean_pred_a"] - evaluable["ray_a"])))
        b_mae = float(np.mean(np.abs(evaluable["mean_pred_b"] - evaluable["ray_b"])))
    else:
        L_mae = a_mae = b_mae = np.nan

    return pd.DataFrame([{
        "planned_calls": int(len(runs)),
        "valid_calls": int((runs["status"] == "ok").sum()),
        "valid_call_rate": float((runs["status"] == "ok").mean()) if len(runs) else np.nan,
        "n_cases": int(case_summary["case_id"].nunique()),
        "L_MAE_case_mean": L_mae,
        "a_MAE_case_mean": a_mae,
        "b_MAE_case_mean": b_mae,
        "mean_delta_e00_case_mean_lab_vs_ray": case_de.mean(),
        "median_delta_e00_case_mean_lab_vs_ray": case_de.median(),
        "mean_within_case_pairwise_delta_e00": pair_de.mean(),
        "median_within_case_pairwise_delta_e00": pair_de.median(),
        "derived_vita_exact_accuracy": exact.mean() if exact.notna().any() else np.nan,
        "mean_model_reported_confidence_percent": pd.to_numeric(
            ok.get("confidence_percent"), errors="coerce"
        ).mean() if not ok.empty else np.nan,
    }])


def _bundle_results(
    runs: pd.DataFrame,
    case_summary: pd.DataFrame,
    overall: pd.DataFrame,
    raw_json: dict[str, dict],
    metadata: dict,
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("vlm_lab_runs.csv", runs.to_csv(index=False))
        zf.writestr("vlm_lab_case_summary.csv", case_summary.to_csv(index=False))
        zf.writestr("vlm_lab_overall_summary.csv", overall.to_csv(index=False))
        zf.writestr("run_metadata.json", json.dumps(metadata, indent=2))
        for key, payload in raw_json.items():
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
            zf.writestr(
                f"raw_vlm_lab_json/{safe}.json",
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
    buffer.seek(0)
    return buffer.getvalue()


def render_vlm_lab_research_page():
    st.header("Private VLM CIELAB benchmark")
    st.caption(
        "Blinded zero-shot experiment: tooth image → VLM-estimated L*, a*, b*. "
        "Rayplicker and VITA references are used only after each inference."
    )

    if not _require_research_access():
        return

    api_key = _read_secret("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not configured in Streamlit secrets.")
        return

    default_model = _read_secret(
        "OPENAI_VISION_MODEL",
        _read_secret("OPENAI_MODEL", "gpt-5.6-luna"),
    )

    st.success(
        "Research workspace unlocked. The existing OpenAI API secret will be used server-side."
    )
    st.info(
        "This experiment estimates apparent CIELAB from the image. It does not constitute "
        "spectrophotometric measurement or calibrated colorimetry."
    )

    c1, c2 = st.columns(2)
    with c1:
        image_zip = st.file_uploader(
            "Private ZIP containing de-identified tooth crops",
            type=["zip"],
            key="vlm_lab_zip",
        )
        manifest_upload = st.file_uploader(
            "Private case manifest CSV",
            type=["csv"],
            key="vlm_lab_manifest",
        )
    with c2:
        vita_upload = st.file_uploader(
            "Optional VITA 3D-Master reference CSV",
            type=["csv"],
            key="vlm_lab_vita",
        )
        model = st.text_input(
            "Vision model",
            value=str(default_model),
            key="vlm_lab_model",
        )

    repeats = st.number_input(
        "Repeated runs per tooth",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key="vlm_lab_repeats",
    )
    language = st.selectbox(
        "Descriptive output language",
        ["English", "Arabic"],
        key="vlm_lab_language",
    )

    if image_zip is None or manifest_upload is None:
        st.info("Upload the same private image ZIP and manifest used for the shade benchmark.")
        return

    try:
        manifest = _validate_manifest(pd.read_csv(manifest_upload))
        indexed = _index_zip_images(image_zip.getvalue())
        missing = []
        for p in manifest["image_path"].astype(str):
            try:
                _resolve_image(indexed, p)
            except Exception:
                missing.append(p)
        if missing:
            st.error(f"{len(missing)} manifest images could not be resolved in the ZIP.")
            st.dataframe(pd.DataFrame({"missing_image_path": missing}), hide_index=True)
            return
        refs = _load_reference(vita_upload)
    except Exception as exc:
        st.error(str(exc))
        return

    planned = len(manifest) * int(repeats)
    st.success(f"Validated {len(manifest)} cases; all manifest images were found.")
    st.caption(f"Planned API calls: {planned}")

    if "vlm_lab_records" not in st.session_state:
        st.session_state["vlm_lab_records"] = {}
    if "vlm_lab_raw" not in st.session_state:
        st.session_state["vlm_lab_raw"] = {}

    records: dict = st.session_state["vlm_lab_records"]
    raw: dict = st.session_state["vlm_lab_raw"]

    keys = {
        _run_key(str(row["case_id"]), run_no, model)
        for _, row in manifest.iterrows()
        for run_no in range(1, int(repeats) + 1)
    }
    completed_ok = sum(
        1 for k in keys if k in records and records[k].get("status") == "ok"
    )
    st.progress(completed_ok / planned if planned else 0.0)
    st.caption(f"Completed valid runs: {completed_ok}/{planned}")

    c_run, c_reset = st.columns([2, 1])
    with c_run:
        do_run = st.button(
            "Run / resume blinded VLM-Lab benchmark",
            type="primary",
            key="run_vlm_lab_batch",
        )
    with c_reset:
        do_reset = st.button(
            "Reset VLM-Lab checkpoint",
            key="reset_vlm_lab_batch",
        )

    if do_reset:
        st.session_state["vlm_lab_records"] = {}
        st.session_state["vlm_lab_raw"] = {}
        st.rerun()

    if do_run:
        prog = st.progress(completed_ok / planned if planned else 0.0)
        status_box = st.empty()
        done = completed_ok

        for _, case in manifest.iterrows():
            cid = str(case["case_id"])
            image_bytes = _resolve_image(indexed, str(case["image_path"]))
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            for run_no in range(1, int(repeats) + 1):
                key = _run_key(cid, run_no, model)
                if key in records and records[key].get("status") == "ok":
                    continue

                status_box.write(f"Running {cid}, repeat {run_no}/{int(repeats)}...")
                base = {
                    "case_id": cid,
                    "image_path": case["image_path"],
                    "run_number": run_no,
                    "model": model,
                    "status": "error",
                    "pred_L": np.nan,
                    "pred_a": np.nan,
                    "pred_b": np.nan,
                    "confidence_percent": np.nan,
                    "image_quality": "",
                    "glare": "",
                    "blur": "",
                    "exposure": "",
                    "notes": "",
                    "error": "",
                }

                try:
                    result = estimate_visual_lab(
                        image=image,
                        api_key=api_key,
                        model=model,
                        language=language,
                    )
                    base.update({
                        "status": "ok",
                        "pred_L": result["pred_L"],
                        "pred_a": result["pred_a"],
                        "pred_b": result["pred_b"],
                        "confidence_percent": result.get("confidence_percent"),
                        "image_quality": result.get("image_quality", ""),
                        "glare": result.get("glare", ""),
                        "blur": result.get("blur", ""),
                        "exposure": result.get("exposure", ""),
                        "notes": result.get("notes", ""),
                    })
                    raw[key] = {
                        "case_id": cid,
                        "run_number": run_no,
                        "model": model,
                        "response": result.get("raw_response", ""),
                    }
                except Exception as exc:
                    base["error"] = str(exc)

                records[key] = base
                done = sum(
                    1 for k in keys
                    if k in records and records[k].get("status") == "ok"
                )
                prog.progress(done / planned if planned else 0.0)

        st.session_state["vlm_lab_records"] = records
        st.session_state["vlm_lab_raw"] = raw
        status_box.success("Batch pass complete. Failed calls, if any, can be retried with Run / resume.")

    current = [records[k] for k in keys if k in records]
    if not current:
        return

    runs = pd.DataFrame(current).sort_values(["case_id", "run_number"])
    case_summary = _case_summary(manifest, runs, refs)
    overall = _overall_summary(case_summary, runs)

    st.subheader("Current results")
    st.dataframe(overall, use_container_width=True, hide_index=True)

    if len(overall):
        r = overall.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Valid calls",
            f"{int(r['valid_calls'])}/{int(r['planned_calls'])}",
        )
        c2.metric(
            "Mean ΔE00 vs Rayplicker",
            f"{r['mean_delta_e00_case_mean_lab_vs_ray']:.2f}"
            if pd.notna(r["mean_delta_e00_case_mean_lab_vs_ray"]) else "N/A",
        )
        c3.metric(
            "Within-case repeat ΔE00",
            f"{r['mean_within_case_pairwise_delta_e00']:.2f}"
            if pd.notna(r["mean_within_case_pairwise_delta_e00"]) else "N/A",
        )
        c4.metric(
            "Derived VITA exact accuracy",
            f"{100*r['derived_vita_exact_accuracy']:.1f}%"
            if pd.notna(r["derived_vita_exact_accuracy"]) else "N/A",
        )

    st.dataframe(case_summary, use_container_width=True, hide_index=True)

    metadata = {
        "experiment": "Blinded zero-shot VLM CIELAB regression",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "repeats_per_case": int(repeats),
        "n_cases": int(len(manifest)),
        "planned_calls": int(planned),
        "valid_calls": int((runs["status"] == "ok").sum()),
        "blinding": (
            "The VLM received the tooth image only. Rayplicker values, VITA labels, "
            "VITA reference coordinates, calibrated deterministic CIELAB, and other-model "
            "predictions were withheld during inference."
        ),
        "interpretation": (
            "Returned L*, a*, b* values are VLM image estimates, not physical "
            "spectrophotometric measurements."
        ),
    }

    bundle = _bundle_results(runs, case_summary, overall, raw, metadata)
    st.download_button(
        "Download VLM-Lab batch results ZIP",
        data=bundle,
        file_name="VLM_Lab_batch_results.zip",
        mime="application/zip",
    )
    st.download_button(
        "Download VLM-Lab run CSV",
        data=runs.to_csv(index=False).encode("utf-8"),
        file_name="vlm_lab_runs.csv",
        mime="text/csv",
    )

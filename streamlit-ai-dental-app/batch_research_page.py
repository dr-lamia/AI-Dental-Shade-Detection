from __future__ import annotations

import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from gpt_vision import ALLOWED_3D_MASTER_SHADES, estimate_visual_shade, normalize_shade_name
from shade_engine import delta_e00, validate_reference_table


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

    entered = st.text_input("Research benchmark passphrase", type="password")
    if not entered:
        st.info("Enter the private research passphrase to unlock this workspace.")
        return False
    if entered != expected:
        st.error("Incorrect passphrase.")
        return False
    return True


def _normalize_optional_shade(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return normalize_shade_name(str(value))


def _load_reference(uploaded_reference):
    if uploaded_reference is not None:
        return validate_reference_table(pd.read_csv(uploaded_reference))
    if DEFAULT_VITA_REFERENCE.exists():
        return validate_reference_table(pd.read_csv(DEFAULT_VITA_REFERENCE))
    return None


def _index_zip_images(zip_bytes: bytes) -> dict[str, bytes]:
    indexed = {}
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png"}:
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

    suffix_matches = [
        data for name, data in indexed.items()
        if name.replace("\\", "/").endswith(raw)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise FileNotFoundError(f"Image '{image_path}' was not found uniquely in the uploaded ZIP.")


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


def _run_key(case_id: str, run_number: int, model: str) -> str:
    return f"{model}::{case_id}::{run_number}"


def _reference_lab(refs: pd.DataFrame | None, shade):
    shade = _normalize_optional_shade(shade)
    if refs is None or shade is None:
        return None
    work = refs.copy()
    work["_shade_norm"] = work["shade"].astype(str).map(normalize_shade_name)
    hit = work.loc[work["_shade_norm"] == shade]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return np.array([float(row["L"]), float(row["a"]), float(row["b"])], dtype=float)


def _case_summary(manifest: pd.DataFrame, runs: pd.DataFrame, refs: pd.DataFrame | None):
    rows = []
    successful = runs.loc[runs["status"] == "ok"].copy()

    for _, case in manifest.iterrows():
        cid = str(case["case_id"])
        sub = successful.loc[successful["case_id"].astype(str) == cid].copy()
        shades = [
            normalize_shade_name(x)
            for x in sub["predicted_shade"].dropna().astype(str).tolist()
        ]
        counts = Counter(shades)
        most = counts.most_common()
        consensus = None
        modal_count = 0
        if most:
            modal_count = most[0][1]
            tied = [s for s, n in most if n == modal_count]
            if len(tied) == 1 and modal_count >= 2:
                consensus = tied[0]

        ray_shade = _normalize_optional_shade(case.get("ray_shade"))
        exact = (
            consensus == ray_shade
            if consensus is not None and ray_shade is not None
            else np.nan
        )

        run_match_rate = np.nan
        if ray_shade is not None and shades:
            run_match_rate = float(np.mean([s == ray_shade for s in shades]))

        ray_lab = None
        if all(c in manifest.columns for c in ["ray_L", "ray_a", "ray_b"]):
            vals = [case.get("ray_L"), case.get("ray_a"), case.get("ray_b")]
            if all(pd.notna(v) for v in vals):
                ray_lab = np.asarray(vals, dtype=float)

        tab_lab = _reference_lab(refs, consensus)
        tab_de = (
            delta_e00(tab_lab, ray_lab)
            if tab_lab is not None and ray_lab is not None
            else np.nan
        )

        rows.append({
            "case_id": cid,
            "image_path": case["image_path"],
            "ray_shade": ray_shade,
            "ray_L": case.get("ray_L"),
            "ray_a": case.get("ray_a"),
            "ray_b": case.get("ray_b"),
            "gpt_successful_runs": len(shades),
            "gpt_unique_shades": len(counts),
            "gpt_unanimous": bool(shades and len(counts) == 1),
            "gpt_modal_share": (modal_count / len(shades)) if shades else np.nan,
            "gpt_consensus_shade": consensus,
            "gpt_exact_match_rayplicker_consensus": exact,
            "gpt_run_level_exact_match_rate": run_match_rate,
            "gpt_mean_confidence_percent": pd.to_numeric(
                sub.get("confidence_percent"), errors="coerce"
            ).mean() if not sub.empty else np.nan,
            "gpt_selected_tab_vs_rayplicker_delta_e00": tab_de,
        })
    return pd.DataFrame(rows)


def _merge_method_comparison(
    manifest: pd.DataFrame,
    gpt_summary: pd.DataFrame,
    refs: pd.DataFrame | None,
    external: pd.DataFrame | None,
):
    prediction_rows = []
    for _, r in gpt_summary.iterrows():
        prediction_rows.append({
            "case_id": str(r["case_id"]),
            "method": "GPT Vision consensus",
            "predicted_shade": r.get("gpt_consensus_shade"),
            "pred_L": np.nan,
            "pred_a": np.nan,
            "pred_b": np.nan,
        })

    if external is not None:
        required = {"case_id", "method"}
        if not required.issubset(external.columns):
            raise ValueError("External predictions require case_id and method columns.")
        ext = external.copy()
        for c in ["predicted_shade", "pred_L", "pred_a", "pred_b"]:
            if c not in ext.columns:
                ext[c] = np.nan
        ext["case_id"] = ext["case_id"].astype(str)
        valid_ids = set(manifest["case_id"].astype(str))
        ext = ext.loc[ext["case_id"].isin(valid_ids)]
        if ext.duplicated(["case_id", "method"]).any():
            raise ValueError("External predictions must contain at most one row per case_id + method.")
        prediction_rows.extend(
            ext[["case_id", "method", "predicted_shade", "pred_L", "pred_a", "pred_b"]]
            .to_dict(orient="records")
        )

    pred = pd.DataFrame(prediction_rows)
    truth_cols = ["case_id"] + [
        c for c in ["ray_shade", "ray_L", "ray_a", "ray_b"] if c in manifest.columns
    ]
    truth = manifest[truth_cols].copy()
    truth["case_id"] = truth["case_id"].astype(str)
    merged = pred.merge(truth, on="case_id", how="left", validate="many_to_one")

    exact_vals = []
    direct_de = []
    tab_de = []

    for _, r in merged.iterrows():
        ray_shade = _normalize_optional_shade(r.get("ray_shade"))
        pred_shade = _normalize_optional_shade(r.get("predicted_shade"))
        exact_vals.append(
            pred_shade == ray_shade
            if pred_shade is not None and ray_shade is not None
            else np.nan
        )

        ray_lab = None
        if all(pd.notna(r.get(c)) for c in ["ray_L", "ray_a", "ray_b"]):
            ray_lab = np.array(
                [float(r["ray_L"]), float(r["ray_a"]), float(r["ray_b"])],
                dtype=float,
            )

        pred_lab = None
        if all(pd.notna(r.get(c)) for c in ["pred_L", "pred_a", "pred_b"]):
            pred_lab = np.array(
                [float(r["pred_L"]), float(r["pred_a"]), float(r["pred_b"])],
                dtype=float,
            )

        direct_de.append(
            delta_e00(pred_lab, ray_lab)
            if pred_lab is not None and ray_lab is not None
            else np.nan
        )

        ref_lab = _reference_lab(refs, pred_shade)
        tab_de.append(
            delta_e00(ref_lab, ray_lab)
            if ref_lab is not None and ray_lab is not None
            else np.nan
        )

    merged["exact_shade_match"] = exact_vals
    merged["delta_e00_pred_lab_vs_ray"] = direct_de
    merged["delta_e00_selected_tab_vs_ray"] = tab_de

    summaries = []
    for method, sub in merged.groupby("method", sort=True):
        exact = pd.to_numeric(sub["exact_shade_match"], errors="coerce")
        lab_de = pd.to_numeric(sub["delta_e00_pred_lab_vs_ray"], errors="coerce")
        tab = pd.to_numeric(sub["delta_e00_selected_tab_vs_ray"], errors="coerce")
        summaries.append({
            "method": method,
            "n_cases_present": int(sub["case_id"].nunique()),
            "n_exact_shade_evaluable": int(exact.notna().sum()),
            "exact_shade_accuracy": exact.dropna().mean() if exact.notna().any() else np.nan,
            "n_direct_lab_evaluable": int(lab_de.notna().sum()),
            "mean_delta_e00_pred_lab_vs_ray": lab_de.dropna().mean() if lab_de.notna().any() else np.nan,
            "median_delta_e00_pred_lab_vs_ray": lab_de.dropna().median() if lab_de.notna().any() else np.nan,
            "n_selected_tab_evaluable": int(tab.notna().sum()),
            "mean_delta_e00_selected_tab_vs_ray": tab.dropna().mean() if tab.notna().any() else np.nan,
            "median_delta_e00_selected_tab_vs_ray": tab.dropna().median() if tab.notna().any() else np.nan,
        })

    return merged, pd.DataFrame(summaries)


def _bundle_results(
    runs: pd.DataFrame,
    case_summary: pd.DataFrame,
    case_comparison: pd.DataFrame,
    method_summary: pd.DataFrame,
    raw_json: dict[str, dict],
    metadata: dict,
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("gpt_vision_runs.csv", runs.to_csv(index=False))
        zf.writestr("gpt_case_summary.csv", case_summary.to_csv(index=False))
        zf.writestr("case_level_method_comparison.csv", case_comparison.to_csv(index=False))
        zf.writestr("method_summary.csv", method_summary.to_csv(index=False))
        zf.writestr("run_metadata.json", json.dumps(metadata, indent=2))
        for key, payload in raw_json.items():
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
            zf.writestr(
                f"raw_gpt_json/{safe}.json",
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
    buffer.seek(0)
    return buffer.getvalue()


def render_batch_research_page():
    st.header("Private batch research benchmark")
    st.caption(
        "Runs repeated blinded GPT Vision shade estimates on de-identified tooth crops. "
        "Rayplicker and other-model results are used only after each GPT inference."
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

    st.success("Research workspace unlocked. The existing OpenAI API secret will be used server-side.")

    c1, c2 = st.columns(2)
    with c1:
        image_zip = st.file_uploader(
            "Private ZIP containing de-identified tooth crops",
            type=["zip"],
            key="batch_image_zip",
        )
        manifest_upload = st.file_uploader(
            "Private case manifest CSV",
            type=["csv"],
            key="batch_manifest",
        )
    with c2:
        external_upload = st.file_uploader(
            "Optional same-case model predictions CSV",
            type=["csv"],
            key="batch_external_predictions",
        )
        vita_upload = st.file_uploader(
            "Optional VITA 3D-Master reference CSV",
            type=["csv"],
            key="batch_vita_reference",
        )

    model = st.text_input("Vision model", value=str(default_model))
    repeats = st.number_input("Repeated runs per tooth", min_value=1, max_value=10, value=3, step=1)
    language = st.selectbox("Descriptive output language", ["English", "Arabic"])

    st.warning(
        "The completed manifest and images stay in this Streamlit session. "
        "Do not upload identifiable patient data. This tool does not commit research files to GitHub."
    )

    if image_zip is None or manifest_upload is None:
        st.info("Upload the private image ZIP and manifest to validate the batch.")
        return

    try:
        manifest = _validate_manifest(pd.read_csv(manifest_upload))
        indexed_images = _index_zip_images(image_zip.getvalue())
        missing = []
        for p in manifest["image_path"].astype(str):
            try:
                _resolve_image(indexed_images, p)
            except Exception:
                missing.append(p)
        if missing:
            st.error(f"{len(missing)} manifest images could not be resolved in the ZIP.")
            st.dataframe(pd.DataFrame({"missing_image_path": missing}), hide_index=True)
            return

        refs = _load_reference(vita_upload)
        external = pd.read_csv(external_upload) if external_upload is not None else None
    except Exception as exc:
        st.error(str(exc))
        return

    st.success(f"Validated {len(manifest)} cases; all manifest images were found in the ZIP.")
    st.caption(f"Planned API calls: {len(manifest) * int(repeats)}")

    if "batch_gpt_records" not in st.session_state:
        st.session_state["batch_gpt_records"] = {}
    if "batch_gpt_raw" not in st.session_state:
        st.session_state["batch_gpt_raw"] = {}

    run_records: dict = st.session_state["batch_gpt_records"]
    raw_records: dict = st.session_state["batch_gpt_raw"]

    model_keys = {
        _run_key(str(row["case_id"]), run_no, model)
        for _, row in manifest.iterrows()
        for run_no in range(1, int(repeats) + 1)
    }
    completed = sum(
        1 for k in model_keys
        if k in run_records and run_records[k].get("status") == "ok"
    )

    st.metric("Completed successful calls in this session", f"{completed}/{len(model_keys)}")

    b1, b2 = st.columns(2)
    start = b1.button("Run / resume blinded GPT benchmark", type="primary")
    clear = b2.button("Clear current batch session")

    if clear:
        st.session_state["batch_gpt_records"] = {}
        st.session_state["batch_gpt_raw"] = {}
        st.rerun()

    if start:
        progress = st.progress(completed / max(1, len(model_keys)))
        status = st.empty()
        error_box = st.empty()

        for _, case in manifest.iterrows():
            cid = str(case["case_id"])
            image_bytes = _resolve_image(indexed_images, str(case["image_path"]))

            for run_no in range(1, int(repeats) + 1):
                key = _run_key(cid, run_no, model)
                if key in run_records and run_records[key].get("status") == "ok":
                    continue

                status.write(f"Running {cid} — repeat {run_no}/{int(repeats)}")
                try:
                    with Image.open(BytesIO(image_bytes)) as img:
                        roi = img.convert("RGB")
                        result = estimate_visual_shade(
                            image=roi,
                            api_key=api_key,
                            model=model,
                            allowed_shades=ALLOWED_3D_MASTER_SHADES,
                            language=language,
                        )

                    row = {
                        "case_id": cid,
                        "image_path": str(case["image_path"]),
                        "run_number": run_no,
                        "model": model,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "ok",
                        "predicted_shade": result.get("predicted_shade"),
                        "confidence_percent": result.get("confidence_percent"),
                        "image_quality": result.get("image_quality"),
                        "glare": result.get("glare"),
                        "blur": result.get("blur"),
                        "exposure": result.get("exposure"),
                        "cervical_middle_incisal_variation": result.get(
                            "cervical_middle_incisal_variation"
                        ),
                        "notes": result.get("notes"),
                        "error": None,
                    }
                    run_records[key] = row
                    raw_records[key] = {**row, "raw_response": result.get("raw_response")}
                    error_box.empty()
                except Exception as exc:
                    row = {
                        "case_id": cid,
                        "image_path": str(case["image_path"]),
                        "run_number": run_no,
                        "model": model,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "error",
                        "predicted_shade": None,
                        "confidence_percent": None,
                        "image_quality": None,
                        "glare": None,
                        "blur": None,
                        "exposure": None,
                        "cervical_middle_incisal_variation": None,
                        "notes": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    run_records[key] = row
                    raw_records[key] = row
                    error_box.warning(row["error"])

                st.session_state["batch_gpt_records"] = run_records
                st.session_state["batch_gpt_raw"] = raw_records

                completed_now = sum(
                    1 for k in model_keys
                    if k in run_records and run_records[k].get("status") == "ok"
                )
                progress.progress(completed_now / max(1, len(model_keys)))

        status.success("Batch pass finished. Failed calls, if any, can be retried with Run / resume.")

    relevant = [
        row for key, row in run_records.items()
        if key in model_keys
    ]
    if not relevant:
        return

    runs = pd.DataFrame(relevant).sort_values(["case_id", "run_number"])
    st.subheader("Current batch results")
    st.dataframe(runs, use_container_width=True, hide_index=True)

    case_summary = _case_summary(manifest, runs, refs)
    try:
        comparison, method_summary = _merge_method_comparison(
            manifest,
            case_summary,
            refs,
            external,
        )
    except Exception as exc:
        st.error(f"Could not merge optional comparison models: {exc}")
        comparison = pd.DataFrame()
        method_summary = pd.DataFrame()

    if not method_summary.empty:
        st.subheader("Method summary")
        display_summary = method_summary.copy()
        st.dataframe(display_summary, use_container_width=True, hide_index=True)

    successful_n = int((runs["status"] == "ok").sum())
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "repeats_requested": int(repeats),
        "n_cases": int(len(manifest)),
        "successful_api_calls": successful_n,
        "planned_api_calls": int(len(manifest) * int(repeats)),
        "blinding": (
            "GPT inference received only the tooth ROI and fixed VITA shade list. "
            "Rayplicker and external model results were merged after inference."
        ),
    }

    bundle = _bundle_results(
        runs,
        case_summary,
        comparison,
        method_summary,
        {k: v for k, v in raw_records.items() if k in model_keys},
        metadata,
    )

    st.download_button(
        "Download complete batch results ZIP",
        data=bundle,
        file_name="GPT_Vision_batch_results.zip",
        mime="application/zip",
    )
    st.download_button(
        "Download GPT run table CSV",
        data=runs.to_csv(index=False).encode("utf-8"),
        file_name="gpt_vision_runs.csv",
        mime="text/csv",
    )

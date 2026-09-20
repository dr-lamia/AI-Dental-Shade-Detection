from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import deltaE_ciede2000


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "streamlit-ai-dental-app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from gpt_vision import (  # noqa: E402
    ALLOWED_3D_MASTER_SHADES,
    estimate_visual_shade,
    normalize_shade_name,
)


MANIFEST_REQUIRED = ["case_id", "image_path"]
EXTERNAL_REQUIRED = ["case_id", "method"]
EXTERNAL_OPTIONAL = ["predicted_shade", "pred_L", "pred_a", "pred_b"]


def delta_e00(lab1, lab2) -> float:
    a = np.asarray(lab1, dtype=float).reshape(1, 1, 3)
    b = np.asarray(lab2, dtype=float).reshape(1, 1, 3)
    return float(deltaE_ciede2000(a, b)[0, 0])


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in MANIFEST_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if df["case_id"].astype(str).duplicated().any():
        dups = df.loc[df["case_id"].astype(str).duplicated(), "case_id"].tolist()
        raise ValueError(f"case_id must be unique in the manifest. Duplicates: {dups}")
    df["case_id"] = df["case_id"].astype(str)
    return df


def resolve_image_path(raw_path: str, manifest_dir: Path, image_root: Path | None) -> Path:
    p = Path(str(raw_path))
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    if image_root is not None:
        candidates.append(image_root / p)
        candidates.append(image_root / p.name)
    candidates.append(manifest_dir / p)
    candidates.append(manifest_dir / p.name)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Image not found for '{raw_path}'. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def load_vita_reference(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        default_path = APP_DIR / "vita_3d_master_reference_published.csv"
        path = default_path if default_path.exists() else None
    if path is None:
        return None

    refs = pd.read_csv(path)
    required = {"shade", "L", "a", "b"}
    if not required.issubset(refs.columns):
        raise ValueError(
            f"VITA reference must contain {sorted(required)}; found {list(refs.columns)}"
        )
    refs = refs.copy()
    refs["shade"] = refs["shade"].astype(str).map(normalize_shade_name)
    refs = refs.drop_duplicates("shade", keep="first")
    return refs


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def save_raw_json(raw_dir: Path, case_id: str, run_number: int, payload: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_case = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in case_id)
    out = raw_dir / f"{safe_case}__run{run_number:02d}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_optional_shade(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return normalize_shade_name(str(value))


def reference_lab_for_shade(refs: pd.DataFrame | None, shade: str | None):
    if refs is None or shade is None:
        return None
    hit = refs.loc[refs["shade"] == normalize_shade_name(shade)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return np.array([float(row["L"]), float(row["a"]), float(row["b"])], dtype=float)


def validate_manifest_images(
    manifest: pd.DataFrame,
    manifest_path: Path,
    image_root: Path | None,
) -> dict[str, Path]:
    resolved = {}
    for _, row in manifest.iterrows():
        case_id = str(row["case_id"])
        resolved[case_id] = resolve_image_path(
            str(row["image_path"]),
            manifest_path.parent,
            image_root,
        )
        with Image.open(resolved[case_id]) as img:
            img.verify()
    return resolved


def existing_success_keys(runs_path: Path) -> set[tuple[str, int]]:
    if not runs_path.exists():
        return set()
    old = pd.read_csv(runs_path)
    if old.empty or not {"case_id", "run_number", "status"}.issubset(old.columns):
        return set()
    ok = old.loc[old["status"].astype(str).str.lower() == "ok"]
    return {
        (str(r["case_id"]), int(r["run_number"]))
        for _, r in ok.iterrows()
    }


def append_checkpoint(runs_path: Path, row: dict) -> None:
    new = pd.DataFrame([row])
    if runs_path.exists():
        old = pd.read_csv(runs_path)
        combined = pd.concat([old, new], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(
            subset=["case_id", "run_number"],
            keep="last",
        )
    else:
        combined = new
    combined = combined.sort_values(["case_id", "run_number"]).reset_index(drop=True)
    atomic_csv(combined, runs_path)


def run_gpt_batch(
    manifest: pd.DataFrame,
    manifest_path: Path,
    image_paths: dict[str, Path],
    output_dir: Path,
    api_key: str,
    model: str,
    repeats: int,
    language: str,
    max_retries: int,
    retry_seconds: float,
    resume: bool,
) -> pd.DataFrame:
    runs_path = output_dir / "gpt_vision_runs.csv"
    raw_dir = output_dir / "raw_gpt_json"
    already_done = existing_success_keys(runs_path) if resume else set()

    allowed = ALLOWED_3D_MASTER_SHADES

    for _, row in manifest.iterrows():
        case_id = str(row["case_id"])
        image_path = image_paths[case_id]

        for run_number in range(1, repeats + 1):
            key = (case_id, run_number)
            if key in already_done:
                print(f"SKIP {case_id} run {run_number}: already completed")
                continue

            error_text = None
            result = None
            for attempt in range(1, max_retries + 1):
                try:
                    with Image.open(image_path) as im:
                        roi = im.convert("RGB")
                        result = estimate_visual_shade(
                            image=roi,
                            api_key=api_key,
                            model=model,
                            allowed_shades=allowed,
                            language=language,
                        )
                    error_text = None
                    break
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    if attempt < max_retries:
                        time.sleep(retry_seconds * attempt)

            timestamp = datetime.now(timezone.utc).isoformat()
            out_row = {
                "case_id": case_id,
                "image_path": str(row["image_path"]),
                "run_number": run_number,
                "model": model,
                "timestamp_utc": timestamp,
                "status": "ok" if result is not None else "error",
                "predicted_shade": result.get("predicted_shade") if result else None,
                "confidence_percent": result.get("confidence_percent") if result else None,
                "image_quality": result.get("image_quality") if result else None,
                "glare": result.get("glare") if result else None,
                "blur": result.get("blur") if result else None,
                "exposure": result.get("exposure") if result else None,
                "cervical_middle_incisal_variation": (
                    result.get("cervical_middle_incisal_variation") if result else None
                ),
                "notes": result.get("notes") if result else None,
                "error": error_text,
            }
            append_checkpoint(runs_path, out_row)

            raw_payload = {
                **out_row,
                "raw_response": result.get("raw_response") if result else None,
            }
            save_raw_json(raw_dir, case_id, run_number, raw_payload)

            if result is not None:
                print(
                    f"OK   {case_id} run {run_number}: "
                    f"{result['predicted_shade']} "
                    f"(confidence={result.get('confidence_percent')})"
                )
            else:
                print(f"ERR  {case_id} run {run_number}: {error_text}")

    return pd.read_csv(runs_path)


def summarize_gpt_cases(
    manifest: pd.DataFrame,
    runs: pd.DataFrame,
    refs: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    successful = runs.loc[runs["status"].astype(str).str.lower() == "ok"].copy()

    for _, case in manifest.iterrows():
        case_id = str(case["case_id"])
        sub = successful.loc[successful["case_id"].astype(str) == case_id].copy()
        shades = [
            normalize_shade_name(s)
            for s in sub["predicted_shade"].dropna().astype(str).tolist()
        ]
        counts = Counter(shades)
        n_success = len(shades)
        n_unique = len(counts)
        most_common = counts.most_common()

        consensus_shade = None
        majority_count = 0
        if most_common:
            top_count = most_common[0][1]
            tied = sorted([shade for shade, count in most_common if count == top_count])
            majority_count = top_count
            if len(tied) == 1 and top_count >= 2:
                consensus_shade = tied[0]

        ray_shade = normalize_optional_shade(case.get("ray_shade"))
        exact_consensus = (
            bool(consensus_shade == ray_shade)
            if consensus_shade is not None and ray_shade is not None
            else None
        )

        run_exact = None
        if ray_shade is not None and n_success:
            run_exact = float(np.mean([s == ray_shade for s in shades]))

        ray_lab = None
        if all(col in manifest.columns for col in ["ray_L", "ray_a", "ray_b"]):
            try:
                vals = [case.get("ray_L"), case.get("ray_a"), case.get("ray_b")]
                if all(pd.notna(v) for v in vals):
                    ray_lab = np.array([float(v) for v in vals], dtype=float)
            except Exception:
                ray_lab = None

        consensus_ref_lab = reference_lab_for_shade(refs, consensus_shade)
        gpt_tab_vs_ray_de00 = (
            delta_e00(consensus_ref_lab, ray_lab)
            if consensus_ref_lab is not None and ray_lab is not None
            else None
        )

        rows.append(
            {
                "case_id": case_id,
                "image_path": case["image_path"],
                "ray_shade": ray_shade,
                "ray_L": case.get("ray_L"),
                "ray_a": case.get("ray_a"),
                "ray_b": case.get("ray_b"),
                "gpt_model": (
                    sub["model"].iloc[0] if not sub.empty and "model" in sub.columns else None
                ),
                "gpt_successful_runs": n_success,
                "gpt_unique_shades": n_unique,
                "gpt_unanimous": bool(n_success > 0 and n_unique == 1),
                "gpt_majority_count": majority_count,
                "gpt_modal_share": (
                    float(majority_count / n_success) if n_success else None
                ),
                "gpt_consensus_shade": consensus_shade,
                "gpt_exact_match_rayplicker_consensus": exact_consensus,
                "gpt_run_level_exact_match_rate": run_exact,
                "gpt_mean_confidence_percent": (
                    float(pd.to_numeric(sub["confidence_percent"], errors="coerce").mean())
                    if not sub.empty
                    else None
                ),
                "gpt_selected_tab_vs_rayplicker_delta_e00": gpt_tab_vs_ray_de00,
            }
        )

    return pd.DataFrame(rows)


def standardize_external_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in EXTERNAL_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"External prediction file is missing required columns: {missing}. "
            "Required: case_id,method; optional: predicted_shade,pred_L,pred_a,pred_b."
        )

    for col in EXTERNAL_OPTIONAL:
        if col not in df.columns:
            df[col] = np.nan

    df = df[EXTERNAL_REQUIRED + EXTERNAL_OPTIONAL].copy()
    df["case_id"] = df["case_id"].astype(str)
    df["method"] = df["method"].astype(str)
    df["predicted_shade"] = df["predicted_shade"].map(normalize_optional_shade)

    if df.duplicated(["case_id", "method"]).any():
        raise ValueError(
            "External predictions must contain at most one row per case_id + method."
        )
    return df


def build_method_comparison(
    manifest: pd.DataFrame,
    gpt_summary: pd.DataFrame,
    refs: pd.DataFrame | None,
    external_predictions: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []

    # GPT consensus arm.
    for _, row in gpt_summary.iterrows():
        long_rows.append(
            {
                "case_id": str(row["case_id"]),
                "method": "GPT Vision consensus",
                "predicted_shade": row.get("gpt_consensus_shade"),
                "pred_L": np.nan,
                "pred_a": np.nan,
                "pred_b": np.nan,
            }
        )

    if external_predictions is not None:
        valid_case_ids = set(manifest["case_id"].astype(str))
        external = external_predictions.loc[
            external_predictions["case_id"].isin(valid_case_ids)
        ].copy()
        long_rows.extend(external.to_dict(orient="records"))

    preds = pd.DataFrame(long_rows)
    truth_cols = ["case_id"]
    for c in ["ray_shade", "ray_L", "ray_a", "ray_b"]:
        if c in manifest.columns:
            truth_cols.append(c)

    truth = manifest[truth_cols].copy()
    truth["case_id"] = truth["case_id"].astype(str)
    combined = preds.merge(truth, on="case_id", how="left", validate="many_to_one")

    if "ray_shade" in combined.columns:
        combined["ray_shade"] = combined["ray_shade"].map(normalize_optional_shade)
        combined["exact_shade_match"] = combined.apply(
            lambda r: (
                normalize_optional_shade(r["predicted_shade"]) == r["ray_shade"]
                if normalize_optional_shade(r["predicted_shade"]) is not None
                and r["ray_shade"] is not None
                else np.nan
            ),
            axis=1,
        )
    else:
        combined["exact_shade_match"] = np.nan

    combined["delta_e00_pred_lab_vs_ray"] = np.nan
    combined["delta_e00_selected_tab_vs_ray"] = np.nan

    has_ray_lab = all(c in combined.columns for c in ["ray_L", "ray_a", "ray_b"])
    if has_ray_lab:
        for idx, r in combined.iterrows():
            if all(pd.notna(r.get(c)) for c in ["ray_L", "ray_a", "ray_b"]):
                ray_lab = np.array(
                    [float(r["ray_L"]), float(r["ray_a"]), float(r["ray_b"])],
                    dtype=float,
                )
                if all(pd.notna(r.get(c)) for c in ["pred_L", "pred_a", "pred_b"]):
                    pred_lab = np.array(
                        [float(r["pred_L"]), float(r["pred_a"]), float(r["pred_b"])],
                        dtype=float,
                    )
                    combined.at[idx, "delta_e00_pred_lab_vs_ray"] = delta_e00(
                        pred_lab, ray_lab
                    )
                shade = normalize_optional_shade(r.get("predicted_shade"))
                tab_lab = reference_lab_for_shade(refs, shade)
                if tab_lab is not None:
                    combined.at[idx, "delta_e00_selected_tab_vs_ray"] = delta_e00(
                        tab_lab, ray_lab
                    )

    metric_rows = []
    for method, sub in combined.groupby("method", sort=True):
        exact = pd.to_numeric(sub["exact_shade_match"], errors="coerce")
        lab_de = pd.to_numeric(sub["delta_e00_pred_lab_vs_ray"], errors="coerce")
        tab_de = pd.to_numeric(sub["delta_e00_selected_tab_vs_ray"], errors="coerce")

        metric_rows.append(
            {
                "method": method,
                "n_cases_present": int(sub["case_id"].nunique()),
                "n_exact_shade_evaluable": int(exact.notna().sum()),
                "exact_shade_accuracy": (
                    float(exact.dropna().mean()) if exact.notna().any() else np.nan
                ),
                "n_direct_lab_evaluable": int(lab_de.notna().sum()),
                "mean_delta_e00_pred_lab_vs_ray": (
                    float(lab_de.dropna().mean()) if lab_de.notna().any() else np.nan
                ),
                "median_delta_e00_pred_lab_vs_ray": (
                    float(lab_de.dropna().median()) if lab_de.notna().any() else np.nan
                ),
                "n_selected_tab_evaluable": int(tab_de.notna().sum()),
                "mean_delta_e00_selected_tab_vs_ray": (
                    float(tab_de.dropna().mean()) if tab_de.notna().any() else np.nan
                ),
                "median_delta_e00_selected_tab_vs_ray": (
                    float(tab_de.dropna().median()) if tab_de.notna().any() else np.nan
                ),
            }
        )

    return combined, pd.DataFrame(metric_rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a blinded repeated GPT Vision shade benchmark and optionally merge "
            "case-matched ShadeGPT/RF/SVM/PLS predictions."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("gpt_vision_batch_output"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--language", default="English", choices=["English", "Arabic"])
    parser.add_argument("--vita-reference", type=Path, default=None)
    parser.add_argument("--external-predictions", type=Path, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=2.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    image_root = args.image_root.resolve() if args.image_root else None
    image_paths = validate_manifest_images(manifest, manifest_path, image_root)

    refs = load_vita_reference(
        args.vita_reference.resolve() if args.vita_reference else None
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = pd.DataFrame(
        [
            {
                "n_manifest_cases": len(manifest),
                "n_images_resolved": len(image_paths),
                "vita_reference_loaded": refs is not None,
                "dry_run": args.dry_run,
            }
        ]
    )
    atomic_csv(validation, output_dir / "input_validation.csv")

    if args.dry_run:
        print(f"Dry run OK: {len(manifest)} cases and {len(image_paths)} images resolved.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in the local environment before running "
            "the batch. Do not commit the key to GitHub."
        )

    model = (
        args.model
        or os.getenv("OPENAI_VISION_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-5.6-luna"
    )

    runs = run_gpt_batch(
        manifest=manifest,
        manifest_path=manifest_path,
        image_paths=image_paths,
        output_dir=output_dir,
        api_key=api_key,
        model=model,
        repeats=args.repeats,
        language=args.language,
        max_retries=args.max_retries,
        retry_seconds=args.retry_seconds,
        resume=not args.no_resume,
    )

    gpt_summary = summarize_gpt_cases(manifest, runs, refs)
    atomic_csv(gpt_summary, output_dir / "gpt_case_summary.csv")

    external = None
    if args.external_predictions is not None:
        external = standardize_external_predictions(
            args.external_predictions.resolve()
        )

    case_comparison, method_summary = build_method_comparison(
        manifest=manifest,
        gpt_summary=gpt_summary,
        refs=refs,
        external_predictions=external,
    )
    atomic_csv(case_comparison, output_dir / "case_level_method_comparison.csv")
    atomic_csv(method_summary, output_dir / "method_summary.csv")

    repeatability = pd.DataFrame(
        [
            {
                "model": model,
                "n_cases": int(len(gpt_summary)),
                "n_cases_with_successful_runs": int(
                    (gpt_summary["gpt_successful_runs"] > 0).sum()
                ),
                "unanimous_case_fraction": float(
                    gpt_summary.loc[
                        gpt_summary["gpt_successful_runs"] > 0,
                        "gpt_unanimous",
                    ].mean()
                )
                if (gpt_summary["gpt_successful_runs"] > 0).any()
                else np.nan,
                "mean_modal_share": float(
                    pd.to_numeric(gpt_summary["gpt_modal_share"], errors="coerce").mean()
                ),
                "mean_model_reported_confidence_percent": float(
                    pd.to_numeric(
                        gpt_summary["gpt_mean_confidence_percent"],
                        errors="coerce",
                    ).mean()
                ),
            }
        ]
    )
    atomic_csv(repeatability, output_dir / "gpt_repeatability_summary.csv")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "repeats": args.repeats,
        "n_cases": int(len(manifest)),
        "manifest": str(manifest_path),
        "vita_reference": (
            str(args.vita_reference.resolve())
            if args.vita_reference
            else (
                str(APP_DIR / "vita_3d_master_reference_published.csv")
                if (APP_DIR / "vita_3d_master_reference_published.csv").exists()
                else None
            )
        ),
        "external_predictions": (
            str(args.external_predictions.resolve())
            if args.external_predictions
            else None
        ),
        "methodological_note": (
            "GPT is blinded to Rayplicker, calibrated CIELAB/CIEDE2000, and other model "
            "predictions during inference. Other models are merged only by explicit case_id."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Completed. Results written to: {output_dir}")


if __name__ == "__main__":
    main()

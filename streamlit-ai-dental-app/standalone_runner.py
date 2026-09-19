"""Standalone ShadeGPT runner.

Runs the calibrated shade-analysis pipeline without Streamlit.

Example:
python standalone_runner.py \
  --image tooth.jpg \
  --reference vita_3d_master_reference.csv \
  --roi 0.25 0.15 0.75 0.90 \
  --map-mode "9 zones" \
  --output-dir results

Optional:
  --calibration-matrix shade_session_calibration_matrix.csv
  --cross-polarized
  --explain
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from shade_engine import (
    crop_fraction,
    dataframe_to_matrix,
    delta_e00,
    lab_to_rgb01,
    nearest_shades,
    region_grid,
    robust_lab,
    shade_map_table,
    validate_reference_table,
)

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def parse_args():
    p = argparse.ArgumentParser(description="Run ShadeGPT color analysis without Streamlit.")
    p.add_argument("--image", required=True, help="Clinical tooth photograph.")
    p.add_argument("--reference", required=False, help="CSV with shade,L,a,b.")
    p.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=[0.25, 0.15, 0.75, 0.90],
        help="Normalized tooth ROI coordinates in 0-1.",
    )
    p.add_argument(
        "--calibration-matrix",
        default=None,
        help="Optional session calibration matrix CSV.",
    )
    p.add_argument(
        "--map-mode",
        choices=["3 zones", "9 zones", "Detailed"],
        default="9 zones",
    )
    p.add_argument("--cross-polarized", action="store_true")
    p.add_argument("--incisal-at-top", action="store_true")
    p.add_argument("--previous-lab", nargs=3, type=float, default=None)
    p.add_argument("--output-dir", default="shade_results")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--language", choices=["English", "Arabic"], default="English")
    p.add_argument("--audience", choices=["Dentist", "Patient"], default="Dentist")
    return p.parse_args()


def make_map_figure(roi: Image.Image, table: pd.DataFrame, mode: str, output_path: Path):
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

        ax.add_patch(
            Rectangle(
                (x, y), cw, ch,
                facecolor=rgb,
                alpha=0.42,
                edgecolor="white",
                linewidth=1,
            )
        )
        ax.text(
            x + cw / 2,
            y + ch / 2,
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
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def optional_explanation(summary: dict, language: str, audience: str) -> str | None:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    prompt = f"""
You are explaining the output of a dental shade-analysis program.
Do not calculate or change the shade.
Do not alter L*, a*, b*, or DeltaE00.
Do not claim clinical-grade accuracy.

Audience: {audience}
Language: {language}
Analysis summary:
{json.dumps(summary, indent=2)}

Explain the result briefly and state that shade assignment was obtained
deterministically from CIELAB and CIEDE2000 nearest-reference matching.
"""

    response = client.responses.create(model=model, input=prompt)
    return response.output_text


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    roi = crop_fraction(image, *args.roi)

    matrix = None
    if args.calibration_matrix:
        matrix = dataframe_to_matrix(pd.read_csv(args.calibration_matrix))

    overall = robust_lab(
        roi,
        matrix=matrix,
        cross_polarized=args.cross_polarized,
        trim_percent=5.0,
    )

    summary = {
        "image": str(args.image),
        "roi": [float(x) for x in args.roi],
        "cross_polarized": bool(args.cross_polarized),
        "calibrated": matrix is not None,
        "L": overall.L,
        "a": overall.a,
        "b": overall.b,
        "usable_pixels": overall.n_pixels,
        "excluded_fraction": overall.excluded_fraction,
    }

    refs = None
    if args.reference:
        refs = validate_reference_table(pd.read_csv(args.reference))
        top3 = nearest_shades(overall.array(), refs, top_k=3)
        summary["best_shade"] = str(top3.iloc[0]["shade"])
        summary["best_shade_delta_e00"] = float(top3.iloc[0]["delta_e00"])
        summary["top3"] = [
            {
                "shade": str(r["shade"]),
                "delta_e00": float(r["delta_e00"]),
            }
            for _, r in top3.iterrows()
        ]

        map_table = shade_map_table(
            roi,
            refs,
            args.map_mode,
            matrix=matrix,
            cross_polarized=args.cross_polarized,
            incisal_at_top=args.incisal_at_top,
        )
        map_table.to_csv(out / "shade_map.csv", index=False)
        make_map_figure(roi, map_table, args.map_mode, out / "shade_map.png")

    if args.previous_lab is not None:
        summary["delta_e00_vs_previous"] = delta_e00(
            args.previous_lab,
            overall.array(),
        )

    roi.save(out / "tooth_roi.png")

    explanation = None
    if args.explain:
        explanation = optional_explanation(summary, args.language, args.audience)
        if explanation:
            (out / "chatgpt_explanation.txt").write_text(explanation, encoding="utf-8")
            summary["chatgpt_explanation_file"] = "chatgpt_explanation.txt"

    pd.DataFrame([summary]).to_csv(out / "overall_result.csv", index=False)
    (out / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

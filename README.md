# 🦷 ShadeGPT / AI Dental Shade Detection

A Streamlit app for calibrated photographic tooth-color analysis and VITA 3D-Master mapping.

## Current v2 workflow

**Photograph → optional color calibration → tooth ROI → glare/specular exclusion → CIELAB → CIEDE2000 → nearest measured VITA 3D-Master reference → regional shade map**

No tooth-shade classifier is trained in the current app.

### What changed from the original prototype

- Removed the hard-coded VITA Classic `A2` output.
- Removed the incomplete seven-shade 3D-Master lookup.
- Replaced Euclidean ΔE with **CIEDE2000 (ΔE00)**.
- Added **photo calibration before CIELAB**.
- Added support for a session calibration matrix derived from a photographed color target.
- Added optional neutral-patch white-balance correction.
- Added handling for **true cross-polarized photographs** and specular-highlight suppression for ordinary photographs.
- Added **3-zone, 9-zone and detailed regional shade maps**, conceptually following the regional mapping workflow used by Rayplicker.
- Added top-3 nearest 3D-Master references.
- Added an optional **GPT Vision independent visual comparator** that receives only the tooth ROI and selects one VITA 3D-Master shade from the allowed list. It does not replace or modify the deterministic shade engine.\n- Retained the separate optional ChatGPT explanation layer for dentist- or patient-facing interpretation.
- Retained before/after ΔE00 comparison and PDF reporting.

## VITA 3D-Master reference library

For publishable work, provide a CSV:

```text
shade,L,a,b
```

Prefer measurements from the physical VITA 3D-Master guide obtained under a defined reference setup.

The app can also build **exploratory per-shade medians** from Rayplicker observations. This is deterministic reference construction, not machine-learning training. Do not evaluate exact categorical shade accuracy using the same observations used to build the centroids.

## Polarization

Rayplicker acquisition provides a polarized image and full-tooth shade mapping. This app therefore distinguishes:

- a photograph that was **truly cross-polarized at acquisition**, and
- an ordinary clinical photograph where software can only suppress likely specular highlights.

Software cannot reconstruct a genuinely cross-polarized optical acquisition from an ordinary JPEG.

## Calibration

The app supports:

1. no calibration — exploratory only;
2. a neutral reference patch — white-balance/channel correction;
3. a multi-patch **session calibration matrix** fitted in XYZ before CIELAB conversion.

For a definitive clinical validation, photograph a color target under the same camera/flash/exposure/white-balance geometry as the tooth photograph.

## ChatGPT / GPT Vision

If `OPENAI_API_KEY` is configured in Streamlit secrets, the app exposes two separate optional OpenAI roles:

1. **Independent GPT Vision comparator:** the model receives only the selected tooth ROI and must choose one permitted VITA 3D-Master shade. It does not receive calibrated CIELAB, CIEDE2000, Rayplicker values, or predictions from other models.
2. **Explanation layer:** the model explains the already-computed deterministic CIELAB, ΔE00 and shade-map result for a dentist or patient.

The app reports agreement/disagreement between the GPT visual shade and the calibrated deterministic shade. When a VITA reference table is available, it also reports ΔE00 between the CIELAB coordinates of the GPT-selected VITA tab and the calibrated tooth CIELAB. This does **not** mean GPT directly measures L*, a*, or b*.

Optional Streamlit secrets:

```toml
OPENAI_API_KEY = "..."
OPENAI_MODEL = "gpt-5.6-luna"
OPENAI_VISION_MODEL = "gpt-5.6-luna"
```

If `OPENAI_VISION_MODEL` is omitted, the app falls back to `OPENAI_MODEL`.

## Run

```bash
cd streamlit-ai-dental-app
pip install -r requirements.txt
streamlit run app.py
```

## Research-validation branch

The repository also contains `research-validation/`, which performs paired agreement analysis against Rayplicker reference values, including Bland–Altman analysis, MAE/RMSE, concordance, longitudinal ΔE00 and a leakage-free mean-color negative control.

## Important methodological point

The current upgraded clinical app is **not trained on the shade labels of the two thesis patients**. Shade assignment is based on calibrated CIELAB values and nearest-reference matching with CIEDE2000.


### Private batch research benchmark

The Streamlit app includes a protected **Batch research benchmark** workspace for repeated GPT Vision evaluation. It uses the same server-side `OPENAI_API_KEY` already configured for the app, so the API key is never uploaded through the research interface.

To enable the protected page, add one additional Streamlit secret of your choice:

```toml
RESEARCH_BENCHMARK_PASSWORD = "choose-a-private-passphrase"
```

The page accepts:
- a private ZIP containing de-identified tooth crops;
- the private manifest CSV;
- an optional same-case RF/SVM/PLS/ShadeGPT prediction CSV;
- an optional VITA 3D-Master reference CSV.

The default is 3 repeated GPT Vision calls per case. Results remain in the active Streamlit session until downloaded and are not committed to GitHub.

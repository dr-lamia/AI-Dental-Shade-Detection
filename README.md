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
- Added optional ChatGPT explanation. ChatGPT explains the deterministic result; it does not assign the shade.
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

## ChatGPT

If `OPENAI_API_KEY` is configured in Streamlit secrets, the app can produce a dentist- or patient-facing explanation of the already-computed CIELAB, ΔE00 and shade map.

Optional:

```toml
OPENAI_API_KEY = "..."
OPENAI_MODEL = "gpt-5.6-luna"
```

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

# Shade AI Research Validation

This folder is a research-validation branch for the existing **AI-Dental-Shade-Detection** project.

It is designed to answer a different question from the original clinical thesis:

> Can a standardized photographic color-analysis workflow reproduce instrument-based Rayplicker CIELAB measurements and longitudinal color change?

## Why this branch exists

The original prototype had three major problems for publication-quality validation:

1. VITA Classic was hard-coded to `A2`.
2. The 3D-Master lookup contained only seven approximate reference entries.
3. Color difference used simple Euclidean LAB distance instead of CIEDE2000.

This branch removes those shortcuts.

## What is implemented

- CIELAB conversion with `skimage.color.rgb2lab`
- robust trimmed-mean color extraction
- exclusion of very dark and specular/high-L pixels
- manual tooth ROI analysis
- cervical / middle / incisal thirds
- CIEDE2000 (ΔE00)
- optional measured shade-guide reference table
- batch pairing of clinical photographs with Rayplicker reference rows
- session-level multi-patch color calibration QC
- L*, a*, b* MAE and RMSE
- Lin concordance correlation coefficient
- Bland–Altman plots
- T0 → T4 longitudinal ΔE00 comparison
- exact and top-3 shade agreement when a valid measured shade reference library is supplied

## Current audited thesis data

The current audit identified:

- 28 tooth-specific Borea/Rayplicker folders across two patients.
- The uploaded numeric CSV contains 17 teeth and maps to the full-veneer teeth, with T0–T4 L*, a*, b*, shade labels and ΔE00.
- The thesis statistical analysis used 22 restorations (11 full and 11 sectional), but the currently uploaded numeric table is not the complete 22-restoration analysis table.
- Actual TRIOS shade/LAB export data have not yet been located in the supplied folders.

No clinical images or identifiable patient data are committed to this public repository.

## Run locally

```bash
cd research-validation
pip install -r requirements.txt
streamlit run app.py
```

## Batch filename convention

Use de-identified pre-cropped tooth images:

```text
P1_11_T0.jpg
P1_11_T4.jpg
P2_21_T0.jpg
P2_21_T4.jpg
```

The Rayplicker CSV must contain:

```text
patient_id,tooth_fdi,timepoint,ray_L,ray_a,ray_b
```

An optional `ray_shade` column may be added.

## Shade-reference table

For publication-quality shade-code matching, upload a CSV containing:

```text
shade,L,a,b
```

The LAB coordinates should be measured from the actual physical shade guide under the study's defined measurement conditions. The repository intentionally does **not** invent generic VITA LAB coordinates.

## Scientific safeguards

- Do not train or tune the algorithm on the same two thesis patients used for validation.
- Use the averaged Rayplicker values as the instrument reference.
- Do not treat Borea screenshot ΔE as interchangeable with the thesis ΔE00 unless the metric is verified.
- Multiple teeth from the same patient are clustered observations.
- Do not claim a three-way AI–Rayplicker–TRIOS comparison until true TRIOS color/shade outputs are recovered.
- Absolute shade estimation from ordinary photographs is sensitive to illumination, camera white balance, exposure, and calibration. A neutral reference or standardized imaging protocol materially strengthens the study.

## Status

**v1.0 research scaffold** — ready for de-identified T0/T4 paired validation after image-to-tooth mapping.


## Session calibration workflow

For a prospective or repeat session:

1. Lock camera body, macro lens, flash, aperture, shutter speed, ISO, white balance, distance, and magnification.
2. Photograph a multi-patch color reference under the same optical setup.
3. On the **Calibration QC** page, upload that image and a patch-specification CSV containing normalized patch boxes and reference CIELAB values.
4. Inspect the pre/post patch ΔE00 QC.
5. Freeze the exported session matrix before viewing the Rayplicker validation results.
6. Apply that matrix only to clinical images from the same session.

The supplied historical thesis photographs reviewed so far do not contain a visible calibration target, and sampled P1 images show exposure-setting changes within the same dated session. Accordingly, the historical photographs remain a feasibility dataset unless an in-session reference image can be recovered.

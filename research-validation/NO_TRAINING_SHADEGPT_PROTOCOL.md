# No-training ShadeGPT validation protocol

## Study concept

Validate the upgraded **ShadeGPT calibrated 3D-Master v2.0** against Rayplicker spectrophotometer outputs using the existing thesis material.

The index method deliberately avoids training a tooth-shade classifier or regressor.

## Index-test pipeline

1. Clinical photograph.
2. Record whether the image was truly cross-polarized at acquisition.
3. Apply color calibration before CIELAB whenever a valid neutral/multi-patch reference is available.
4. Select the tooth region of interest.
5. Suppress likely specular-highlight pixels in ordinary photographs.
6. Convert the calibrated image to CIELAB.
7. Estimate overall and regional L*, a*, b*.
8. Map each region to the nearest VITA 3D-Master reference by CIEDE2000.
9. Generate 3-zone, 9-zone, or detailed shade maps.
10. Optionally use ChatGPT to explain the already-computed output. ChatGPT does not assign the shade.

## Why this differs from the earlier pilot paper

The earlier pilot used a supervised Random Forest model to estimate Rayplicker-like CIELAB values and a separate classifier for direct VITA 3D-Master labels.

This protocol retains the clinically useful parts of that workflow:
- CIELAB as the primary continuous representation;
- CIEDE2000 agreement;
- nearest-reference VITA 3D-Master assignment;
- Rayplicker as the reference method;
- longitudinal color-change comparison.

It removes the trained shade/regression models.

## Rayplicker correspondence

Rayplicker is used as the reference because the thesis records its shade, L*, a*, b*, and color-change outputs.

Rayplicker also provides a polarized acquisition and regional shade mapping. The app mirrors the *structure* of that workflow by offering regional mapping, but it does not claim to reproduce Rayplicker's proprietary optics or algorithms.

## VITA 3D-Master reference table

Preferred reference:
- measured L*, a*, b* values from the physical VITA 3D-Master tabs under a documented setup.

Allowed exploratory fallback:
- median L*, a*, b* per shade computed from Rayplicker observations.

The fallback is deterministic reference construction, not machine-learning training. However, categorical shade accuracy must not be evaluated on the same observations used to construct those shade centroids.

## Primary comparison

For each defensibly paired tooth/timepoint:

- ShadeGPT calibrated L*, a*, b*
- Rayplicker L*, a*, b*
- ΔE00 between ShadeGPT and Rayplicker

Primary endpoint:
- mean/median ShadeGPT-versus-Rayplicker ΔE00

Secondary endpoints:
- L*, a*, b* absolute errors
- Bland-Altman agreement
- Lin concordance correlation coefficient
- exact 3D-Master shade agreement
- top-3 shade agreement
- agreement of T0-to-follow-up ΔE00 when the photo timepoint is verified

## Regional map comparison

If the Rayplicker export provides region-level shade labels that can be linked to the same tooth image:
- compare ShadeGPT and Rayplicker at the region level;
- report exact regional shade agreement;
- report adjacent/near-reference agreement by ΔE00;
- preserve the 3-zone / 9-zone / detailed resolution used for each comparison.

Do not infer regional labels from screenshots when the region identity is ambiguous.

## Historical-photo limitations

The historical clinical photographs currently reviewed do not contain a visible color target.

Therefore:
- uncalibrated historical-photo CIELAB is exploratory;
- glare suppression is not equivalent to optical polarization;
- an ordinary JPEG cannot be converted into a genuinely cross-polarized acquisition by software;
- photo-to-study-timepoint provenance must be verified before final paired analysis.

## ChatGPT role

ChatGPT receives only the already-computed colorimetry and shade-map summary.

It may:
- explain L*, a*, b*;
- explain the nearest 3D-Master match;
- summarize regional variation;
- produce dentist-facing or patient-facing text.

It must not:
- infer a different shade;
- overwrite deterministic CIELAB or ΔE00 values;
- create missing reference data;
- claim clinical accuracy beyond the validation results.

## Reporting terminology

Until independent clinical validation is complete, describe the system as:

**calibrated computer-vision-assisted photographic tooth shade analysis with optional generative-AI explanation**

Do not describe the deterministic shade-mapping component itself as a trained AI classifier.

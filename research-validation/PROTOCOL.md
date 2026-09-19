# Frozen Validation Protocol v1.0

## Study type
Retrospective paired pilot validation of a photographic color-analysis workflow against an instrument reference.

## Index method
Standardized clinical photograph → predefined tooth ROI → robust sRGB-to-CIELAB transformation → CIEDE2000 comparison.

A machine-learning tooth segmentation module may be added later, but if added it must be frozen before validation. Until then, describe the current implementation as **computer-vision-assisted photographic color analysis**, not as a trained shade-classification model.

## Reference method
Rayplicker spectrophotometer averaged L*, a*, b* values.

## Primary endpoint
ΔE00 between photographic L*a*b* and Rayplicker L*a*b* at the same tooth/timepoint.

## Secondary endpoints
- absolute L* error
- absolute a* error
- absolute b* error
- RMSE for each CIELAB coordinate
- Bland–Altman bias and 95% limits of agreement
- Lin concordance correlation coefficient
- exact shade agreement, only with a validated measured shade-reference library
- top-3 shade agreement
- agreement of ΔE00(T0→T4) between methods

## Initial validation set
Use the currently available full-veneer numeric dataset (17 teeth) with T0 and T4 clinical photographs where a defensible tooth-image match exists.

Maximum initial paired tooth-timepoint observations: 34.

The unit of clinical sampling is the patient. Teeth and timepoints are clustered/repeated observations and must not be described as independent patients.

## Data-lock rules
1. Do not fit model parameters to these two patients.
2. Do not select ROI rules after looking at which settings produce the best Rayplicker agreement.
3. Lock crop/segmentation rules before the final batch analysis.
4. Preserve all exclusions with reasons.
5. Keep raw patient images out of the public repository.
6. Use de-identified patient codes.
7. Preserve original Rayplicker numeric values without manual correction.

## TRIOS rule
TRIOS is not part of the primary validation until an actual shade/color result export (shade code and/or L*a*b*) is recovered and linked to tooth/timepoint. The presence of TRIOS scan files is insufficient.

## Sectional-veneer rule
Do not reconstruct the missing averaged sectional dataset from screenshots unless each screenshot can be tied to:
- tooth ID,
- timepoint,
- replicate/acquisition number,
- the documented triplicate averaging scheme,
- and the same ΔE00 definition used in the thesis.

## Reporting language
Until a trained segmentation/classification component is frozen and independently validated, use the term:

**computer-vision-assisted photographic tooth color analysis**

If a pretrained/frozen AI segmentation model is incorporated, the workflow may be described as:

**AI-assisted photographic tooth shade analysis with deterministic colorimetric estimation**

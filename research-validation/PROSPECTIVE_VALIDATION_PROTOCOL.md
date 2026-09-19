# Prospective calibrated validation protocol

## Objective

Validate an AI/computer-vision-assisted photographic tooth shade workflow against Rayplicker CIELAB measurements under a locked, reproducible imaging protocol.

The historical thesis cohort remains useful as a feasibility dataset. The definitive accuracy claim should come from a prospectively calibrated cohort with broader shade variation and true same-visit pairing.

## Acquisition unit

The patient is the clinical sampling unit. Multiple teeth may be included, but tooth-level observations are clustered within patient and must be handled accordingly in the statistical model.

## Same-session sequence

For every study session:

1. Perform prophylaxis/cleaning according to the clinical protocol.
2. Allow the tooth surface to reach the study-defined moisture state and keep that rule identical across participants.
3. Calibrate the Rayplicker according to the manufacturer's procedure.
4. Acquire three Rayplicker measurements per tooth and retain all three raw readings.
5. Photograph the color-calibration target under the exact camera/flash settings to be used for the teeth.
6. Acquire the clinical dental photograph without changing camera, lens, flash, white balance, magnification, or exposure.
7. Record the acquisition settings and file names immediately.
8. Repeat the calibration-target image if any setting is changed.

## Camera protocol

Use the same camera body and macro lens across the validation cohort whenever possible.

The historical dataset demonstrates that Canon EOS R + 100 mm macro + f/32 + flash + manual white balance is workable, but the exposure setting must be locked prospectively. Do not copy one historical shutter/ISO pair blindly: establish a well-exposed pilot using the calibration target and then freeze that setting for the study.

Record:
- camera body
- lens and focal length
- aperture
- shutter speed
- ISO
- white-balance mode/value
- flash system and power/mode
- magnification / working distance
- image file format
- whether any in-camera or post-processing was applied

Prefer RAW+JPEG acquisition when available, retaining RAW as the archival source.

## Calibration target

Photograph a validated multi-patch target in the same optical setup as the dental image.

Requirements:
- same camera and flash settings
- same session
- no clipped patches
- no specular glare on reference patches
- known reference CIELAB values for the target
- at least 6 valid patches for the current affine correction; >=18 is preferred
- preserve independent check patches when possible rather than fitting and evaluating on exactly the same patches

The app's **Calibration QC** page fits a session matrix and reports pre/post patch ΔE00. Freeze the matrix before evaluating the Rayplicker validation results.

## Tooth image standardization

- use a frontal orientation whenever possible
- keep camera-to-subject geometry consistent
- use retractors/black background consistently
- avoid saliva pools and strong specular highlights
- do not include a shade tab in the tooth ROI
- define the ROI rule before analyzing the reference results
- analyze the same anatomical region across methods; the middle third should be prespecified if the Rayplicker protocol represents that region

## Reference method

Rayplicker averaged L*, a*, b* values from three acquisitions are the primary reference.

Store the three raw readings as well as their mean. CIEDE2000 should be calculated from the paired mean color coordinates using one fixed implementation.

## Primary endpoint

ΔE00 between calibrated photographic CIELAB and averaged Rayplicker CIELAB for the same tooth at the same session.

## Secondary endpoints

- absolute error in L*, a*, b*
- RMSE for each coordinate
- Bland–Altman bias and 95% limits of agreement
- Lin concordance correlation coefficient
- exact shade agreement
- top-3 shade agreement
- clinically acceptable/perceptible proportions using thresholds prespecified in the manuscript
- longitudinal ΔE00 agreement when repeat sessions are available

## Mandatory negative controls

Because a narrow shade range can make average-shade guessing look accurate, report at least one leakage-free trivial comparator.

Current implementation:
- for each held-out patient, predict the mean Rayplicker L*a*b* from the remaining patient(s)

For the definitive larger cohort, replace this with grouped patient-level cross-validation and a training-set mean/majority-shade baseline.

The image method should materially outperform these baselines before claiming useful image-derived shade signal.

## Dataset split / leakage

If a trained tooth-segmentation or shade model is introduced:
- split at patient level
- never place teeth from the same patient in both training and validation/test sets
- freeze the final pipeline before the independent test set is opened
- keep calibration-target fitting independent of Rayplicker tooth outcomes

## Historical thesis cohort

Use only as:
- feasibility testing
- pipeline debugging
- data-linkage validation
- exploratory error characterization

Do not use the two historical patients to both tune the color correction and claim independent clinical validation.

## TRIOS

TRIOS should be added only if a real shade/LAB output tied to tooth and timepoint is recovered. The presence of an intraoral scan does not itself create a shade comparator.

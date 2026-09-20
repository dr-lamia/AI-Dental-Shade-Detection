# GPT Vision Batch Comparison Protocol

## Purpose

This protocol evaluates GPT Vision as an **additional independent multimodal comparator** for VITA 3D-Master shade estimation. It does not replace the calibrated ShadeGPT pipeline, Random Forest, RBF-SVM, or direct-image PLS models.

## Primary principle: blinding during GPT inference

For the GPT comparison arm, the model receives only:

- one de-identified tooth ROI image;
- the fixed list of permitted VITA 3D-Master shade labels;
- the locked research prompt.

The model must **not** receive:

- Rayplicker L*, a*, b* or shade;
- calibrated ShadeGPT L*, a*, b*, CIEDE2000 or shade;
- Random Forest, SVM or PLS predictions;
- any patient outcome or ground-truth information.

This keeps GPT Vision an independent comparison arm.

## Repeated runs

The default experiment performs **3 identical runs per tooth image** using the same model, prompt and image.

For each run, record:

- exact model identifier;
- predicted VITA 3D-Master shade;
- model-reported confidence;
- image-quality assessment;
- glare;
- blur;
- exposure;
- cervical/middle/incisal variation;
- notes;
- raw model response;
- UTC timestamp.

The raw response for every call is retained in a separate JSON file.

## GPT outcomes

### Primary categorical outcome

Exact VITA 3D-Master shade agreement with the Rayplicker recorded shade.

Report both:

1. run-level exact agreement; and
2. case-level consensus agreement.

A GPT case-level consensus shade is defined only when one shade receives at least 2 of 3 runs. If all three runs disagree, the consensus shade is left missing rather than forcing an arbitrary choice.

### Repeatability outcomes

Report:

- number of unique shades returned across repeated runs;
- unanimous agreement (3/3 identical);
- modal share;
- proportion of cases with unanimous predictions.

Model-reported confidence is descriptive only and must not be interpreted as a calibrated probability.

### Secondary color-difference outcome

When a valid VITA 3D-Master CIELAB reference table is available, the CIELAB coordinates of the **shade tab selected by GPT** may be compared with Rayplicker tooth L*, a*, b* using CIEDE2000.

This metric must be described as:

**ΔE00 between the reference CIELAB coordinates of the GPT-selected VITA tab and the Rayplicker tooth measurement.**

It must **not** be described as GPT-predicted L*, a*, b* or as direct GPT colorimetry.

## Comparison with ShadeGPT, RF, SVM and PLS

Other model predictions are supplied in a separate long-format CSV with:

- case_id
- method
- predicted_shade
- pred_L
- pred_a
- pred_b

Only predictions with an explicit matching `case_id` are merged.

The script intentionally does not align rows by position. This prevents accidental comparison of predictions from different datasets.

For methods that directly predict L*, a*, b*, report:

- L* MAE;
- a* MAE;
- b* MAE if calculated separately in the source analysis;
- direct predicted-Lab vs Rayplicker ΔE00;
- exact shade agreement when a categorical prediction exists.

For methods that only select a shade label, report:

- exact shade agreement;
- optional selected-tab-reference ΔE00.

## Dataset interpretation

The current 17-tooth T0 private working set contains observations from only two patients/source photographs. Results from this set are therefore a **pilot/feasibility comparison**, not independent clinical validation.

Do not combine summary metrics from the 17-tooth T0 set with metrics from the larger 342-image linked dataset as if they were one head-to-head test. A method can enter the common comparison only when prediction rows are available for the same explicit case IDs.

## Privacy

Clinical images are not committed to the public repository.

Run the benchmark locally with de-identified crops. Keep `OPENAI_API_KEY` in the local environment or a secret manager; never write it into a CSV, script, notebook, or Git commit.

## Reproducibility

The batch runner checkpoints after every API call. If execution stops, rerunning with the same output directory resumes completed successful case/run pairs by default.

The exact model identifier is stored with each prediction and in run metadata.

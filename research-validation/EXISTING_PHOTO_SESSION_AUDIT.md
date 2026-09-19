# Existing thesis photo-session audit

This audit records what can be supported from the supplied Drive material. It is not a claim that the photographed sessions are synchronized to a specific Rayplicker study visit.

## P1 post-treatment session sampled

- Camera: Canon EOS R
- Lens focal length in sampled EXIF: 100 mm
- Aperture: f/32
- Flash: fired
- White balance EXIF: manual
- Example early sequence: 1/50 s, ISO 100
- Later images from the same dated session: 1/200 s, ISO 400
- EXIF date sampled: 2025-03-23
- No gray card, ColorChecker, or shade tab was visible in the reviewed clinical images.
- No Canon CR3/RAW file was located in the supplied Drive search.

Implication: the acquisition was not exposure-locked across the whole session. Use only one fixed photograph for within-image tooth comparisons, or treat each exposure setting as a separate calibration session.

## P2 post-treatment session sampled

- Camera: Canon EOS R
- Lens focal length in sampled EXIF: 100 mm
- Aperture: f/32
- Exposure: 1/160 s in sampled images
- ISO: 200
- Flash: fired
- White balance EXIF: manual
- EXIF date sampled: 2025-10-08
- No gray card, ColorChecker, or shade tab was visible in the reviewed clinical images.
- No Canon CR3/RAW file was located in the supplied Drive search.

## Rayplicker file chronology visible in Drive

The top-level BOREA binary filenames currently visible for P1 are dated 2025-10-09; the P2 top-level BOREA binary filenames are dated 2025-10-10. Those filenames demonstrate that the folders contain at least one later Rayplicker acquisition session, but they do not by themselves identify which study timepoint is represented in the averaged CSV.

Therefore:

1. Do not label a clinical photograph T0/T1/T2/T3/T4 solely from the Drive folder name.
2. Confirm image-to-study-visit provenance from original clinical records before final paired analysis.
3. Use the averaged Rayplicker table as the numeric reference, not individual screenshot values.
4. The current historical photographs can support a feasibility analysis, but not a definitive calibrated color-accuracy claim unless a session reference target is recovered.

## Calibration decision

For the historical thesis images, the preferred hierarchy is:

1. Recover an in-session multi-patch target photograph, if it exists.
2. If unavailable, recover a neutral reference acquired in the same session.
3. If neither exists, retain the historical cohort as an uncalibrated feasibility dataset only.
4. Perform the definitive validation prospectively with locked camera/flash settings and a reference target photographed in every session.

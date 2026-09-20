# Published VITA 3D-Master reference set

This CSV contains the 26 non-bleached VITA 3D-Master shade-tab CIELAB reference coordinates used as an **independent deterministic mapping table** by the standalone ShadeGPT workflow.

Source provenance:
- Bayindir F, Kuo S, Johnston WM, Wee AG. Coverage error of three conceptually different shade guide systems to vital unrestored dentition. J Prosthet Dent. 2007;98(3):175-185.
- The same manufacturer laboratory values are reproduced as the manufacturer reference in a 2026 in-vitro comparison of DSLR/smartphone shade measurement.

Important:
- These coordinates are a published reference set, not a universal truth for every device/illuminant/geometry.
- They are preferable to learning shade labels from the two thesis patients because they keep the current app **no-training**.
- For the final clinical validation, the most defensible hierarchy remains:
  1. measured physical shade-guide values under the study setup;
  2. a published manufacturer/reference table such as this one;
  3. Rayplicker-derived centroids only as an exploratory fallback.
- Rayplicker may use its own internal proprietary conversion to VITA 3D-Master; exact shade agreement is therefore a comparison between systems, not proof that one table replicates Rayplicker's internal algorithm.

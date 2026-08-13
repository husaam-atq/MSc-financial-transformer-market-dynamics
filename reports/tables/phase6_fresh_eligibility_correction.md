# Phase 6 Fresh Eligibility Correction

The frozen fresh predictions were not recomputed or adapted. Eligibility was corrected by retaining only assets represented in all three frozen strict-window splits.

- Original rows/assets: 218/79.
- Strict rows/assets: 204/72.
- Excluded rows: 14 from 7 never-trained embedding IDs.
- Strict F1=0.0645, balanced accuracy=0.6600, ROC-AUC=0.6212, PR-AUC=0.0304.
- The archive remains underpowered and cannot confirm or refute generalisation.

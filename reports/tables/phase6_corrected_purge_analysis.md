# Phase 6 Corrected-Purge Analysis

## Decision

Retraining was necessary because the historical global purge of 10 dates allowed 120 training labels from 60 assets to cross into validation and 60 validation labels to cross into test. A boundary-specific audit showed that 18 global dates remove every measured crossing. Phase 6 retrained the exact three-seed Phase 4 model with only the purge changed, then trained the observed-session-corrected model separately.

## Isolated purge result

| Run | Train | Validation | Test | F1 | BA | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen purge 10 | 221,376 | 19,284 | 19,764 | 0.5618 | 0.7660 | 0.8009 | 0.4226 | 0.1143 |
| Legacy data, purge 18 | 220,980 | 18,828 | 19,764 | 0.5323 | 0.7475 | 0.8011 | 0.4206 | 0.1158 |

Ranking metrics were almost unchanged (ROC-AUC +0.0002; PR-AUC -0.0020), but validation-selected thresholding reduced F1 by 0.0296 and balanced accuracy by 0.0184. The defect did not create the pooled ranking result, but it affected decision metrics and invalidates any claim that the original split was fully purged.

## Corrected observed-session result

The corrected panel has 245,055 training, 20,494 validation and 21,514 test windows over 79 represented assets. The three-seed Transformer achieved F1 0.5114, balanced accuracy 0.7452, ROC-AUC 0.7891, PR-AUC 0.4029 and Brier 0.1109. This run changes both label eligibility and purge, so it is not an isolated purge comparison.

Historical results remain preserved as provenance. The corrected run is the authoritative Phase 6 diagnostic, not independent confirmation: the historical test had already influenced the research programme.

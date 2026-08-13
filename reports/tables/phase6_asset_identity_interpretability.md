# Phase 6 Asset-Identity Interpretability

Asset identity is the first interpretation result, not a nuisance footnote.

- The static training asset prior beats the corrected Transformer on ROC-AUC (0.8239 versus 0.7891), PR-AUC (0.4531 versus 0.4029) and Brier score (0.1033 versus 0.1109).
- Removing the explicit asset embedding lowers pooled ROC-AUC from 0.7891 to 0.7159 but does not improve pair-weighted within-asset ROC-AUC (0.4982 to 0.4783).
- A cyclic counterfactual ID swap changes probabilities by mean 0.0900 and median 0.0206. ROC-AUC falls to 0.6829. This is an out-of-distribution causal diagnostic, not a realistic market intervention.
- A linear probe recovers 79-way asset identity from the full model summary with 0.1669 test accuracy versus chance 0.0127. The no-ID representation falls to 0.0160.
- Family balanced accuracy is 0.3064 with ID and 0.2176 without ID versus chance 0.1667.

These results show that the model encodes and uses cross-sectional identity. They do not prove that every prediction is only a prior. The decisive complementary result is that pair-weighted within-asset ranking remains at chance.

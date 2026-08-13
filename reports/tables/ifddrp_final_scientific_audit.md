# IFDDRP Final Scientific Audit

Freeze date: 2026-08-08. Evidence-producing source commit: `e66470a6de35019cf8239eb4f8e6b8aab38e0bbb`.

## Verdict

The central numbers are reproducible from preserved predictions. One reporting convention required reconciliation: raw ensemble probabilities are authoritative for ROC-AUC, PR-AUC and within-asset ranking, while validation-selected calibrated probabilities are authoritative for Brier score, log loss and thresholded decisions. Isotonic ties explain the small historical differences (`0.7891/0.4982` versus raw `0.789814/0.491638`). No model rerun is required.

| check | finding | status |
| --- | --- | --- |
| Prediction reconstruction | Exact for 3 conditioned and 3 no-ID checkpoints | pass |
| Ranking metric basis | Raw ensemble scores are authoritative | reconciled |
| Probability metric basis | Validation-selected isotonic scores govern Brier/log loss and thresholds | reconciled |
| Stress target | t+1 through t+10 OR of terminal return, minimum path loss and volatility spike | pass_with_comparability_limit |
| Split | Fold 3, purge 18 global dates, embargo 1; no measured boundary crossing | pass |
| Scaling | Per-asset feature scaling fit on training endpoints only | pass |
| Final recovery experiments | A, B and C all failed preregistered promotion gates | valid_negative |
| Regime/latent gate | 0 of 14 regimes pass; latent change-point work locked | pass |

## Authoritative Transformer contract

- Universe: 80 configured daily instruments across six families; 79 are represented in the corrected split windows.
- Data: observed OHLCV sessions, not a forward-filled union calendar. Adjusted Close is preferred for targets; all 80 final target series had complete adjusted prices, so row-level Close fallback was not exercised.
- Origin and horizon: 60 observed sessions through market close at t; `target_stress_10d` uses only t+1 through t+10.
- Target: one when terminal ten-session return is at most -5%, the minimum future path relative to origin is at most -7%, or future realised volatility is at least twice trailing 20-session volatility scaled to ten sessions.
- Features: 34 train-scaled inputs (27 technical and 7 FRED macro/context series) plus 12 learned asset-embedding channels repeated over time. Raw OHLCV, family identity and missingness indicators are not direct inputs.
- Model: hidden width 128, two encoder layers, four heads, feed-forward width 256, sinusoidal positions to 1,024, temporal-attention pooling, LayerNorm/dropout 0.3/linear logit; 272,449 parameters.
- Training: soft-F1 loss, AdamW, learning rate 3e-4, weight decay 1e-4, batch 1,024, at most 12 epochs, seeds 7/42/123, validation-only early stopping/calibration/threshold selection.
- Split: train `2010-01-04 00:00:00` to `2023-11-24 00:00:00`, validation `2023-12-14 00:00:00` to `2025-02-23 00:00:00`, test `2025-03-15 00:00:00` to `2026-06-13 00:00:00`; purge 18, embargo 1. Window counts are 245,055/20,494/21,514.

## Reconciled central metrics

- Conditioned Transformer raw ranking: pooled ROC-AUC `0.789814`, PR-AUC `0.421056`, pair-weighted within-asset ROC-AUC `0.491638`.
- Conditioned calibrated probabilities: Brier `0.110920`, log loss `0.360548`.
- No-ID raw ranking: pooled ROC-AUC `0.715477`, within-asset ROC-AUC `0.472570`.
- Training-only priors: asset ROC-AUC `0.823906`; family ROC-AUC `0.816549`. Both exceed the learned Transformer in pooled ranking.
- Fixed cross-model panel: best learned pooled model was the MLP at `0.796478`; best learned within-asset point estimate was the MLP at `0.556967`; zero of five strict temporal-skill gates passed.

## Final bounded experiments

Experiment A selected an MLP family-prior residual on validation. Its test within-asset AUC was `0.525360`, but the paired lift and chronology conditions failed. Experiment B selected the Transformer BCE-plus-pairwise objective. Its test within-asset AUC was `0.561924` with interval `[0.492527, 0.625022]`; paired lift and chronology again failed. Experiment C selected the no-ID Transformer, whose equal-asset MAE `0.037816` was materially worse than the training asset mean `0.020896` and ridge `0.020691`; order effects were not established. All three are valid negatives.

## Interpretation and title

Exact checkpoint reconstruction passed. Mean zero-occlusion response was largest for macro/context (`0.160730`), followed by asset-embedding removal (`0.136454`) and returns/momentum (`0.129481`). The largest lag-block response was distant days 41-60 (`0.141292`). These are sensitivity diagnostics, not causal importance. A trained-versus-randomised attribution rank control failed; attention is supplementary only. Asset identity is materially decodable, but the head does not demonstrate useful within-asset timing. No regime passed the dynamic gate.

The fixed title is defensible only as an adversarial discovery: the apparent emergent dynamics mainly reflected static shortcut structure. It is not evidence that predictive emergent temporal dynamics were recovered.

## Defects and rerun decision

The purge-10 and union-calendar defects are historically preserved and superseded by the corrected run. Macro/context attribution is provenance-limited because the final seven FRED inputs are not a complete point-in-time vintage reconstruction. The test is repeatedly opened and all final historical results remain adaptive. The initial final-interpretability grouping mismatch was corrected with an exhaustive fixed 34-feature partition and a checkpoint-only rerun. No material defect remains that requires model retraining. Independent or prospective confirmation, not another historical architecture search, is the scientifically justified next empirical action.

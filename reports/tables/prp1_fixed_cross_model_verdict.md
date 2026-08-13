# PRP-1 Fixed Cross-Model Verdict

## Execution status

The frozen comparison completed all 30 registered model/identity/seed cells: 24 newly fitted cells and six immutable historical Transformer reconstructions. Evaluation produced 163 rows. Temporal falsification produced 90 per-seed and 30 ensemble rows. Identity swapping produced 15 rows. Asset/family probing produced 48 rows. Training/evaluation-stage failures: 0.

The empirical cells are complete, but strict protocol completion is not claimed. The smoke manifest covered only four of ten model/identity combinations; flattened logistic and every no-ID arm were omitted before execution proceeded. The post-execution audit also repaired the ignored temporal summary, computed the registered stride-10 sensitivity from existing predictions, and added artefact hashes. These qualifications do not change the negative scientific verdict.

This is opened, adaptive historical evidence for methodological falsification. It is not independent confirmation and cannot be promoted without a distinct-provider replication.

## Ensemble results

| model | identity variant | roc auc | pr auc | pair weighted within asset roc auc | per asset macro roc auc | f1 | balanced accuracy | brier score | degenerate prediction |
|---|---|---|---|---|---|---|---|---|---|
| mlp | asset_conditioned | 0.7965 | 0.4063 | 0.5570 | 0.6082 | 0.4964 | 0.7458 | 0.1129 | False |
| mlp | no_explicit_asset_id | 0.5250 | 0.1703 | 0.5447 | 0.6225 | 0.2808 | 0.5088 | 0.1358 | True |
| lstm | asset_conditioned | 0.6929 | 0.3113 | 0.5183 | 0.5734 | 0.3443 | 0.6220 | 0.1286 | False |
| lstm | no_explicit_asset_id | 0.7779 | 0.3940 | 0.4807 | 0.5624 | 0.4735 | 0.7322 | 0.1174 | False |
| tcn | asset_conditioned | 0.7759 | 0.3820 | 0.5475 | 0.5777 | 0.4498 | 0.7165 | 0.1217 | False |
| tcn | no_explicit_asset_id | 0.7326 | 0.3646 | 0.4710 | 0.4983 | 0.4233 | 0.6931 | 0.1272 | False |
| transformer_encoder | asset_conditioned | 0.7898 | 0.4211 | 0.4916 | 0.5387 | 0.5114 | 0.7452 | 0.1109 | False |
| transformer_encoder | no_explicit_asset_id | 0.7155 | 0.3629 | 0.4726 | 0.5356 | 0.4400 | 0.6843 | 0.1241 | False |
| flattened_logistic | asset_conditioned | 0.5762 | 0.1875 | 0.5122 | 0.5201 | 0.3076 | 0.5711 | 0.1347 | False |
| flattened_logistic | no_explicit_asset_id | 0.3987 | 0.1282 | 0.5321 | 0.5513 | 0.2870 | 0.5630 | 0.1354 | False |

## Static controls

| model | roc auc | pr auc | pair weighted within asset roc auc | f1 | balanced accuracy | brier score |
|---|---|---|---|---|---|---|
| static_global_prior | 0.5000 | 0.1613 | 0.5000 | 0.0000 | 0.5000 | 0.1359 |
| static_family_prior | 0.8165 | 0.4149 | 0.5000 | 0.0000 | 0.5000 | 0.1049 |
| static_asset_prior | 0.8239 | 0.4531 | 0.5000 | 0.4808 | 0.6865 | 0.1033 |

The strongest pooled model/variant was `mlp` / `asset_conditioned` at ROC-AUC 0.7965. The strongest ensemble within-asset point estimate was `mlp` / `asset_conditioned` at 0.5570. The train-only static asset prior reached 0.8239 pooled ROC-AUC.

## Frozen temporal-skill gate

| model | ensemble within asset auc | ensemble within auc ci lower | all seeds within auc above 0 5 | base temporal skill gate | all ensemble perturbation drops ge 0 02 with positive lower ci | all seed perturbation drops nonnegative | strict model gate pass |
|---|---|---|---|---|---|---|---|
| mlp | 0.5570 | 0.5048 | True | True | False | False | False |
| lstm | 0.5183 | 0.4600 | False | False | False | False | False |
| tcn | 0.5475 | 0.4922 | False | False | False | False | False |
| transformer_encoder | 0.4916 | 0.4295 | False | False | False | False | False |
| flattened_logistic | 0.5122 | 0.4789 | True | False | False | False | False |

Strict model passes: 0 of 5 (none). Sequence-family passes: 0 of 3. Cross-model recurrence required at least three of five models, including two of LSTM/TCN/Transformer. **Recurrence gate: FAIL.**

## Scientific decision

The registered evidence does not support recurring genuine temporal skill across model families. Pooled performance is not treated as temporal skill. Order sensitivity alone is not treated as useful predictive information. No architecture is protected, and Study B is not unlocked unless the recurrence gate passes.

The no-ID historical Transformer uses an inference-equivalent 34-to-46 input adapter with twelve immutable zero channels. It is an identity ablation, not a causal capacity-matched retraining contrast. All ranking metrics use raw scores; calibration and validation-selected thresholds affect probability and decision metrics only.

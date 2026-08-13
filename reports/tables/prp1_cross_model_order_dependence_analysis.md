# PRP-1 Cross-Model Order-Dependence Analysis

The analysis applies reverse order, one deterministic permutation and a half-window circular shift to every registered model, identity variant and seed without changing endpoint membership or labels. The table reports ensemble raw-score results. Date-block bootstrap intervals use 1,000 circular draws of 20 global dates.

| model | identity variant | method | original roc auc | perturbed roc auc | roc auc drop | original within asset roc auc | perturbed within asset roc auc | within asset auc drop | within asset auc drop ci lower | within asset auc drop ci upper | prediction spearman | mean absolute probability displacement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | asset_conditioned | reverse | 0.7965 | 0.7873 | 0.0091 | 0.5570 | 0.5088 | 0.0482 | -0.0161 | 0.1033 | 0.8262 | 0.1248 |
| mlp | asset_conditioned | deterministic_permutation | 0.7965 | 0.7905 | 0.0060 | 0.5570 | 0.5157 | 0.0413 | -0.0143 | 0.0898 | 0.8419 | 0.1151 |
| mlp | asset_conditioned | circular_shift | 0.7965 | 0.7783 | 0.0182 | 0.5570 | 0.5157 | 0.0413 | -0.0356 | 0.1122 | 0.7950 | 0.1318 |
| mlp | no_explicit_asset_id | reverse | 0.5250 | 0.4927 | 0.0323 | 0.5447 | 0.5197 | 0.0250 | -0.0322 | 0.0778 | 0.5065 | 0.0328 |
| mlp | no_explicit_asset_id | deterministic_permutation | 0.5250 | 0.4911 | 0.0338 | 0.5447 | 0.5182 | 0.0265 | -0.0196 | 0.0697 | 0.4266 | 0.0289 |
| mlp | no_explicit_asset_id | circular_shift | 0.5250 | 0.5081 | 0.0168 | 0.5447 | 0.5263 | 0.0184 | -0.0462 | 0.0786 | 0.3654 | 0.0377 |
| lstm | asset_conditioned | reverse | 0.6929 | 0.6766 | 0.0162 | 0.5183 | 0.5067 | 0.0116 | -0.0605 | 0.0804 | 0.6314 | 0.0339 |
| lstm | asset_conditioned | deterministic_permutation | 0.6929 | 0.6900 | 0.0029 | 0.5183 | 0.5167 | 0.0016 | -0.0486 | 0.0490 | 0.7346 | 0.0196 |
| lstm | asset_conditioned | circular_shift | 0.6929 | 0.6928 | 0.0000 | 0.5183 | 0.5061 | 0.0123 | -0.0529 | 0.0742 | 0.6883 | 0.0265 |
| lstm | no_explicit_asset_id | reverse | 0.7779 | 0.7249 | 0.0530 | 0.4807 | 0.4555 | 0.0252 | -0.0691 | 0.1266 | 0.5094 | 0.0934 |
| lstm | no_explicit_asset_id | deterministic_permutation | 0.7779 | 0.7549 | 0.0230 | 0.4807 | 0.4465 | 0.0342 | -0.0290 | 0.0965 | 0.6341 | 0.0619 |
| lstm | no_explicit_asset_id | circular_shift | 0.7779 | 0.7360 | 0.0418 | 0.4807 | 0.4290 | 0.0517 | -0.0148 | 0.1135 | 0.5039 | 0.0776 |
| tcn | asset_conditioned | reverse | 0.7759 | 0.7600 | 0.0159 | 0.5475 | 0.4935 | 0.0539 | 0.0030 | 0.1100 | 0.8491 | 0.0769 |
| tcn | asset_conditioned | deterministic_permutation | 0.7759 | 0.7692 | 0.0067 | 0.5475 | 0.5199 | 0.0275 | -0.0191 | 0.0782 | 0.8534 | 0.0805 |
| tcn | asset_conditioned | circular_shift | 0.7759 | 0.7759 | -0.0000 | 0.5475 | 0.5354 | 0.0120 | -0.0325 | 0.0557 | 0.8651 | 0.0736 |
| tcn | no_explicit_asset_id | reverse | 0.7326 | 0.6913 | 0.0413 | 0.4710 | 0.4676 | 0.0034 | -0.0445 | 0.0497 | 0.6563 | 0.0626 |
| tcn | no_explicit_asset_id | deterministic_permutation | 0.7326 | 0.6930 | 0.0396 | 0.4710 | 0.4681 | 0.0029 | -0.0384 | 0.0433 | 0.6851 | 0.0632 |
| tcn | no_explicit_asset_id | circular_shift | 0.7326 | 0.7132 | 0.0194 | 0.4710 | 0.4810 | -0.0100 | -0.0461 | 0.0326 | 0.7057 | 0.0565 |
| transformer_encoder | asset_conditioned | reverse | 0.7898 | 0.7908 | -0.0009 | 0.4916 | 0.5027 | -0.0111 | -0.0493 | 0.0247 | 0.9324 | 0.0481 |
| transformer_encoder | asset_conditioned | deterministic_permutation | 0.7898 | 0.7882 | 0.0016 | 0.4916 | 0.4897 | 0.0019 | -0.0309 | 0.0350 | 0.9437 | 0.0452 |
| transformer_encoder | asset_conditioned | circular_shift | 0.7898 | 0.7894 | 0.0004 | 0.4916 | 0.5010 | -0.0094 | -0.0399 | 0.0220 | 0.9415 | 0.0440 |
| transformer_encoder | no_explicit_asset_id | reverse | 0.7155 | 0.7255 | -0.0100 | 0.4726 | 0.4805 | -0.0080 | -0.0400 | 0.0228 | 0.8195 | 0.1094 |
| transformer_encoder | no_explicit_asset_id | deterministic_permutation | 0.7155 | 0.7188 | -0.0034 | 0.4726 | 0.4760 | -0.0034 | -0.0297 | 0.0189 | 0.8259 | 0.1073 |
| transformer_encoder | no_explicit_asset_id | circular_shift | 0.7155 | 0.7154 | 0.0000 | 0.4726 | 0.4757 | -0.0031 | -0.0343 | 0.0268 | 0.8153 | 0.1120 |
| flattened_logistic | asset_conditioned | reverse | 0.5762 | 0.5765 | -0.0002 | 0.5122 | 0.5162 | -0.0040 | -0.0413 | 0.0289 | 0.3819 | 0.3068 |
| flattened_logistic | asset_conditioned | deterministic_permutation | 0.5762 | 0.5990 | -0.0228 | 0.5122 | 0.5154 | -0.0032 | -0.0515 | 0.0548 | 0.1020 | 0.3504 |
| flattened_logistic | asset_conditioned | circular_shift | 0.5762 | 0.5783 | -0.0021 | 0.5122 | 0.5094 | 0.0028 | -0.0384 | 0.0475 | 0.3061 | 0.3280 |
| flattened_logistic | no_explicit_asset_id | reverse | 0.3987 | 0.3900 | 0.0087 | 0.5321 | 0.5454 | -0.0133 | -0.0497 | 0.0207 | 0.7088 | 0.0222 |
| flattened_logistic | no_explicit_asset_id | deterministic_permutation | 0.3987 | 0.3797 | 0.0190 | 0.5321 | 0.5488 | -0.0167 | -0.0686 | 0.0401 | 0.4334 | 0.0237 |
| flattened_logistic | no_explicit_asset_id | circular_shift | 0.3987 | 0.3926 | 0.0061 | 0.5321 | 0.5388 | -0.0067 | -0.0544 | 0.0418 | 0.6011 | 0.0215 |

Order dependence is interpreted jointly with within-asset skill. A model can react to reordering without forecasting usefully, and an off-manifold perturbation is not a causal explanation. The strict gate and cross-model recurrence decision are recorded in `prp1_fixed_cross_model_temporal_skill_gates.csv` and `prp1_fixed_cross_model_verdict.md`.

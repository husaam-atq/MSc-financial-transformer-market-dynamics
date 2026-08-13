# IFDDRP Open-Source Comparator Review

| implementation | scope | reproducibility | closest_use | missing_from_dissertation_question | url |
| --- | --- | --- | --- | --- | --- |
| PatchTST | patched channel-independent Transformer | official Apache-2.0 repository | architecture comparator | static label priors, within-asset ROC, identity swap/probe and full order controls | https://github.com/yuqinie98/PatchTST |
| iTransformer | variate-token Transformer | official MIT repository | alternative multivariate tokenisation | financial target-prior and grouped-ranking falsification | https://github.com/thuml/iTransformer |
| DLinear | simple linear LTSF baselines | official repository | strong simplicity control | identity/static-prior mechanism and financial grouped evaluation | https://github.com/honeywell21/DLinear |
| MASTER | market-guided stock Transformer | official code plus data caveats | closest financial Transformer | within-asset timing and static-prior/order falsification; repo discloses validation/test processing issue | https://github.com/SJTU-DMTai/MASTER |
| STID | MLP with spatial/temporal identity | official CIKM repository | identity-input precedent | tests identity as beneficial signal rather than harmful pooled shortcut | https://github.com/GestaltCogTeam/STID |
| GluonTS | probabilistic global forecasting library | mature Apache-2.0 package | global model and probabilistic baseline framework | no turnkey financial shortcut audit | https://github.com/awslabs/gluonts |
| NeuralForecast | 30+ neural forecasting models | active Apache-2.0 package | model implementations and fair baseline APIs | generic metrics do not supply the complete identity/order/static-prior bundle | https://github.com/Nixtla/neuralforecast |
| PyTorch Forecasting | TFT and neural forecasting utilities | active open-source package | TFT implementation | no bespoke multi-asset shortcut protocol | https://github.com/sktime/pytorch-forecasting |
| TimesFM | pretrained time-series foundation model | official Google Research repository | future zero-shot comparator | not justified on current opened panel without a new frozen protocol | https://github.com/google-research/timesfm |
| Time-Series-Library | broad advanced-model benchmark library | official research library | reference implementations | leaderboard breadth is not the scientific objective; project notes benchmark limitations | https://github.com/thuml/Time-Series-Library |

These projects provide credible architecture and forecasting implementations. None supplies the complete dissertation protocol: training-only global/family/asset priors, raw pooled plus pair-weighted within-asset ranking, identity removal/swap/probes, three endpoint-preserving sequence controls, and a controlled target-heterogeneity simulation. That absence supports an "apparently distinctive combination" claim, not first-ever novelty.

MASTER is the closest financial Transformer implementation, but its repository documents validation/test processing and data-reproduction caveats. Foundation-model or broader architecture comparisons would require a new preregistered dataset and are not justified as additions to the frozen historical dissertation evidence.

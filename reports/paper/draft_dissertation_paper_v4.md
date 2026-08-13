# Interpretable Transformer Models for Financial Time Series Forecasting: Discovering Emergent Market Dynamics

Author: Husaam Ateeq

Programme: MSc Data Science Dissertation

## Abstract

Financial forecasting models are often evaluated by pooling observations from many assets, but strong pooled performance may reflect persistent differences between assets rather than changes through time. This study trained a Transformer on 60-session windows from 79 instruments across six asset families to predict an adverse event during the next ten sessions. Chronological splits with explicit label-overlap controls, train-only preprocessing and validation-only calibration limited information leakage. The Transformer appeared strong when all test windows were pooled (ROC-AUC 0.790), yet a training-only asset-prevalence baseline scored 0.824 and the Transformer's within-asset ROC-AUC was 0.492. Removing or swapping identity changed predictions, whereas reversing, permuting or shifting temporal order barely changed pooled rankings. No learned model met the registered gate for robust ordered temporal skill. Controlled simulation then showed that heterogeneous asset priors alone can create high pooled discrimination and that the diagnostics recover deliberately planted temporal signal. The contribution is therefore a validated framework for separating cross-sectional shortcut learning from genuine temporal forecasting skill in multi-asset models, not a deployable forecaster.

Keywords: financial time series; Transformer; shortcut learning; pooled evaluation; within-asset ROC-AUC; model interpretability.

## 1 Introduction

Pooling many financial series gives a forecasting model more observations and exposes it to common market states. It also creates an evaluation trap. Suppose one asset has frequent adverse events and another has few. A score that is high for every observation of the first asset and low for every observation of the second can rank the pooled sample well without identifying a single high-risk date within either asset. A flexible model may learn this shortcut from an asset identifier, from scale or missingness fingerprints, or from persistent feature levels. Its pooled metric can then look like temporal forecasting skill even when the model does not use temporal order.

This distinction matters for financial machine learning. For a model intended to forecast changing risk through time, it must distinguish high- and low-risk periods within an asset rather than only rank assets by historical risk. Cross-sectional ranking may still be useful for a different screening objective. The distinction also matters scientifically because global forecasting models are commonly trained across related series. The benefits of global estimation are well established, but pooled evaluation can mix cross-sectional discrimination with within-series timing (Salinas et al., 2020; Montero-Manso and Hyndman, 2021). The problem becomes acute when target prevalence differs across assets or families.

This dissertation asks: **does a pooled multi-asset Transformer learn genuine temporal market dynamics, or does its apparent predictive performance arise mainly from static cross-sectional priors?** The empirical task predicts whether an instrument will experience a large loss, a severe interim drawdown or a volatility spike over the next ten observed sessions. The model sees 60 sessions of engineered price, return, volatility, trend, volume, drawdown and macroeconomic information. It is assessed chronologically on a broad panel of equities, bonds, commodities, foreign exchange, crypto assets and real-asset proxies.

The aim is to determine whether pooled multi-asset Transformer performance reflects time-varying market information or mainly stable differences among instruments. The objectives are to: (1) build and chronologically evaluate a leakage-aware forecasting framework; (2) separate pooled cross-sectional discrimination from within-asset temporal ranking using static priors and identity/order diagnostics; and (3) test whether the diagnosis is model-specific or methodologically reliable through a bounded model comparison and controlled simulation.

These objectives produce three contributions. First, the study implements explicit future-horizon labels, strict within-asset windows, train-only preprocessing, chronological validation, measured purging and validation-only calibration. Second, it combines static asset and family priors, within-asset evaluation, identity removal and swapping, representation probing and temporal-order perturbations to test what pooled discrimination measures. This is the primary empirical contribution. Third, it validates the diagnosis using 1,040 controlled simulations and a bounded five-model comparison. The simulations show both failure and recovery: static prevalence heterogeneity can inflate pooled ROC-AUC without temporal signal, while planted dynamics raise within-asset performance and make predictions sensitive to temporal destruction.

This is a supervised forecasting and model-evaluation study, not a trading strategy. It does not estimate transaction costs, portfolio returns or execution feasibility. Its practical uses are financial model validation and risk-model governance: deciding whether scores react to changing conditions, identifying misleading aggregate metrics and prioritising models or instruments for further review. The fitted Transformer is not proposed for deployment because no learned model passed the final gate for robust ordered temporal skill.

## 2 Related work

### 2.1 Global forecasting and entity identity

Global models estimate one forecasting function across many series rather than fitting one model per series. DeepAR shares an autoregressive recurrent network across related series to forecast full future distributions and can condition on series-level covariates (Salinas et al., 2020). Montero-Manso and Hyndman (2021) show more generally that a sufficiently expressive global model need not be more restrictive than a collection of local models; its parameter count also need not grow with the number of series. These results justify pooled estimation even for heterogeneous panels. Their usual evaluation target, however, is aggregate forecast error for future values. Such an error can establish useful prediction without answering whether a pooled binary classifier ranks changing states within each entity or persistent levels between entities.

Sirignano and Cont (2019) provide a closer financial comparison. They trained a universal recurrent model on billions of high-frequency order-book observations pooled across US equities to predict subsequent price-move direction. Performance transferred across time and to stocks excluded from training, while longer order-flow histories improved prediction. That is evidence for a common dynamic mapping and therefore a strong case for financial pooling. The present task differs in frequency, features and outcome: it classifies overlapping ten-session adverse events across six asset families whose event prevalences differ sharply. Transfer across unseen assets was not its design. A static prevalence control is consequently necessary before pooled AUC can be interpreted as temporal skill.

Entity information can also be useful rather than spurious. STID attaches spatial and temporal identity embeddings to a deliberately simple multivariate forecaster to resolve indistinguishable samples, then evaluates aggregate forecasting error (Shao et al., 2022). MASTER learns temporal and cross-stock representations guided by market information for joint stock forecasting (Li et al., 2024). These studies demonstrate why node identity or cross-entity context may improve a valid prediction objective. They do not ask whether a constant entity prevalence can explain pooled classification, nor do they compare rankings before and after destroying temporal order. This dissertation therefore does not prohibit identity. It separates identity-conditioned cross-sectional ranking from dynamic ranking within the same asset.

### 2.2 Transformers and strong simple baselines

The Transformer replaces recurrence with self-attention over a sequence (Vaswani et al., 2017). Time-series variants alter how temporal positions, variables and static information enter that mechanism. Temporal Fusion Transformers combine recurrent local processing, attention, gating, variable selection and static covariates for multi-horizon probabilistic forecasting (Lim et al., 2021). MASTER instead focuses attention on interactions among stocks and market states. Both illustrate plausible routes by which attention can use temporal or cross-entity context. Neither architecture guarantees that a fitted score depends on chronological order, and attention weights alone do not establish the mechanism used.

Simple controls remain essential because architecture complexity is not evidence of temporal information. DLinear decomposes a series and applies simple linear mappings; it outperformed several elaborate Transformers on standard long-horizon benchmarks (Zeng et al., 2023). Those benchmarks concern future-value errors rather than heterogeneous event priors, but the result motivates a broader principle: the strongest simple comparator should match the claim being tested. Here that set includes not only a flattened logistic model, multilayer perceptron (MLP), long short-term memory network (LSTM) and temporal convolutional network (TCN), but also training-only asset and family prevalences. The bounded five-model comparison tests whether the diagnosis recurs across model classes; it is not an architecture search.

### 2.3 Evaluation, shortcuts and interpretation

Time-series validation must preserve order because random splits can leak future information and misrepresent performance under non-stationarity (Cerqueira et al., 2019). Financial event labels add a separate overlap problem: neighbouring forecast origins can share future returns. Purging removes earlier labels whose future intervals cross an evaluation boundary, whereas embargoing adds a gap after that boundary (López de Prado, 2018). This study audits the actual interval endpoints rather than assuming that a nominal horizon-sized gap is sufficient across mixed trading calendars.

The metric estimand also matters. ROC-AUC is the probability that a randomly selected positive receives a higher score than a randomly selected negative. A pooled multi-asset AUC combines two kinds of pair: dates from the same asset and dates from different assets. If event prevalence differs by asset, many between-asset pairs can be ordered correctly by a score that never changes through time. Covariate-adjusted ROC analysis shows more generally that discrimination can change after conditioning on a covariate (Janes and Pepe, 2009). Here asset identity is the conditioning variable. Pair-weighted within-asset AUC removes between-asset pairs, while static priors quantify how much pooled ranking is available from training prevalence alone.

The broader machine-learning literature defines shortcut learning as reliance on predictive signals that satisfy a benchmark but not the intended mechanism (Geirhos et al., 2020). Representation probes can reveal whether identity is recoverable, although probe accuracy alone does not prove how that information is used (Hewitt and Liang, 2019). Attention weights should not be treated automatically as explanations (Jain and Wallace, 2019), and attributions should respond to model-parameter randomisation if they are to support model-specific interpretation (Adebayo et al., 2018). The safe novelty claim is therefore narrow: this dissertation combines training-only static priors, within-asset metrics, identity ablation and probing, endpoint-preserving temporal perturbations, cross-model comparison and controlled simulation to distinguish cross-sectional shortcut learning from temporal skill in a heterogeneous financial panel. It does not claim that any individual diagnostic is new. Table 1 locates that combined design against the closest comparators.

| Work | Relevant design | Diagnostic gap relative to this study |
| --- | --- | --- |
| DeepAR; global/local theory | future-value forecasting with parameters shared across related series | aggregate errors do not isolate within-series event timing |
| Sirignano and Cont | pooled order books predict price-move direction, including unseen stocks | no heterogeneous long-horizon event-prior control |
| STID | explicit node/time identities improve multivariate forecast errors | no static-prior or within-entity ranking test |
| DLinear | linear mappings challenge Transformers on long-horizon errors | no pooled entity-prevalence problem |
| MASTER | market-guided Transformer models temporal and cross-stock structure | no training-prior or order-destruction control |
| This dissertation | pooled event prediction plus conditional, identity and order attacks | historical test evidence is adaptive; independent confirmation remains open |

The closest strands therefore answer complementary questions rather than the question posed here. Global-forecasting work asks whether shared estimation can improve future-value prediction; STID and MASTER ask how identity or cross-entity context can improve an aggregate objective; DLinear asks whether architectural complexity beats a strong simple predictor. Shortcut and conditional-ROC work instead ask what a successful metric is measuring. This dissertation joins those strands at the evaluation stage. It asks whether improvement in a pooled financial event metric survives a baseline that contains only entity prevalence, an estimand restricted to within-entity pairs, interventions on identity, and attacks on temporal order. Its contribution is this combined, simulation-validated falsification design, not a claim that pooled models, identity embeddings, AUC conditioning or temporal perturbations are individually new.

The comparison also separates transfer from timing. DeepAR and the global/local framework justify estimating common parameters when related series share structure, but their usual aggregate forecast-error objective can improve even if the model mostly learns persistent differences in scale or seasonality. Sirignano and Cont offer stronger evidence of transferable financial dynamics because one model generalised across order books and to stocks excluded from training, with additional order-flow history improving prediction. Their outcome, frequency and entity support differ from the present ten-session event task, however, so that result does not remove the need for a prevalence-only control here. Conversely, STID makes identity an intentional predictive input and MASTER uses market-wide context; both are legitimate designs when the estimand rewards cross-entity structure. The issue is not whether identity is present, but whether the reported metric distinguishes its static contribution from changing state.

DLinear warns that a sophisticated sequence model should not be credited when a simpler model matches the relevant objective. Here that principle applies to information rather than parameter count: the strongest baseline is deliberately non-temporal. A learned model must beat static group information, rank events conditionally within an asset and depend coherently on the ordering it claims to model. This combined test is the closest-work gap addressed below.

## 3 Methodology

### 3.1 Data, forecast origin and inputs

The configured daily universe contained 80 current instruments from six families: 39 equities, 11 bonds, eight commodities, six foreign-exchange instruments, 13 crypto assets and three real-asset proxies. Seventy-nine instruments formed valid windows in the corrected final split. The panel spans 4 January 2010 to 23 June 2026 where history is available and contains 306,174 observed open-high-low-close-volume (OHLCV) rows. Calendar-aligned placeholders were excluded from model windows; ETF weekend prices were not forward-filled to imitate the continuous crypto calendar. This avoids treating stale prices as observations.

Yahoo Finance supplied open, high, low, close, adjusted close and volume. Adjusted close was preferred for returns and the target because it accounts for distributions and splits; all 80 final target series had complete adjusted prices, so the row-level close fallback was not used. Raw open, high, low and close were retained only to construct intraday range, open-close and overnight-gap variables. Raw price levels did not enter the network directly.

At each observed close t, the model received 60 within-asset sessions ending at t:

> X_{i,t} = [x_{i,t-59}, ..., x_{i,t}], where L = 60.

Each x contained 34 train-scaled channels: 27 engineered market channels and seven macro/context channels. The market channels covered six price/return variables, six volatility variables, six momentum/trend variables, four volume/activity variables and five stress/drawdown variables. The seven context channels were the federal funds rate, 2-year and 10-year Treasury yields, the 10y-2y slope, the VIX equity-volatility index, the US high-yield spread and a broad dollar index. Federal Reserve Economic Data (FRED) observations were lagged by at least one business day before alignment. A separate 12-dimensional learned asset embedding was repeated at each timestep, giving the Transformer 46 channels after concatenation. No family label or explicit missingness indicator was supplied. Table 2 enumerates every direct input.

| Group | Direct channels |
| --- | --- |
| Price/return (6) | close return; log return; open-close return; high-low range; overnight gap; intraday range proxy |
| Volatility (6) | 5-, 10- and 20-session volatility; downside volatility; rolling high-low range; EWMA volatility |
| Momentum/trend (6) | 5- and 20-session cumulative return; moving-average distance; 5/20 crossover; RSI; MACD |
| Volume/activity (4) | volume change; 20-session volume z-score; volume/average ratio; price-volume interaction |
| Stress/drawdown (5) | 60-session drawdown; distance from 20- and 60-session highs; negative-return streak; downside indicator |
| Macro/context (7) | DFF; DGS2; DGS10; T10Y2Y; VIXCLS; BAMLH0A0HYM2; DTWEXBGS |

For each asset, a median imputer and standard scaler were fitted only on its training dates, then applied unchanged to validation and test dates. A feature with no observed training values became zero after scaling. There was no backward filling from future observations. Windows never crossed assets or split boundaries. This design answers the forecast-origin question directly: the Transformer saw engineered information available by asset i's close at t, together with an identity embedding, and nothing from t+1 onward. Figure 1 summarises the forecast construction and corrected chronological split.

[[FIGURE:methodology]]

### 3.2 Ten-session adverse-event target

Let P(i,t) be adjusted close and r(i,t) = log[P(i,t) / P(i,t-1)]. For horizon H = 10, the implementation computes:

> R_{i,t}^{(10)} = P_{i,t+10}/P_{i,t} - 1,

> D_{i,t}^{(10)} = min_{1 <= k <= 10}(P_{i,t+k}/P_{i,t} - 1),

> V_{i,t}^{future} = sqrt(sum_{k=1}^{10} r_{i,t+k}^2), and V_{i,t}^{hist} = sqrt(10) sd(r_{i,t-19:t}).

The binary label is:

> y_{i,t} = 1{R_{i,t}^{(10)} <= -0.05 OR D_{i,t}^{(10)} <= -0.07 OR V_{i,t}^{future} >= 2 V_{i,t}^{hist}}.

Thus a positive denotes at least one of three events after t: a 5% endpoint loss, a 7% interim drawdown, or realised volatility at least twice its trailing scale. Only t+1 through t+10 enter the label. This composite target captures adverse conditions, but it is not equally rare or equally severe across families. Validation prevalence ranged from 1.74% for bonds to 46.80% for crypto. That heterogeneity is central to the later falsification and limits cross-family interpretation.

### 3.3 Chronological split and leakage controls

The corrected fold used train dates from 4 January 2010 to 24 November 2023, validation dates from 14 December 2023 to 23 February 2025, and test dates from 15 March 2025 to 13 June 2026. Strict windowing produced 245,055 training, 20,494 validation and 21,514 test examples. Validation selected training duration, calibration and the decision threshold. Test labels were not used for those choices.

Neighbouring ten-session labels can share future returns. Purging removes observations before a boundary when their label intervals overlap the following split. Embargoing drops the first date after a boundary to reduce residual dependence. A horizon-based purge of ten global dates still left 120 training and 60 validation labels crossing the boundaries. The measured correction used purge 18 and embargo one; a direct interval audit then found zero crossings. The gaps are deliberately longer than the ten-session horizon because instruments follow different observed-session calendars.

### 3.4 Transformer and training protocol

The 46-channel sequence was projected to width 128 and combined with fixed sinusoidal positional encodings. The encoder had two pre-normalised layers, four attention heads, a 256-unit feed-forward block, GELU activation and dropout 0.3. Temporal-attention pooling reduced the 60 hidden states to one vector, followed by layer normalisation, dropout and a linear logit head. The complete model had 272,449 parameters.

Training used a differentiable soft-F1 loss, AdamW with learning rate 3 x 10^-4 and weight decay 10^-4, batch size 1,024, gradient clipping and at most 12 epochs. Early stopping patience was three. Seeds 7, 42 and 123 produced a probability ensemble. Calibration method and the F1 decision threshold were selected on validation only. The model was trained with mixed precision where supported.

Ranking and probability metrics use different score objects. Raw ensemble probabilities are authoritative for ROC-AUC, precision-recall AUC (PR-AUC) and within-asset ROC-AUC because post-hoc isotonic calibration can create ties and alter rankings. Validation-selected isotonic probabilities are authoritative for Brier score, log loss and threshold-based F1 or balanced accuracy. This convention prevents calibration-induced ties from being mistaken for changes in ranking performance.

### 3.5 Metrics, baselines and falsification tests

Pooled ROC-AUC measures threshold-independent ranking across all test windows, but can reward between-asset prevalence differences. Within-asset ROC-AUC asks whether high-risk dates rank above low-risk dates for the same asset. For each asset with both classes, AUC_i was weighted by its number of positive-negative pairs:

> AUC_within = sum_i(n_i^+ n_i^- AUC_i) / sum_i(n_i^+ n_i^-).

This pair-weighted statistic uses the same pair interpretation as pooled AUC while removing between-asset pairs. Per-asset macro AUC gives every eligible asset equal weight. PR-AUC is reported because positives are imbalanced. Balanced accuracy weights sensitivity and specificity equally; F1 combines precision and recall at the frozen threshold. Brier score and log loss assess probability accuracy, with log loss penalising confident errors more strongly.

These metrics answer different questions and are not interchangeable. Pair weighting estimates discrimination over all eligible same-asset positive-negative pairs, so assets with greater class support contribute more. Macro AUC asks whether performance is broad by giving each eligible asset equal weight, but becomes noisier for assets with few events. Pooled AUC asks a cross-sectional-and-temporal mixture question. Reporting all three prevents a favourable aggregation choice from silently defining success. Threshold metrics and calibration remain useful operational descriptions, but a validation-selected threshold cannot turn a date-invariant score into temporal ranking skill; that is why the static-prior and within-asset comparisons precede thresholded results in the interpretation.

The training-only asset prior assigns every validation or test date for an asset the same prevalence estimated from that asset's training labels. The family prior does the same at the coarser family level. Because these baselines are constant through time within an asset or family, any pooled discrimination they achieve arises from cross-sectional differences in training prevalence rather than temporal ranking. The asset prior's within-asset AUC is therefore 0.5 by construction whenever both classes are present. The no-ID Transformer removes the explicit asset embedding. A cyclic ID swap measures prediction sensitivity to incorrect identities, while a regularised representation probe tests whether the trained latent vector encodes the asset.

Three endpoint-preserving perturbations attack temporal dependence without changing the asset or final forecast origin: reverse the 60 steps, apply one deterministic permutation, or circularly shift the sequence. A genuinely order-dependent model should change its rankings materially. These attacks test reliance on the ordered rows, not whether every input is time-varying: rolling volatility, momentum, trend and drawdown channels already summarise parts of the historical path. A registered empirical gate required support from paired within-asset improvement, chronology sensitivity and uncertainty controls; it was evaluated across the Transformer, MLP, LSTM, TCN and flattened logistic model.

The gate deliberately separates three questions that a single point estimate cannot answer. First, is conditional ranking above chance with dependence-aware uncertainty? Second, is any lift stable across seeds and paired observations rather than driven by composition? Third, does performance deteriorate when the chronological information claimed to matter is destroyed? A model was promoted only when these questions supported the same interpretation of ordered temporal skill. This criterion is stricter than reporting an ensemble AUC above 0.5, but the simulation below tests whether it still permits recovery when ordered signal truly exists.

Finally, a controlled simulation varied asset-prior heterogeneity, dynamic signal strength and persistence. It compared pooled and within-asset AUC and repeated temporal perturbations over 20 seeds per condition. Across registered conditions there were 1,040 runs and no execution failures. Simulation provides a known data-generating process: it tests whether the diagnostics remain at chance without dynamics and respond when ordered signal is deliberately planted.

## 4 Results

### 4.1 Apparent pooled performance was cross-sectional

The asset-conditioned Transformer achieved raw pooled ROC-AUC 0.789814 and PR-AUC 0.421056 over 21,514 test windows. Its validation-selected probabilities gave Brier score 0.110920, log loss 0.360548, F1 0.511420 and balanced accuracy 0.745220. Considered alone, those values suggest useful discrimination.

The static controls changed that interpretation. The training-only asset prior achieved pooled ROC-AUC 0.823906 and the family prior 0.816549. Both exceeded the Transformer despite being constant through time within each group. The asset prior is a particularly strong falsification control: it assigns one training-period prevalence to every date of the same asset, so its within-asset ROC-AUC is exactly 0.500000 by construction. Its pooled AUC of 0.823906 therefore arises entirely from cross-sectional differences in event prevalence, not from identifying changing risk dates.

The reason is visible from the pair interpretation of AUC. Pooled AUC compares every positive test window with every negative test window. Some comparisons involve two dates from the same asset, but many involve different assets. When, for example, positives occur much more often in one family than another, a constant asset score can order many between-asset pairs correctly. Within-asset AUC discards those between-asset comparisons and asks the intended timing question only: within a fixed instrument, did the model score its adverse dates above its non-adverse dates? The Transformer's pair-weighted within-asset AUC was 0.491638, effectively chance, while its equal-asset macro AUC was 0.538743. It separated historically high-prevalence assets from low-prevalence assets far better than it timed adverse periods within an asset. Figure 2 makes this contrast explicit.

[[FIGURE:core_results]]

Table 3 shows that the pattern was not specific to one architecture. The MLP had the strongest learned pooled AUC (0.796478) and within-asset estimate, 0.556967 with dependence-aware interval [0.504831, 0.600824]. This supports a modest within-asset predictive association, but not by itself robust ordered temporal skill. The registered gate also required material AUC loss after temporal destruction, positive ensemble uncertainty support and non-negative seed-level losses; those chronology conditions were not satisfied. Other models had intervals crossing 0.5 or also lacked order sensitivity. No learned model passed the complete gate: 0/5. This does not make temporal prediction impossible; it distinguishes conditional association from ordered forecasting.

| Model | Pooled ROC-AUC | Within-asset ROC-AUC |
| --- | ---: | ---: |
| Static asset prior | 0.823906 | 0.500000 |
| MLP | 0.796478 | 0.556967 |
| Transformer encoder | 0.789814 | 0.491638 |
| TCN | 0.775881 | 0.547453 |
| LSTM | 0.692870 | 0.518344 |
| Flattened logistic | 0.576220 | 0.512171 |

The MLP illustrates why the gate is conjunctive. Its interval supports modest conditional discrimination; chronology tests ask whether that association requires the ordered 60-session path. Failure of the chronology gate means that the association could not be attributed robustly to that path, not that all changing-state information was absent: engineered rolling features already encode time-varying summaries. Conversely, an order-sensitive model with chance-level within-asset AUC is also insufficient. Robust ordered temporal skill requires both conditional discrimination and coherent chronology dependence. None of the five models supplied that combination.

The MLP is therefore the strongest learned conditional ranking and warrants independent replication. Its interval addresses uncertainty around one estimand, not paired robustness or chronology. The within-asset-objective Transformer is also numerically interesting at 0.561924, but its interval crosses 0.5 and its uncertainty, paired-lift and chronology requirements were not met. Neither result establishes robust ordered temporal forecasting under the registered rule.

### 4.2 Identity mattered; chronological order barely did

Removing the 12-channel asset embedding reduced pooled AUC from 0.789814 to 0.715477 and did not improve within-asset AUC, which fell to 0.472570. This ablation isolates the explicit identifier, but not all implicit identity. Persistent volatility, trading calendar, volume scale and feature missingness can still reveal which asset generated a window. The decline therefore shows that explicit identity contributed to pooled ranking; the absence of within-asset recovery shows that removing it did not expose a hidden temporal forecaster.

Cyclically swapping asset IDs changed the mean absolute predicted probability by 0.090025 and reduced pooled AUC to 0.682928 while leaving each market-history window fixed. This controlled mismatch demonstrates score sensitivity to the supplied identity, although it does not establish that identity was the model's only shortcut. A probe decoded the asset from the conditioned latent representation with test accuracy 0.166930, more than thirteen times the approximate 1/79 = 0.0127 chance rate. Probe success establishes decodability, not functional use; the ablation and swap provide the complementary behavioural evidence. Together the three diagnostics support identity dependence without converting that dependence into a causal market explanation.

Temporal destruction produced the opposite response. Reversal, deterministic permutation and circular shift yielded pooled AUCs 0.791263, 0.788585 and 0.790321. Each was within 0.0015 of the authoritative unperturbed raw AUC in absolute terms. Reversal tests directional chronology, deterministic permutation breaks local order consistently, and circular shift disrupts phase while retaining all rows. These attacks preserve endpoint or marginal information to different degrees, and engineered rolling channels already summarise parts of the past. The small changes therefore do not prove that no time-varying information existed. They do, however, weaken the claim that the original ordered 60-session path drove the observed pooled ranking.

Feature-group occlusion reinforced the identity diagnosis but was not treated as a faithful explanation. Removing macro/context inputs produced mean prediction displacement 0.160730; masking days 41-60 produced 0.141292; removing the asset embedding produced 0.136454; and masking returns/momentum produced 0.129481. Following model-randomisation sanity-check logic (Adebayo et al., 2018), the trained-versus-randomised attribution rank check failed its threshold. These values are therefore sensitivity diagnostics, not evidence that macro variables were causally or definitively the most important features. None of the 14 predefined regimes passed the temporal-dynamics gate, so latent-state interpretation remained locked.

### 4.3 Simulation validated the diagnosis

The simulation created two controlled worlds in which the data-generating mechanism was known. World A contained heterogeneous asset event priors but no dynamic signal. As prior heterogeneity increased, static asset-prior pooled AUC rose by 0.400021 (95% interval 0.391244 to 0.408797). In the core persistent setting it moved from approximately 0.499 to 0.899, even though the classifier's within-asset AUC remained 0.500086 (0.497032 to 0.503141). This reproduces the empirical concern in isolation: cross-sectional prevalence differences are sufficient to create apparently excellent pooled discrimination.

World B removed prior heterogeneity and planted an ordered, persistent temporal signal. Under strong signal, within-asset AUC rose to 0.783997 (0.781142 to 0.786852). Reversing the sequence then reduced AUC by 0.380991, while deterministic permutation reduced it by 0.102297. Every registered mechanism gate passed. These are not estimates of real market effect sizes; they test the behaviour of the diagnostics under controlled conditions.

This distinction matters for the empirical interpretation. A weak real-data response to temporal perturbation could otherwise mean that the tests were too severe, badly designed or simply unable to detect sequence dependence. The simulation rejects that explanation within the tested mechanism: the diagnostics stayed at chance when only static heterogeneity existed, then recovered within-asset skill and order sensitivity when genuine temporal information was present. Figure 3 displays both worlds and links the high empirical pooled AUC to a concrete failure mechanism rather than to a visual analogy.

Simulation does not prove that real markets follow this synthetic process, nor that the simulated effect sizes transfer to finance. Its role is narrower and important: mechanism sufficiency and diagnostic calibration. It shows that prior heterogeneity alone can produce the qualitative metric pattern observed empirically, and that the same evaluation suite does not automatically reject every sequence model. The empirical and simulated results therefore support each other asymmetrically: market data motivate the shortcut diagnosis, while simulation verifies that the proposed mechanism can generate it and that planted dynamics would have produced a different diagnostic signature.

The two panels should therefore be read as a controlled contrast, not as two forecasts of market performance. In panel A, increasing heterogeneity changes only which assets tend to be positive; it does not create useful ordering of dates within an asset. The widening gap between pooled prior AUC and within-asset AUC is the shortcut mechanism. In panel B, prior heterogeneity is removed and ordered signal is increased. Conditional discrimination then rises and destroying order becomes costly. Recovery along both axes matters: it demonstrates that the diagnostics distinguish the planted worlds rather than merely penalising pooled models by design.

[[FIGURE:simulation]]

### 4.4 Recovery checks did not overturn the finding

Three targeted redesigns tested plausible alternative explanations for the negative timing result. The first separated a static prevalence component from a learned dynamic residual. The second aligned part of the training objective directly with within-asset ranking. The third replaced the binary composite event with a continuous downside outcome. Table 4 reports the validation-selected historical test results.

| Recovery test | Main result | Interpretation |
| --- | --- | --- |
| Static + dynamic residual | within AUC 0.525360; interval [0.443706, 0.602033] | paired-lift and chronology gates failed |
| Within-asset objective Transformer | within AUC 0.561924; interval [0.492527, 0.625022] | uncertainty, paired-lift and chronology gates failed |
| Continuous downside Transformer | equal-asset MAE 0.037816 | asset mean 0.020896 and ridge 0.020691 were stronger |

The residual model asks whether a static component had hidden a smaller dynamic signal; its interval crossed chance and the complete gate failed. The within-asset objective addresses training-objective mismatch directly, but its interval also crossed 0.5 and neither paired lift nor chronology evidence supported promotion. The continuous target addresses the possibility that binarisation caused the failure, yet simple training-mean and ridge baselines had substantially lower error. These tests are secondary negative controls, not equal-status headline experiments. Together they show that static/dynamic decomposition, objective alignment and target continuity did not overturn the main conclusion under the bounded designs.

Each recovery test removes one plausible explanation: residualisation separates static prevalence, within-asset optimisation aligns the objective with conditional ranking, and continuous downside removes the event threshold. Their failures are informative but bounded. They reject these implementations on this opened historical period, not every possible decomposition, ranking loss or continuous outcome, and they are not independent confirmation.

## 5 Discussion and conclusion

The central finding is a measurement result. A pooled Transformer ROC-AUC near 0.79 did not establish temporal forecasting skill. The stronger static prior, chance-level within-asset AUC, identity sensitivity and lack of material order sensitivity show that the pooled score mainly rewarded persistent cross-sectional structure. Target prevalence varied sharply across families, so between-asset positive-negative pairs dominated an estimand that appeared dynamic but could be ranked from training prevalence alone. Under known data-generating conditions, the controlled simulation reproduced this qualitative pattern and responded when ordered information was deliberately planted. It is a mechanism test, not external market replication or an estimate of real-market effect sizes.

This does not imply that global forecasting, identity embeddings or Transformers are invalid. A series identifier can calibrate persistent scale, seasonality or baseline risk, while pooled estimation can transfer genuinely shared structure. The problem is interpretive: identity conditioning becomes a shortcut relative to a temporal claim when the evaluation rewards stable entity differences and those differences are reported as changing-state skill. Cross-sectional ranking may support screening when the objective is to compare unconditional risk across assets; pooled AUC and identity are then relevant. If the objective is to detect changing risk within an asset, static-prior comparison, conditional ranking and temporal perturbation are indispensable. In this study the second objective was intended, and it was not supported.

The title is treated as an empirical question rather than a presupposed positive result. Transformer modelling and financial time-series forecasting are strongly supported. Interpretability is partial: ablations, probes and perturbations reveal model dependence, but the failed attribution randomisation check prevents treating saliency or attention as faithful explanation. The study did not establish robust emergent temporal dynamics. What emerged from the pooled benchmark was largely static cross-sectional structure rather than robust temporal dynamics.

Four evidence statements should remain distinct. First, pooled discrimination existed on the corrected historical test period; this is descriptive performance. Second, static prevalence was sufficient to outperform the Transformer in pooled AUC, and the Transformer's within-asset ranking was near chance; this supports the cross-sectional measurement conclusion. Third, identity and order interventions reveal model dependence but do not identify an economic cause. Fourth, simulation validates a sufficient shortcut mechanism but is not external market replication. The evidence does not support a deployable temporal risk model, a causal account of market stress, or a general claim that Transformers cannot learn financial dynamics. It supports a narrower methodological conclusion about how this pooled benchmark should be evaluated.

The study has important limits. The historical test set became adaptive as successive falsification and recovery analyses were performed, so the empirical evidence is historical test evidence rather than untouched confirmation. The current-symbol universe has survivorship risk and does not reconstruct all delisted instruments. The composite target is not rarity- or severity-equivalent across families; that heterogeneity is diagnosed, not removed. FRED inputs were lagged but obtained as current-vintage rather than fully point-in-time ALFRED series, and one credit-spread channel was mostly missing in training. Overlapping labels and common market shocks reduce effective support despite boundary-purge correction. Within-asset AUC removes between-asset pairs but does not by itself remove common-date dependence. The identity probe and occlusion tests establish recoverability and sensitivity, not causality. Finally, no externally confirmed positive temporal signal was found.

External validity is also limited: the panel uses one provider, one engineered feature set, one composite horizon and one historical era. Five models strengthen method generality but do not exhaust architectures; further test-driven searching would deepen adaptivity. Simulation validates diagnostic logic under a known mechanism, not market realism. These boundaries preserve the methodological result while precluding a universal statement about financial predictability.

The practical conclusion is correspondingly narrow. This work evaluates supervised forecasting models rather than deployable trading strategies. The fitted Transformer is not validated for time-varying operational risk monitoring, and no trading or profitability claim follows. The diagnostic framework is nevertheless useful for financial model validation and governance. Before a pooled score advances to deployment review, practitioners can compare it with training-only group priors, report within-entity and equal-entity metrics, remove or swap identity, perturb the temporal information claimed to matter, and validate those tests in simulation. This sequence distinguishes a model that ranks persistently risky instruments from one that reacts to changing conditions and directs review effort toward the actual source of performance.

## 6 Future work

The highest-value next step is independent confirmation, not further tuning on the opened test period. A future replication should be preregistered by freezing the target, features, models, static priors, within-asset metric, perturbations, minimum event support and stopping rule before any replication outcomes are inspected. A different provider and geography could test external transportability; prospectively collected dates could provide untouched confirmation. Training with leave-one-asset and leave-one-family-out generalisation designs would test whether learned dynamics transfer to previously unseen entities. A second line of work should design asset-relative outcomes with comparable rarity and severity, then test whether cross-asset information adds within-asset timing value. Interpretability should proceed only after a model passes the gate for ordered temporal skill and the attribution randomisation checks.

## References

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M. and Kim, B. (2018) 'Sanity checks for saliency maps', Advances in Neural Information Processing Systems, 31, pp. 9505-9515.

Cerqueira, V., Torgo, L. and Mozetic, I. (2019) 'Evaluating time series forecasting models: an empirical study on performance estimation methods', arXiv:1905.11744.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M. and Wichmann, F.A. (2020) 'Shortcut learning in deep neural networks', Nature Machine Intelligence, 2, pp. 665-673. doi: 10.1038/s42256-020-00257-z.

Hewitt, J. and Liang, P. (2019) 'Designing and interpreting probes with control tasks', Proceedings of EMNLP-IJCNLP 2019, pp. 2733-2743. doi: 10.18653/v1/D19-1275.

Jain, S. and Wallace, B.C. (2019) 'Attention is not explanation', Proceedings of NAACL-HLT 2019, pp. 3543-3556. doi: 10.18653/v1/N19-1357.

Janes, H. and Pepe, M.S. (2009) 'Adjusting for covariate effects on classification accuracy using the covariate-adjusted receiver operating characteristic curve', Biometrika, 96(2), pp. 371-382. doi: 10.1093/biomet/asp002.

Li, T., Liu, Z., Shen, Y., Wang, X., Chen, H. and Huang, S. (2024) 'MASTER: market-guided stock Transformer for stock price forecasting', Proceedings of the AAAI Conference on Artificial Intelligence, 38(1), pp. 162-170. doi: 10.1609/aaai.v38i1.27767.

Lim, B., Arik, S.O., Loeff, N. and Pfister, T. (2021) 'Temporal Fusion Transformers for interpretable multi-horizon time series forecasting', International Journal of Forecasting, 37(4), pp. 1748-1764. doi: 10.1016/j.ijforecast.2021.03.012.

López de Prado, M. (2018) Advances in Financial Machine Learning. Hoboken, NJ: Wiley.

Montero-Manso, P. and Hyndman, R.J. (2021) 'Principles and algorithms for forecasting groups of time series: locality and globality', International Journal of Forecasting, 37(4), pp. 1632-1653. doi: 10.1016/j.ijforecast.2021.03.004.

Salinas, D., Flunkert, V., Gasthaus, J. and Januschowski, T. (2020) 'DeepAR: probabilistic forecasting with autoregressive recurrent networks', International Journal of Forecasting, 36(3), pp. 1181-1191. doi: 10.1016/j.ijforecast.2019.07.001.

Shao, Z., Zhang, Z., Wang, F., Wei, W. and Xu, Y. (2022) 'Spatial-temporal identity: a simple yet effective baseline for multivariate time series forecasting', Proceedings of the 31st ACM International Conference on Information and Knowledge Management, pp. 4454-4458. doi: 10.1145/3511808.3557702.

Sirignano, J. and Cont, R. (2019) 'Universal features of price formation in financial markets: perspectives from deep learning', Quantitative Finance, 19(9), pp. 1449-1459. doi: 10.1080/14697688.2019.1622295.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. and Polosukhin, I. (2017) 'Attention is all you need', Advances in Neural Information Processing Systems, 30.

Zeng, A., Chen, M., Zhang, L. and Xu, Q. (2023) 'Are Transformers effective for time series forecasting?', Proceedings of the AAAI Conference on Artificial Intelligence, 37(9), pp. 11121-11128. doi: 10.1609/aaai.v37i9.26317.

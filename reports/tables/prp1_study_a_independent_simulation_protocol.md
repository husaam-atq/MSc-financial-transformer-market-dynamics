# PRP-1 Study A Independent Simulation Protocol

## Independence

The second implementation must not import, call or copy `phase9_shortcut_simulation.py`. It is derived from the registered mathematical mechanism and uses different code paths, seeds and robustness structure. A separate reviewer checks implementation and outputs.

## Core design

Thirty assets belong to five families. Chronological sequences contain static asset/family log-odds, an ordered old-to-recent contrast, AR persistence and independent innovations. The core factorial varies asset-prior heterogeneity over 0, 0.75, 1.5 and 2.5; ordered signal over 0, 0.5, 1.0 and 1.5; and persistence over 0 and 0.7. Each of 32 cells receives 20 common-seed replications, with 400 training and 200 validation periods separated by an eight-period purge.

Train-only global, family and asset priors are compared with a pooled identity-plus-sequence classifier. Metrics include pooled, pair-weighted within-asset and equal-asset AUC; pooled-minus-within gap; reversal and fixed-permutation drops; calibration and actual cell support.

## Robustness and primary gates

Predeclared robustness scenarios add one mechanism at a time: family-prior heterogeneity, common shocks, cross-sectional dependence, four-period events and 10% missingness. Each is applied to four frozen diagnostic anchors: low-prior/no-signal, high-prior/no-signal, low-prior/strong-signal and high-prior/strong-signal. With 20 common-seed replications this adds 400 runs to the 640-run core. These scenarios cannot rescue the core study.

Primary support requires: high-minus-low static-prior AUC increase at least 0.15; no-signal within-asset AUC in [0.47, 0.53]; strong-signal within-asset AUC at least 0.65; reversal drop at least 0.10; permutation drop at least 0.03. Report paired 95% intervals across common seeds and all cells. No breakpoint or phase transition is claimed.

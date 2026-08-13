# Phase 6 Market-Dynamics Dependence Audit

Four previously reported associations were re-estimated on the observed-session-repaired train/validation panel. Moving date blocks of 10, 20, 40 and 60 observations preserve serial dependence approximately; intervals remain sensitivity analyses rather than confirmatory population inference.

- `equities_to_bonds_lag1`: validation rho=-0.1740; every train and validation block interval excludes zero=False; minimum validation same-sign fraction=0.944.
- `equities_to_equities_lag5`: validation rho=0.2058; every train and validation block interval excludes zero=False; minimum validation same-sign fraction=0.938.
- `equity_bond_correlation`: validation rho=0.1819; every train and validation block interval excludes zero=False; minimum validation same-sign fraction=0.924.
- `momentum_dispersion`: validation rho=0.2473; every train and validation block interval excludes zero=True; minimum validation same-sign fraction=0.992.

No result is causal. Common shocks, target composition and post-hoc selection remain. A block interval that excludes zero does not establish economic importance.

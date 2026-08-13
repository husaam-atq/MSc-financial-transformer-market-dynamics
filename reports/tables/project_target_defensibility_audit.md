# Target Defensibility Audit

## Definition And Estimand

`target_stress_10d` is an operational binary adverse-event label formed from future return,
drawdown and volatility conditions over ten observed sessions. It is useful for studying
rare-event classification within the chosen universe. It is not a validated systemic-risk
measure, a severity-equivalent cross-family event, or a direct measure of economic loss.

## Adversarial Findings

- The ten-session horizon is a research design choice, not an economically unique horizon.
- The OR construction conflates negative returns, drawdowns and volatility spikes.
- The same thresholds imply very different event frequencies and severity across families.
- Validation prevalence ranged from 1.71% for bonds to 47.62% for crypto.
- Ten-session forward conditions create overlapping labels and prolonged positive runs.
- Asset identity therefore predicts the label even without time-varying market information.
- The static asset-prevalence baseline's superiority shows that target composition explains
  a material share of pooled discrimination.

## Verdict

The target is **defensible only as an operational, heterogeneous adverse-event label**. It
does not invalidate the stored predictions. It materially narrows the claim: this project
does not establish forecasting of a common cross-market stress construct.

## Required Wording

Use: "asset-level operational adverse-event classification over the next ten observed
sessions." State the three components and family prevalence every time pooled results are
introduced.

Avoid: "systemic stress", "common market stress", "severity-comparable stress", or
"universal cross-asset stress forecasting".

## Future Resolution

A family-relative quantile target, standardized severity score, or market-wide systemic
event definition could answer a different question. It requires preregistration and new
independent outcomes. Opening another target on the historical test would add adaptivity,
not repair the current evidence.

# IFDDRP Final Resume State

- Freeze date: 2026-08-08.
- Evidence-producing source commit: `e66470a6de35019cf8239eb4f8e6b8aab38e0bbb`.
- Final bounded Experiments A, B and C: completed; all promotion gates failed.
- Final Transformer interpretation: completed after explicit fixed-group correction; no model retraining.
- Central metrics: recomputed from preserved Phase 6 predictions and reconciled by score type.
- Evidence freeze and closest-work review: rebuild with `py scripts/build_ifddrp_final_evidence_freeze.py`.
- Empirical modelling status: scientifically frozen on the opened historical panel.
- Next highest-value action: write the dissertation from the three principal contributions and the claim register. Do not reopen modelling unless using a new, independently preregistered replication or prospective dataset.

Resume validation commands:

```text
py scripts/build_ifddrp_final_evidence_freeze.py
py -m pytest
py scripts/check_public_hygiene.py
```

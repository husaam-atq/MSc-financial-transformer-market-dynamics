# PRP-1 Fixed Cross-Model Execution Audit

## Empirical completion

The empirical study is cell-complete: 30 model/identity/seed cells, 163 evaluation rows, 120 temporal-order rows, 15 identity swaps and 48 identity probes. The strict temporal-skill recurrence gate remains failed for all five model families.

## Protocol and provenance qualifications

- Smoke coverage was 4 of 10 model/identity combinations. Flattened logistic and all no-ID arms were not represented in the smoke manifest, despite the protocol stop rule.
- The subsequent training and evaluation cells completed, but this does not retroactively satisfy the smoke gate.
- The ignored local temporal file has been synchronized to the complete 120-row tracked evidence.
- The registered stride-10 sensitivity is now reported from existing predictions; no model was retrained.
- The compact artefact manifest records SHA-256 hashes for 150 ignored checkpoint/prediction files.
- The execution-time runner commit was not persisted inside the ignored run contract. Git history suggests the final execution/report commit is compatible, but the exact execution commit is unavailable and is not inferred as fact.

## Scientific effect

These corrections do not alter the negative scientific verdict. They narrow the completion claim from strict protocol completion to empirical cell completion with a documented preflight/smoke noncompliance.

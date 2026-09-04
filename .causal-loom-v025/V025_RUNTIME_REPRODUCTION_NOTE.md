# Causal Loom v0.25 — canonical pinned-runtime reproduction note

No B0007 or B0018 measurement payload was opened before this note.

The frozen protocol and executable specify Python 3.11, NumPy 2.1.3, and SciPy 1.14.1. The pre-workflow expected development hashes were mistakenly calculated in the local analysis container using NumPy 2.3.5 and SciPy 1.17.0.

The chunk-safe GitHub development job reconstructed the exact frozen protocol and executable, verified their SHA-256 hashes, used the exact nested NASA archive, opened only B0005 and B0006, and passed every frozen development gate. Its access manifest explicitly lists B0007 and B0018 as unopened.

## Scientific equivalence audit

Relative to the local unpinned-environment lock:

- all fixed hyperparameters, cell identities, archive identity, protocol identity, and the development-pass verdict are identical;
- both base modes `h0` are byte-numerically identical;
- the largest absolute difference in either tangent mode `h1` is below `2.0e-15`;
- all discrete outcomes are identical: 60 valid cycles, zero catastrophes, 88.333% overall wins, 88.333% tail wins, 83.636% relaxation wins, and 100% wins over the generic hybrid cubic;
- interpolation-derived aggregate ratios differ by approximately 0.4%–1.6%, while every predeclared development gate still passes by a large margin.

The canonical pinned-runtime development identities are therefore:

- `development_results.json` SHA-256: `c514603c9a3865f5f8f54ccd540f3c2280648ab67efe1f0faff845862d945ab5`
- `CANDIDATE_LOCK.json` SHA-256: `a67b849f8e98af53023fbca1bd85404da9a6df9e2f0002ac9085959654f2ce16`
- source GitHub Actions run: `33903068968`
- source artifact: `causal-loom-v025-chunk-safe-development-evidence`

Only these runtime-specific reproduction hashes are superseded. The frozen scientific protocol, executable, thresholds, candidate architecture, five probe positions, fallback policy, development cells, and holdout cells remain unchanged.

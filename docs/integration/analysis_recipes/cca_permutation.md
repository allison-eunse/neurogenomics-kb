---
title: Analysis Recipe — CCA + Permutation
status: ready
updated: 2025-11-16
---

# CCA + Permutation

Inputs

- X_gene, X_brain: residualized and standardized matrices (N × d_gene, N × d_brain) projected to ~512 dims.
- Covariates: used upstream during residualization.

Protocol

1) Fold discipline
- Use K stratified folds (group/site-aware if needed).
- Within each train fold:
  - Fit CCA on X_gene_train, X_brain_train.
  - Transform train and test to canonical scores.
2) Permutation
- For b in 1..B (B = 1,000):
  - Permute subject alignment in one modality within the train fold.
  - Fit CCA on permuted pairs.
  - Record ρ1_null.
- p = (count(ρ1_null ≥ ρ1_obs) + 1) / (B + 1).
3) Reporting
- ρ1–ρ3 with permutation p-values.
- Optional: bootstrap CIs on ρ1.
- Loadings/feature contributions for interpretation.

Pitfalls
- Never fit CCA on all data.
- Keep the same permutation protocol across folds for comparability.

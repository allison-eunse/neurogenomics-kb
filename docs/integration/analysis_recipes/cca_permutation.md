---
title: Analysis Recipe — CCA + Permutation
status: ready
updated: 2025-11-18
---

# CCA + Permutation

Inputs

- X_gene, X_brain: residualized and standardized matrices (N × d_gene, N × d_brain) projected to ~512 dims.
- Covariates: used upstream during residualization.
- Metadata: record `embedding_strategies.<id>`, `harmonization_methods.<id>`, and (for fMRI) `rsfmri_preprocessing_pipelines.<id>` to ensure results are traceable.

Context in integration plan

- This recipe is part of the **diagnostic / exploration layer** of the integration stack.
- Run it **after per-modality sanity checks** but before heavier fusion models; it tells you whether there is cross-modal structure worth chasing.
- Treat it as a companion to the late-fusion-first baselines rather than a replacement for prediction experiments.

Protocol

### 1. Fold discipline
- Use **K stratified folds** (make them site-/scanner-aware when possible).
- For each train fold:
  1. Fit CCA on `(X_gene_train, X_brain_train)`.
  2. Transform the train **and** held-out fold to canonical scores so downstream metrics share the same projection space.

### 2. Permutation test
- Set `B = 1,000` (or at least 500 for quick scans).
- For each `b ∈ {1 … B}` inside the *training* split:
  1. Permute subject IDs in one modality (genes or brain) while keeping covariates fixed.
  2. Refit CCA on the permuted pair.
  3. Store the first canonical correlation `ρ1^(b)` to build a null.
- Compute the empirical p-value  
  `p = ( # {ρ1^(b) ≥ ρ1_obs} + 1 ) / (B + 1 )`.

### 3. Reporting & interpretation
- Report `ρ1–ρ3` with their permutation p-values (per fold and averaged).
- Optionally bootstrap the canonical correlations for 95 % CIs.
- Surface top loadings / feature contributions for both modalities to explain shared signal.

Why pair CCA with permutations?

- CCA will always produce non-zero canonical correlations—even when there is no shared structure—because it can overfit high-dimensional spaces.
- The permutation loop builds a modality-shuffled null distribution so we can report p-values (or FDR-adjusted thresholds) and avoid over-interpreting noise.
- This statistical check is lightweight enough for “quick tests” while still respecting site/ancestry confounds.

Pitfalls
- Never fit CCA on all data.
- Keep the same permutation protocol across folds for comparability.

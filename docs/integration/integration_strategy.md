---
title: Integration Strategy
status: draft
updated: 2025-11-16
---

# Integration Strategy

Overall philosophy

- Late integration first, then scale if we see gains.

Why this applies to genes × brain

- Heterogeneous semantics: nucleic-acid sequence vs morphology/dynamics → maximize modality specificity before fusion.
- Confounds differ: ancestry/batch vs site/motion/TR → deconfound independently.

Baselines

- Preprocess per modality
  - Z-score features.
  - Residualize against: age, sex, site/scanner, motion (FD), SES (if available), genetic PCs (PC1–PC10).
- Dimensionality
  - Project to 512 dims per modality (PCA or tiny MLP).
- CCA + permutation
  - CCA on train folds; 1,000 shuffles; report ρ1–ρ3 with p-values.
- Prediction
  - LR (balanced) and LightGBM/CatBoost on Gene, Brain, Fusion; same CV folds; AUROC/AUPRC; DeLong/bootstrap for Fusion vs single-modality.

Escalation criteria

- If Fusion > max(Gene, Brain) with p < 0.05 (DeLong/bootstrap), consider:
  - Two-tower contrastive alignment (frozen encoders; small projectors).
  - EI stacking over per-modality models.
  - Harmony-style hub tokens/TAPE if TR/site heterogeneity limits fMRI.

Interpretability

- LOGO ΔAUC with Wilcoxon + FDR for gene attribution.
- CCA loadings; partial correlations of axes with outcomes (covariate-adjusted).

Risks and mitigations

- Leakage: do scaling/residualization within train folds; apply transforms to test.
- Site imbalance: use group/site-aware CV when feasible.
- Overfitting at high dims: prefer 256–512; regularize LR; early stopping for GBDT.

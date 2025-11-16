---
title: Integration Baseline Plan (Nov 2025)
status: active
updated: 2025-11-16
---

# Integration Baseline Plan

Where each step came from (paper → inference → our plan)

- Principle: Prefer late integration first under heterogeneous semantics.
  - Sources: Ensemble Integration (Li et al. 2022), Oncology multimodal review (2024).
  - Inference: Preserve modality-specific signal; avoid premature joint spaces.
  - Plan: Concatenate compact per-modality features; train LR and GBDT baselines.

- Robustness and evaluation discipline.
  - Sources: Oncology review; BrainLM/Brain-JEPA/Harmony practices.
  - Plan: Z-score + residualize per feature vs covariates; same CV folds; AUROC/AUPRC with CIs; DeLong/bootstrap for differences.

- CCA + permutation and partial correlations before heavy fusion.
  - Sources: Review guidance; neuro CCA tradition.
  - Plan: CCA on residualized, standardized inputs; 1,000 permutations; partial correlations/logistic with covariates.

- Modality sequencing.
  - Sources: Harmony, SwiFT, BrainLM/JEPA.
  - Plan: Start with sMRI ROIs; add fMRI as FC vectors; later consider brain FMs.

- Genetics embedding hygiene and attribution.
  - Sources: Caduceus (RC-equivariance), DNABERT-2, GENERator; BIOKDD'25 LOGO.
  - Plan: RC-average; deterministic tokenization; LOGO ΔAUC with Wilcoxon + FDR.

Escalation

- If late fusion adds value: two-tower contrastive (frozen encoders, small projectors, InfoNCE), EI stacking, hub tokens/TAPE if TR/site heterogeneity dominates.

---
title: Ensemble Integration (EI) — Guidance Card
status: draft
updated: 2025-11-16
tags: [integration, ensembles]
---

# Ensemble Integration (EI)

Problem it solves
- Robust multimodal integration under heterogeneous semantics with interpretable ensembles.

Core mechanics
- Train strong per-modality learners with diverse inductive biases (LR, trees, SVM).
- Late fusion via stacking/ensemble selection; rank-based interpretation available.

When to use
- Heterogeneous feature spaces; small-to-medium N; missing data.

Adoption in our pipeline
- LR + LightGBM/CatBoost per modality and concatenated; optional stacking after baselines succeed.

Caveats
- Stacking must be fold-proper to avoid leakage; prefer simple meta-learners.

References
- Li et al., 2022 (Bioinformatics Advances).

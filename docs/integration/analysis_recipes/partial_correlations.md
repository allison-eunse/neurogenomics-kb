---
title: Analysis Recipe — Partial Correlations
status: ready
updated: 2025-11-16
---

# Partial Correlations

Goal
- Associate canonical scores or PCs with outcomes controlling covariates.

Continuous outcome (e.g., PHQ-9)
- Residualize x and y on covariates within train folds → rx, ry.
- Correlate rx, ry (Pearson/Spearman); aggregate across folds.

Binary outcome (e.g., MDD)
- Preferred: logistic regression y ~ x + covariates; report OR, CI, p.
- Optional: approximate partial correlation via residuals y − p̂ from covariate-only logistic.

Report
- Per-axis coefficients/correlations with CIs; FDR across multiple tests if many axes.

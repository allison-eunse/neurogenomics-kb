---
title: Analysis Recipe — Prediction Baselines
status: ready
updated: 2025-11-16
---

# Prediction Baselines

Inputs
- Gene, Brain, and Fusion = [Gene | Brain], all 512-D after preprocessing.

Models
- Logistic Regression
  - penalty=L2, C∈{0.5,1,2}, solver=saga/liblinear, class_weight=balanced, max_iter=5,000.
- LightGBM
  - num_leaves=31, learning_rate=0.05, n_estimators=1,000 with early stopping, scale_pos_weight ≈ N_neg/N_pos.
- CatBoost
  - depth=6–8, learning_rate=0.05, iterations=2,000 with early stopping, loss_function=Logloss, auto class weights.

Evaluation
- Same CV folds across modalities.
- Metrics: AUROC, AUPRC; report mean ± SD across folds.
- Significance: DeLong or bootstrap for Fusion vs each single-modality on held-out predictions.

Outputs to save
- Per-fold predictions and labels for later DeLong/bootstrap and calibration checks.

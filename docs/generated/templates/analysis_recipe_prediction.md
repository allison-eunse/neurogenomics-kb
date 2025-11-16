# Analysis Recipe: Prediction Baselines (LR + GBDT)

## Goal
Estimate classification performance for each modality (genes, sMRI, fMRI) and their late-fusion concatenation using strong, interpretable baselines with rigorous evaluation.

## Inputs
- Fold-specific train/test splits shared across modalities.
- Residualized, z-scored, 512-D embeddings per modality (plus covariates for logging).
- Binary or multi-class clinical targets (e.g., MDD, eoMDD vs loMDD).

## Models
### Logistic Regression
- Solver: `saga` or `lbfgs`.
- Penalty: L2; tune `C` in `{0.01, 0.1, 1, 10}`.
- `class_weight='balanced'`.
- Max iter ≥ 5,000; tolerance `1e-4`.

### LightGBM
- `num_leaves=31`, `max_depth=-1`, `learning_rate=0.05`.
- `n_estimators=1000` with early stopping (patience 50) on validation splits.
- `feature_fraction=0.9`, `bagging_fraction=0.8`, `bagging_freq=5`.
- `scale_pos_weight = N_neg / N_pos` if not using class weights.

### CatBoost (optional)
- `depth=6-8`, `learning_rate=0.05`, `iterations=2000`, `l2_leaf_reg=3`.
- `loss_function='Logloss'`, `eval_metric='AUC'`, `auto_class_weights='Balanced'`.

## Procedure
1. Fit models on train split per modality; tune hyperparameters via inner CV or validation fold.
2. Evaluate on held-out fold; store probabilities and logits.
3. Repeat for concatenated late-fusion features (stacked 512-D per modality).
4. Record metrics: AUROC, AUPRC, accuracy, calibration (Brier/ECE optional).
5. Compare modalities: run DeLong test (or bootstrap) on AUROC differences; report mean ± SD across folds.
6. For LOGO analyses, compute ΔAUC per fold and apply Wilcoxon + FDR.

## Logging
- Persist per-fold predictions in `artifacts/generated/predictions/<experiment_id>/`.
- Store trained hyperparameters, random seeds, and config YAML snapshot.
- Capture ROC/PR curves and metric tables; include class prevalence.

## Reporting Template
- Table with AUROC/AUPRC (mean ± SD) for Gene, Brain, Fusion.
- DeLong/Bootstrap ΔAUROC with 95% CI and p-value.
- Calibration plot or summary if clinically relevant.
- Notes on covariates, residualization, and projector settings.

## References
- Ensemble Integration paper (late fusion motivation).
- Oncology multimodal review (evaluation discipline).


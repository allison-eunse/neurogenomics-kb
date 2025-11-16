# Analysis Recipe: CCA + Permutation

## Goal
Quantify cross-modal associations (e.g., gene embeddings vs sMRI features) while controlling confounds and validating significance via permutation testing.

## Inputs
- Residualized & z-scored feature matrices `X` (modalities A) and `Y` (modalities B) per fold.
- Covariate design matrices (age, sex, site/scanner, motion FD, SES, genetic PCs).
- Train/test splits (identical across modalities).

## Preprocessing Checklist
1. Fit scaler + residualization models on train split only; apply to both train/test within the fold.
2. Optional PCA/MLP projector to 512-D per modality; store fit parameters.
3. Log confound regression coefficients for reproducibility.

## Procedure
1. **Fit CCA on train data:** `cca = CCA(n_components=k, scale=False)` with shrinkage/regularization if needed.
2. **Transform:** Obtain canonical variates for both train and test sets.
3. **Record metrics:** Canonical correlations (ρ₁…ρ_k), variance explained, loadings.
4. **Permutation test:** Shuffle subject order in modality B, refit CCA `B` times (≥1,000); build null distribution for each ρ.
5. **p-values:** `p = (count(ρ_null ≥ ρ_obs) + 1) / (B + 1)`.
6. **Confidence intervals (optional):** Bootstrap subjects within folds.
7. **Partial correlations to outcomes:** Regress canonical scores and clinical targets on covariates; correlate residuals or use covariate-adjusted regression.

## Logging
- Save canonical correlations, permutation distributions, p-values, and loadings to `artifacts/generated/cca/<experiment_id>/`.
- Store config (modalities, projectors, covariates, seeds) alongside results.

## Reporting Template
- Table of top 3 ρ with permutation p-values and 95% CI.
- Heatmap or bar chart of feature loadings (with sparse thresholding if needed).
- Partial correlation table linking canonical scores to clinical outcomes (effect size, p, FDR q).

## References
- EI & oncology multimodal review for integration motivation.
- Classical CCA references and permutation testing guidelines.


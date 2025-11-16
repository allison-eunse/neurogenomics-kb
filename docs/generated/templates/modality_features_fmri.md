# Modality Features: Functional MRI (rs-fMRI)

## Source Inputs
- Preprocessed rs-fMRI timeseries (e.g., fMRIPrep outputs, custom pipelines).
- Atlas definitions (Schaefer-400, Gordon, volumetric grid).
- Motion / QC metrics (FD, DVARS, censoring masks).

## Option A: Functional Connectivity (Fast Baseline)
1. Parcellate time series per subject.
2. Compute Pearson correlations per ROI pair; apply Fisher z-transform.
3. Flatten upper triangle; optionally reduce via PCA (100–256 dims) before projecting to 512-D.
4. Residualize covariates (age, sex, site/scanner, motion FD, TR, SES) within folds.
5. Store embeddings + QC flags in `artifacts/generated/embeddings/fmri_fc/`.

## Option B: Foundation Model Embeddings
- Supported encoders: BrainLM, Brain-JEPA, Brain Harmony, SwiFT, BrainMT.
- Preprocessing needs: TR normalization (Harmony TAPE), mask collators, gradient positional encodings.
- Pooling: CLS token, mean over spatial tokens, hub tokens; document choice.
- 512-D projector (PCA/MLP) fit per fold; log checkpoint versions.

## Motion & Site Handling
- Include FD (mean, max) and number of censored volumes as covariates.
- Consider site-aware splits or mixed-effects modeling if imbalance severe.
- For heterogeneous TRs: align via resampling or Harmony’s TAPE (PI-resize) before embedding.

## Covariate Residualization
- Age, sex, site/scanner, motion FD, TR group, SES, acquisition batch.
- Document design matrices and residualization scripts.

## Integration Notes
- Align subject IDs with genomics/sMRI intersections; record dropouts.
- Provide config references (`configs/projectors/fmri_pca512.yaml`, `kb/datasets/<cohort>.yaml`).
- Track version of preprocessing (e.g., fMRIPrep 23.2) and smoothing parameters.

## References
- BrainLM, Brain-JEPA, Brain Harmony, SwiFT, BrainMT papers.
- Internal notebooks for FC pipeline and FM embedding extraction.


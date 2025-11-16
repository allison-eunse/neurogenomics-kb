# Modality Features: Structural MRI (sMRI)

## Source Inputs
- FreeSurfer / CIVET outputs path.
- Atlas / parcellation used (e.g., Desikan, Schaefer-400 volumetric maps).
- QC reports (Euler number thresholds, manual edits).

## Feature Extraction
- ROI metrics: cortical thickness, volume, surface area, subcortical volumes.
- Derived composites (asymmetry indices, global measures) if needed.
- File formats (`.tsv`, `.csv`, HDF5) and loader scripts.

## Preprocessing & Harmonization
- Z-score within train folds; residualize covariates (age, sex, ICV, site/scanner, SES).
- Optionally apply ComBat or mixed-effects models; document rationale.
- Handle missing ROIs (impute, drop subject, or add mask).

## Embedding / Projection
- Stack ROI vectors per subject; optionally reduce via PCA to 512-D (fit per train fold).
- Alternative: use pretrained encoders (Brain Harmony hub tokens) when available; document pooling strategy.
- Persist embeddings + scalers in `artifacts/generated/embeddings/smri/`.

## Covariates
- Minimum set: age, sex, intracranial volume, site/scanner, scanner software version.
- Additional: handedness, SES, acquisition batch.

## Integration Notes
- Use same subject IDs as genomics/fMRI intersections; log any exclusions.
- Provide YAML hooks for experiments: `kb/datasets/<cohort>.yaml`, `configs/projectors/smri_pca512.yaml`.
- Mention reliability metrics (test-retest ICC if available).

## References
- Documentation for FreeSurfer pipeline version.
- Papers motivating ROI selection / harmonization steps.


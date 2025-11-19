# Data schemas

## genetics_embeddings.parquet
- `eid`
- `embedding_dim`
- `source_model`
- `layer`
- `vector`

## brain_idps.parquet
- `eid`
- `site`
- `modality` (sMRI or fMRI)
- Selected IDPs:
  - **sMRI:** FreeSurfer 7.x `aparc.stats` (cortical thickness, ~68 regions) + `aseg.stats` (subcortical volumes, ~40 structures) + surface area → ~176 features
  - **fMRI:** Parcel-wise BOLD statistics (mean, variance), connectivity matrices (optional), or direct FM embeddings (BrainLM, Brain-JEPA)
- Confounds:
  - `intracranial_volume` (sMRI)
  - `mean_fd` (mean framewise displacement, fMRI)
  - `tsnr` (temporal SNR, fMRI)
  - `euler_number` (FreeSurfer QC metric, sMRI)

## participants.parquet
- `eid`
- `age`
- `sex`
- `income_bin`
- `pcs_1`..`pcs_10`
- `site`
- `mdd_label`

## splits.json
- `fold_id`
- `train` / `val` / `test` EID lists
- `seed`
- `created_at`

Validation: reference `scripts/validate_schemas.py`.







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
- `modality`
- Selected IDPs (columns TBD)
- Confounds (motion, tSNR, etc.)

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




Dataset card schema (one YAML per dataset under `kb/datasets/`):

```
id: unique_slug
name: Human-readable title
description: Short paragraph
storage_location:
  bucket|huggingface|drive|source: URI or path
schema_ref: docs/data/schemas.md#anchor
dtypes: description of expected file types
required_columns:
  - column
counts:
  subjects|records|base_pairs: numeric summary
modalities:
  - dna|fmri|clinical
access: public|restricted|mixed
restrictions: Notes on licensing/compliance
maintainers:
  - name: Contact or team
verified: true|false
last_updated: YYYY-MM-DD
```

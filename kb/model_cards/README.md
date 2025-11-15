Every model must have a YAML card under `kb/model_cards/{id}.yaml`:

```
id: slug
name: Human readable name
modality: genetics|brain|multimodal
domain: dna|fmri|multimodal
summary: |
  3–5 lines
arch:
  type: architecture label
  backbone: key file(s)
  parameters: ~size
  context_length: tokens/timepoints
  special_features:
    - bullet list
repo: https://...
weights:
  huggingface:
    - url
  artifacts:
    - path
tokenizer: {...}
context_length: int
checkpoints:
  - name: ckpt
    path: url
license:
  code: SPDX id
  weights: SPDX id
  data: short note
datasets:
  - dataset_id
tasks:
  - masked_language_modeling
how_to_infer:
  huggingface: |
    code snippet
inference_api:
  provider: huggingface|local
  endpoint: url/instructions
integrations:
  - integration_id
 tags:
  - keyword
verified: false
last_updated: YYYY-MM-DD
notes: Optional paragraph
```

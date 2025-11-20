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

## Model cards coverage (docs ↔ walkthroughs ↔ YAML)

| ID | Domain | Model doc | Walkthrough | Card |
| --- | --- | --- | --- | --- |
| `caduceus` | Genetics / DNA | `docs/models/genetics/caduceus.md` | `docs/code_walkthroughs/caduceus_walkthrough.md` | `kb/model_cards/caduceus.yaml` |
| `dnabert2` | Genetics / DNA | `docs/models/genetics/dnabert2.md` | `docs/code_walkthroughs/dnabert2_walkthrough.md` | `kb/model_cards/dnabert2.yaml` |
| `evo2` | Genetics / DNA | `docs/models/genetics/evo2.md` | `docs/code_walkthroughs/evo2_walkthrough.md` | `kb/model_cards/evo2.yaml` |
| `generator` | Genetics / DNA | `docs/models/genetics/generator.md` | `docs/code_walkthroughs/generator_walkthrough.md` | `kb/model_cards/generator.yaml` |
| `brainlm` | Brain / fMRI | `docs/models/brain/brainlm.md` | `docs/code_walkthroughs/brainlm_walkthrough.md` | `kb/model_cards/brainlm.yaml` |
| `brainjepa` | Brain / fMRI | `docs/models/brain/brainjepa.md` | `docs/code_walkthroughs/brainjepa_walkthrough.md` | `kb/model_cards/brainjepa.yaml` |
| `brainharmony` | Brain / sMRI+fMRI | `docs/models/brain/brainharmony.md` | `docs/code_walkthroughs/brainharmony_walkthrough.md` | `kb/model_cards/brainharmony.yaml` |
| `brainmt` | Brain / fMRI | `docs/models/brain/brainmt.md` | `docs/code_walkthroughs/brainmt_walkthrough.md` | `kb/model_cards/brainmt.yaml` |
| `swift` | Brain / fMRI | `docs/models/brain/swift.md` | `docs/code_walkthroughs/swift_walkthrough.md` | `kb/model_cards/swift.yaml` |
| `m3fm` | Multimodal / vision-language | `docs/models/multimodal/m3fm.md` | `docs/code_walkthroughs/m3fm_walkthrough.md` | `kb/model_cards/m3fm.yaml` |
| `me_llama` | Multimodal / medical LLM | `docs/models/multimodal/me_llama.md` | `docs/code_walkthroughs/melamma_walkthrough.md` | `kb/model_cards/me_llama.yaml` |
| `titan` | Multimodal / pathology | `docs/models/multimodal/titan.md` | `docs/code_walkthroughs/titan_walkthrough.md` | `kb/model_cards/titan.yaml` |
| `llm_semantic_bridge` | Integration / LLM hub | — | — | `kb/model_cards/llm_semantic_bridge.yaml` |
| `tabpfn` | Tabular baseline | — | — | `kb/model_cards/tabpfn.yaml` |
| `vlm_dev_clinical` | Clinical VLM concept | — | — | `kb/model_cards/vlm_dev_clinical.yaml` |

Rows with `—` indicate cards that exist in YAML form but do not yet have a rendered doc or walkthrough.

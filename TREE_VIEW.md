# Neurogenomics-KB Tree View
**Last Updated**: 2025-11-17

## Quick Navigation

```
neurogenomics-kb/
│
├── 📖 README.md                     ← Repository overview
├── 📝 KB_COMPLETION_SUMMARY.md      ← Nov 21 completion report (NEW ✨)
├── 📋 QUICK_REFERENCE.md            ← Commands and stats
├── 📋 ORGANIZATION_SUMMARY.md       ← Historical cleanup log
│
├── docs/                            ← Documentation root
│   ├── index.md                     ← MAIN KB INDEX (start here)
│   │
│   ├── code_walkthroughs/           ← 9 FM implementation guides
│   │   ├── index.md                 ← Walkthrough hub
│   │   ├── caduceus_walkthrough.md
│   │   ├── evo2_walkthrough.md
│   │   ├── generator_walkthrough.md
│   │   ├── brainlm_walkthrough.md
│   │   ├── brainjepa_walkthrough.md
│   │   ├── brainharmony_walkthrough.md
│   │   ├── brainmt_walkthrough.md
│   │   ├── swift_walkthrough.md
│   │   └── dnabert2_walkthrough.md
│   │
│   ├── decisions/                   ← Design rationale
│   │   └── 2025-11-integration-plan.md  ← Paper→plan mapping
│   │
│   ├── integration/                 ← How to integrate modalities
│   │   ├── integration_strategy.md  ← Late fusion first
│   │   ├── analysis_recipes/
│   │   │   ├── cca_permutation.md
│   │   │   ├── prediction_baselines.md
│   │   │   └── partial_correlations.md
│   │   └── modality_features/
│   │       ├── genomics.md          ← RC-averaging, LOGO
│   │       ├── smri.md
│   │       └── fmri.md
│   │
│   ├── models/                      ← Light model cards (markdown)
│   │   ├── brain/
│   │   │   ├── brainlm.md
│   │   │   ├── brainjepa.md
│   │   │   ├── brainharmony.md
│   │   │   ├── brainmt.md
│   │   │   └── swift.md
│   │   └── genetics/
│   │       ├── caduceus.md
│   │       ├── evo2.md
│   │       ├── generator.md
│   │       └── dnabert2.md
│   │
│   ├── kb/                          ← KB card templates
│   │   ├── README.md
│   │   └── templates/
│   │       ├── model_card_template.md
│   │       ├── integration_principles_card.md
│   │       ├── method_family_card.md
│   │       ├── external_model_pattern_card.md
│   │       ├── cross_domain_eval_card.md
│   │       ├── dataset_card.md
│   │       └── experiment_config_stub.md
│   │
│   └── generated/                   ← Exported from pdf<->md repo
│       └── kb_curated/
│           ├── README.md
│           ├── integration_cards/   ← EI, oncology review
│           └── datasets/
│
├── kb/                              ← Structured YAML metadata
│   ├── model_cards/                 ← 7 FM metadata cards
│   │   ├── README.md
│   │   ├── caduceus.yaml
│   │   ├── evo2.yaml
│   │   ├── generator.yaml
│   │   ├── brainlm.yaml
│   │   ├── brainjepa.yaml
│   │   ├── brainmt.yaml
│   │   ├── swift.yaml
│   │   └── template.yaml
│   │
│   ├── paper_cards/                 ← 11 paper summaries (NEW ✨)
│   │   ├── README.md                ← Index + usage
│   │   ├── ensemble_integration_li2022.yaml
│   │   ├── oncology_multimodal_waqas2024.yaml
│   │   ├── caduceus_2024.yaml
│   │   ├── evo2_2024.yaml
│   │   ├── generator_2024.yaml
│   │   ├── brainlm_2024.yaml
│   │   ├── brainjepa_2024.yaml
│   │   ├── brainharmony_2024.yaml
│   │   ├── brainmt_2024.yaml
│   │   ├── yoon_biokdd2025.yaml
│   │   ├── prs_guide.yaml
│   │   ├── gwas_diverse_populations.yaml
│   │   └── template.yaml
│   │
│   ├── datasets/                    ← 11 dataset specs + manifest
│   │   ├── README.md
│   │   ├── ukb_manifest_stub.yaml   ← Fill after data inventory (NEW ✨)
│   │   ├── ukb_fmri_tensor.yaml
│   │   ├── hg38_reference.yaml
│   │   └── [8 more dataset cards...]
│   │
│   └── integration_cards/           ← 2 multimodal strategies
│       ├── genetics_embeddings_pipeline.yaml
│       └── ukb_genetics_brain_alignment.yaml
│
├── configs/                         ← Experiment templates (NEW ✨)
│   └── experiments/
│       ├── README.md                ← Usage guide
│       ├── 01_cca_gene_smri.yaml
│       ├── 02_prediction_baselines.yaml
│       └── 03_logo_gene_attribution.yaml
│
├── scripts/                         ← KB management tools
│   ├── manage_kb.py                 ← Validate cards
│   ├── codex_gate.py                ← Quality gate
│   ├── fetch_external_repos.sh      ← Clone FM repos
│   └── README.md
│
├── external_repos/                  ← Reference implementations (git-ignored)
│   ├── caduceus/
│   ├── evo2/
│   ├── generator/
│   ├── brainlm/
│   ├── brainjepa/
│   ├── brainharmony/
│   ├── brainmt/
│   └── swift/
│
├── rag/                             ← For RAG (defer to Dec)
│   └── vectordb/
│
└── site/                            ← MkDocs build output
```

## Key Entry Points

### For Understanding
- **Start**: `docs/index.md`
- **Integration plan**: `docs/decisions/2025-11-integration-plan.md`
- **Paper summaries**: `kb/paper_cards/README.md`
- **Walkthrough hub**: `docs/code_walkthroughs/index.md`

### For Implementation
- **Analysis recipes**: `docs/integration/analysis_recipes/`
- **Modality features**: `docs/integration/modality_features/`
- **Experiment configs**: `configs/experiments/`

### For Reference
- **Model metadata**: `kb/model_cards/`
- **Paper metadata**: `kb/paper_cards/`
- **Dataset specs**: `kb/datasets/`
- **External code**: `external_repos/[model_name]/`

---

**Total Files**:
- 9 code walkthroughs
- 7 model cards (YAML)
- 11 paper cards (YAML) ← NEW
- 11 dataset cards (YAML)
- 3 experiment configs (YAML) ← NEW
- 99 YAML files total across kb/ and configs/

**Status**: ✅ Nov 21 KB Complete


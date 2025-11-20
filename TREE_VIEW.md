# Neuro-Omics-KB Tree View
**Last Updated**: 2025-11-20

## Quick Navigation

```
neuro-omics-kb/
│
├── 📖 README.md                          ← Repository overview
├── 📋 QUICK_REFERENCE.md                 ← Commands and stats
├── 📄 ARPA-H 인지_연구계획서 (250619).pdf  ← ARPA-H proposal (local reference)
│
├── docs/                                 ← Documentation root (MkDocs)
│   ├── index.md                          ← MAIN KB INDEX (start here)
│   │
│   ├── guide/
│   │   └── kb_overview.md                ← KB architecture & navigation
│   │
│   ├── code_walkthroughs/                ← 15 FM & multimodal walkthroughs
│   │   ├── index.md                      ← Walkthrough hub
│   │   ├── caduceus_walkthrough.md
│   │   ├── dnabert2_walkthrough.md
│   │   ├── evo2_walkthrough.md
│   │   ├── generator_walkthrough.md
│   │   ├── brainlm_walkthrough.md
│   │   ├── brainjepa_walkthrough.md
│   │   ├── brainharmony_walkthrough.md
│   │   ├── brainmt_walkthrough.md
│   │   ├── swift_walkthrough.md
│   │   ├── bagel_walkthrough.md
│   │   ├── mot_walkthrough.md
│   │   ├── titan_walkthrough.md
│   │   ├── m3fm_walkthrough.md
│   │   ├── melamma_walkthrough.md
│   │   └── fms_medical_walkthrough.md
│   │
│   ├── decisions/                        ← Design rationale
│   │   ├── 2025-11-integration-plan.md   ← Integration baseline plan
│   │   └── dev_validation_plan.md        ← CHA developmental validation
│   │
│   ├── integration/                      ← How to integrate modalities
│   │   ├── index.md
│   │   ├── integration_strategy.md       ← Late fusion first
│   │   ├── embedding_policies.md
│   │   ├── multimodal_architectures.md
│   │   ├── design_patterns.md
│   │   ├── benchmarks.md
│   │   ├── analysis_recipes/
│   │   │   ├── cca_permutation.md
│   │   │   ├── prediction_baselines.md
│   │   │   └── partial_correlations.md
│   │   └── modality_features/
│   │       ├── genomics.md               ← RC-averaging, LOGO
│   │       ├── smri.md
│   │       └── fmri.md
│   │
│   ├── models/                           ← Light model guides (markdown)
│   │   ├── brain/
│   │   │   ├── index.md
│   │   │   ├── brainlm.md
│   │   │   ├── brainjepa.md
│   │   │   ├── brainharmony.md
│   │   │   ├── brainmt.md
│   │   │   └── swift.md
│   │   ├── genetics/
│   │   │   ├── index.md
│   │   │   ├── caduceus.md
│   │   │   ├── dnabert2.md
│   │   │   ├── evo2.md
│   │   │   └── generator.md
│   │   ├── integrations/
│   │   │   └── [integration example docs...]
│   │   └── multimodal/
│   │       └── [multimodal FM docs...]
│   │
│   ├── kb/                               ← KB card templates
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
│   ├── data/
│   │   ├── governance_qc.md
│   │   ├── ukb_data_map.md
│   │   ├── schemas.md
│   │   └── subject_keys.md
│   │
│   ├── generated/
│   │   ├── kb_curated/                   ← PDFs + MD from pdf<->md;ai-summaries
│   │   └── templates/                    ← Markdown templates for new cards
│   │
│   └── assets/
│       └── css/
│           └── code-citations.css
│
├── kb/                                   ← Structured YAML metadata
│   ├── model_cards/                      ← 15 FM and hub metadata cards
│   │   ├── README.md
│   │   ├── caduceus.yaml
│   │   ├── dnabert2.yaml
│   │   ├── evo2.yaml
│   │   ├── generator.yaml
│   │   ├── brainharmony.yaml
│   │   ├── brainjepa.yaml
│   │   ├── brainlm.yaml
│   │   ├── brainmt.yaml
│   │   ├── swift.yaml
│   │   ├── m3fm.yaml
│   │   ├── me_llama.yaml
│   │   ├── tabpfn.yaml
│   │   ├── llm_semantic_bridge.yaml
│   │   ├── titan.yaml
│   │   ├── vlm_dev_clinical.yaml
│   │   └── template.yaml
│   │
│   ├── paper_cards/                      ← 20 paper summaries
│   │   ├── README.md                     ← Index + usage
│   │   └── [21 paper card YAML files...]
│   │
│   ├── datasets/                         ← 17 dataset specs + manifest
│   │   ├── README.md
│   │   ├── cha_dev_longitudinal.yaml
│   │   ├── fms_medical_catalog.yaml
│   │   ├── fms_medical_cursor.yaml
│   │   ├── gener_tasks.yaml
│   │   ├── genomic_benchmarks.yaml
│   │   ├── gue_benchmark.yaml
│   │   ├── hcp_fmri_tensor.yaml
│   │   ├── hg38_reference.yaml
│   │   ├── multi_species_corpus.yaml
│   │   ├── nucleotide_transformer_tasks.yaml
│   │   ├── opengenome2.yaml
│   │   ├── refseq_generator.yaml
│   │   ├── ukb_fmri_tensor.yaml
│   │   ├── ukb_genetics_pgs.yaml
│   │   ├── ukb_manifest_stub.yaml       ← Fill after data inventory
│   │   ├── ukb_smri_freesurfer.yaml
│   │   ├── ukb_wes.yaml
│   │   └── template.yaml
│   │
│   ├── integration_cards/                ← Embedding, alignment, harmonization
│   │   ├── README.md
│   │   ├── alignment_strategies.yaml
│   │   ├── embedding_strategies.yaml
│   │   ├── genetics_embeddings_pipeline.yaml
│   │   ├── harmonization_methods.yaml
│   │   ├── rsfmri_preprocessing_pipelines.yaml
│   │   ├── ukb_genetics_brain_alignment.yaml
│   │   └── template.yaml
│   │
│   ├── papers_fulltext/
│   ├── results/
│   │   └── README.md
│   ├── rag/
│   └── scripts/
│
├── configs/                              ← Experiment templates
│   └── experiments/
│       ├── README.md                     ← Usage guide
│       ├── 01_cca_gene_smri.yaml
│       ├── 01_cca_ukb_joo_smri_template.yaml
│       ├── 02_harmonization_ablation_smri.yaml
│       ├── 02_prediction_baselines.yaml
│       ├── 03_logo_gene_attribution.yaml
│       ├── 03_prediction_baselines_tabular.yaml
│       ├── cha_dev_smri_pca_dimsearch_template.yaml
│       ├── dev_01_brain_only_baseline.yaml
│       ├── dev_02_gene_brain_behaviour.yaml
│       └── ukb_smri_pca_dimsearch_template.yaml
│
├── scripts/                              ← KB management tools
│   ├── README.md
│   ├── manage_kb.py                      ← Validate cards, CI, ops
│   ├── codex_gate.py                     ← Quality gate
│   └── fetch_external_repos.sh           ← Clone FM repos
│
├── external_repos/                       ← Reference implementations (mix of tracked snapshots + fetch-on-demand placeholders)
│   ├── README.md
│   ├── bagel/
│   ├── brainharmony/
│   ├── brainjepa/
│   ├── brainlm/
│   ├── brainmt/
│   ├── caduceus/
│   ├── dnabert2/
│   ├── evo2/
│   ├── fms-medical/
│   ├── generator/
│   ├── M3FM/
│   ├── me-lamma/
│   ├── MoT/
│   ├── swift/
│   └── titan/
│
├── rag/
│   └── vectordb/
│
├── mkdocs_plugins/                       ← Local MkDocs extensions
│   ├── __init__.py
│   └── code_citations.py
│
├── neuro_omics_kb_plugins.egg-info/
├── mkdocs.yml                            ← MkDocs site config
├── pyproject.toml
├── requirements.txt
├── setup.py
├── verify_kb.sh
└── TREE_VIEW.md                          ← This file
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

**Total Files (approx.)**:
- 15 code walkthroughs (`docs/code_walkthroughs/*.md` excluding index)
- 15 model cards (YAML)
- 20 paper cards (YAML)
- 17 dataset cards (YAML)
- 10 experiment configs (YAML)

**Status**: ✅ KB updated with multimodal, dev, and ARPA-H assets (as of Nov 20, 2025)


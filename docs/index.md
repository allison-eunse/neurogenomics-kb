---
title: Neurogenomics KB
status: draft
updated: 2025-11-17
---

# Neurogenomics Knowledge Base

This KB captures the design rationale, integration strategy, and reproducible analysis recipes that connect genomics embeddings with neuroimaging features.

Contents

- Decisions
  - [Integration baseline plan](decisions/2025-11-integration-plan.md)
- Integration
  - [Integration strategy](integration/integration_strategy.md)
  - [Analysis recipes](integration/analysis_recipes/) (CCA + permutation, partial correlations, prediction baselines)
  - [Modality features](integration/modality_features/) (Genomics, sMRI, fMRI)
- Models
  - Brain FMs: [BrainLM](models/brain/brainlm.md), [Brain-JEPA](models/brain/brainjepa.md), [Brain Harmony](models/brain/brainharmony.md), [SwiFT](models/brain/swift.md), [BrainMT](models/brain/brainmt.md)
  - Genetics FMs: [Caduceus](models/genetics/caduceus.md), [DNABERT-2](models/genetics/dnabert2.md), [Evo2](models/genetics/evo2.md), [GENERator](models/genetics/generator.md)
- Data
  - [Governance/QC](data/governance_qc.md)
  - [UKB map and schemas](data/ukb_data_map.md)
  - [UKB manifest stub](../kb/datasets/ukb_manifest_stub.yaml)
- KB (cards)
  - [Model cards](../kb/model_cards/)
  - [Paper cards](../kb/paper_cards/) — NEW ✨
  - [Integration principles](generated/kb_curated/integration_cards/)
  - [Dataset cards](../kb/datasets/)
- Experiments
  - [Example configs](../configs/experiments/) — NEW ✨

Quick start

- Baselines
  - Z-score → residualize → per-modality 512-D projection → CCA + permutations → LR/GBDT (Gene, Brain, Fusion) → DeLong/bootstrap
- Confounds
  - Age, sex, site/scanner, motion (FD), SES, genetic PCs
- Roadmap
  - Late fusion first → two-tower contrastive → EI stacking → hub tokens/TAPE if needed

## KB workflow

### Paper cards (NEW ✨)
All papers from `pdf<->md;ai-summaries/input/` now have structured YAML cards in `kb/paper_cards/`:

**Integration principles:**
- [Ensemble Integration (Li et al. 2022)](../kb/paper_cards/ensemble_integration_li2022.yaml) — Late fusion rationale
- [Oncology Multimodal Review (Waqas et al. 2024)](../kb/paper_cards/oncology_multimodal_waqas2024.yaml) — Confounds & evaluation

**Foundation models:**
- [Caduceus](../kb/paper_cards/caduceus_2024.yaml) — RC-equivariant DNA FM
- [Evo2](../kb/paper_cards/evo2_2024.yaml) — 1M context StripedHyena
- [GENERator](../kb/paper_cards/generator_2024.yaml) — 6-mer generative DNA LM
- [BrainLM](../kb/paper_cards/brainlm_2024.yaml) — ViT-MAE for fMRI
- [Brain-JEPA](../kb/paper_cards/brainjepa_2024.yaml) — JEPA for fMRI
- [Brain Harmony](../kb/paper_cards/brainharmony_2024.yaml) — sMRI+fMRI with TAPE
- [BrainMT](../kb/paper_cards/brainmt_2024.yaml) — Hybrid Mamba-Transformer

**Methods & prior work:**
- [Yoon et al. BIOKDD'25](../kb/paper_cards/yoon_biokdd2025.yaml) — MDD gene embeddings + LOGO
- [PRS Guide (Choi & O'Reilly 2019)](../kb/paper_cards/prs_guide.yaml) — Polygenic risk scores
- [GWAS Diverse Populations (Peterson et al. 2019)](../kb/paper_cards/gwas_diverse_populations.yaml) — Ancestry control

### Experiment configs (NEW ✨)
Ready-to-run YAML templates in `configs/experiments/`:
- [01_cca_gene_smri.yaml](../configs/experiments/01_cca_gene_smri.yaml) — CCA + permutation baseline
- [02_prediction_baselines.yaml](../configs/experiments/02_prediction_baselines.yaml) — Gene vs sMRI vs Fusion
- [03_logo_gene_attribution.yaml](../configs/experiments/03_logo_gene_attribution.yaml) — LOGO ΔAUC protocol

### Integration cards
Drop new evidence into `docs/generated/kb_curated/integration_cards/` (run `pdf<->md;ai-summaries/build_docs.sh` after curating).

### Usage
1. Read paper card YAML for context
2. Check linked code walkthrough for implementation
3. Clone experiment config template
4. Fill dataset paths and parameters
5. Run and log outputs back to KB

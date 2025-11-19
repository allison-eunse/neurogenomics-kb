---
title: Neuro-Omics KB
status: active
updated: 2025-11-19
---

# 🧬🧠 Neuro-Omics Knowledge Base

!!! success "Welcome!"
    This knowledge base connects **genomics, brain imaging, and behavioral data** through foundation models and multimodal integration strategies. Whether you're analyzing UK Biobank data, implementing gene-brain fusion pipelines, or exploring developmental cohorts, you'll find structured documentation, ready-to-use recipes, and reproducible workflows here.

> **Maintained by** Allison Eun Se You | **Last updated** November 19, 2025

---

## 🚀 Getting Started

=== "🆕 New to this KB"

    1. Start with the [KB overview](guide/kb_overview.md) to understand the structure
    2. Explore [Genetics Models](models/genetics/index.md) or [Brain Models](models/brain/index.md)
    3. Check out a [code walkthrough](code_walkthroughs/index.md) for hands-on examples

=== "🔬 Planning an analysis"

    1. Review [Integration Strategy](integration/integration_strategy.md)
    2. Pick an [analysis recipe](integration/analysis_recipes/cca_permutation.md)
    3. Clone an [experiment config](https://github.com/allison-eunse/neuro-omics-kb/tree/main/configs/experiments)
    4. Validate with `python scripts/manage_kb.py`

=== "📚 Looking for papers"

    1. Browse [paper cards](https://github.com/allison-eunse/neuro-omics-kb/tree/main/kb/paper_cards)
    2. Read [integration principles](generated/kb_curated/integration_cards/)
    3. Check [decision logs](decisions/2025-11-integration-plan.md)

---

## 🎯 Foundation Model Registry

### Genetics Models
| Model | Best for | Context | Quick link |
|-------|----------|---------|-----------|
| 🧬 [Caduceus](models/genetics/caduceus.md) | RC-equivariant gene embeddings | DNA sequences | [Walkthrough](code_walkthroughs/caduceus_walkthrough.md) |
| 🧬 [DNABERT-2](models/genetics/dnabert2.md) | Cross-species transfer | BPE tokenization | [Walkthrough](code_walkthroughs/dnabert2_walkthrough.md) |
| 🧬 [Evo 2](models/genetics/evo2.md) | Ultra-long regulatory regions | 1M context | [Walkthrough](code_walkthroughs/evo2_walkthrough.md) |
| 🧬 [GENERator](models/genetics/generator.md) | Generative modeling | 6-mer LM | [Walkthrough](code_walkthroughs/generator_walkthrough.md) |

### Brain Models
| Model | Modality | Best for | Quick link |
|-------|----------|----------|-----------|
| 🧠 [BrainLM](models/brain/brainlm.md) | fMRI | Site-robust embeddings | [Walkthrough](code_walkthroughs/brainlm_walkthrough.md) |
| 🧠 [Brain-JEPA](models/brain/brainjepa.md) | fMRI | Lower-latency option | [Walkthrough](code_walkthroughs/brainjepa_walkthrough.md) |
| 🧠 [Brain Harmony](models/brain/brainharmony.md) | sMRI + fMRI | Multi-modal fusion | [Walkthrough](code_walkthroughs/brainharmony_walkthrough.md) |
| 🧠 [BrainMT](models/brain/brainmt.md) | sMRI/fMRI | Mamba efficiency | [Walkthrough](code_walkthroughs/brainmt_walkthrough.md) |
| 🧠 [SwiFT](models/brain/swift.md) | fMRI | Hierarchical spatiotemporal | [Walkthrough](code_walkthroughs/swift_walkthrough.md) |

---

## 📋 Decisions & Roadmaps

- [Integration baseline plan (Nov 2025)](decisions/2025-11-integration-plan.md) — Late fusion first, then escalate if fusion wins.

---

## 🔗 Integration Stack

- **Integration strategy:** [integration/integration_strategy.md](integration/integration_strategy.md)
- **Analysis recipes:**
  - [CCA + permutation](integration/analysis_recipes/cca_permutation.md)
  - [Prediction baselines](integration/analysis_recipes/prediction_baselines.md)
  - [Partial correlations](integration/analysis_recipes/partial_correlations.md)
- **Modality features:**
  - [Genomics](integration/modality_features/genomics.md)
  - [sMRI](integration/modality_features/smri.md)
  - [fMRI](integration/modality_features/fmri.md)
- **Design patterns:** [integration/design_patterns.md](integration/design_patterns.md)
- **Multimodal architectures:** [integration/multimodal_architectures.md](integration/multimodal_architectures.md)

---

## 🏥 Multimodal & Clinical Models

Beyond genetics and brain FMs, the KB documents **multimodal architectures** that inform Brain-Omics Model (BOM) design:

| Model | Type | Key feature | Documentation |
|-------|------|-------------|---------------|
| 🔗 [BAGEL](code_walkthroughs/bagel_walkthrough.md) | Unified multimodal | MoT experts (understanding/generation) | [Card](generated/kb_curated/papers-md/bagel_2025.md) |
| 🔗 [MoT](code_walkthroughs/mot_walkthrough.md) | Sparse transformer | Modality-aware FFNs (~55% FLOPs) | [Card](generated/kb_curated/papers-md/mot_2025.md) |
| 🏥 [M3FM](code_walkthroughs/m3fm_walkthrough.md) | Medical imaging + text | Bilingual CXR reports | Model card: `kb/model_cards/m3fm.yaml` |
| 🏥 [Me-LLaMA](code_walkthroughs/melamma_walkthrough.md) | Medical LLM | Continual pretrained LLaMA | Model card: `kb/model_cards/me_llama.yaml` |
| 🏥 [TITAN](code_walkthroughs/titan_walkthrough.md) | Whole-slide imaging | Multi-scale histopathology | Model card: `kb/model_cards/titan.yaml` |

📖 [See full multimodal architectures guide →](integration/multimodal_architectures.md)

---

## 📚 Research Papers

**14 structured paper cards** in `kb/paper_cards/`:

### 🔗 Integration & Methods (5 papers)
- [Ensemble Integration (Li 2022)](generated/kb_curated/papers-pdf/ensemble_integration_li2022.pdf) — Late fusion rationale
- [Oncology Multimodal (Waqas 2024)](generated/kb_curated/papers-pdf/oncology_multimodal_waqas2024.pdf) — Confounds & evaluation
- [Yoon et al. BioKDD 2025](generated/kb_curated/papers-pdf/yoon_biokdd2025.pdf) — MDD gene embeddings + LOGO
- [PRS Guide](generated/kb_curated/papers-pdf/prs_guide.pdf) — Polygenic risk scores
- [GWAS Diverse Populations](generated/kb_curated/papers-pdf/gwas_diverse_populations.pdf) — Ancestry control

### 🧬 Genetics FMs (3 papers)
- [Caduceus (2024)](generated/kb_curated/papers-pdf/caduceus_2024.pdf) — RC-equivariant DNA FM
- [Evo2 (2024)](generated/kb_curated/papers-pdf/evo2_2024.pdf) — 1M context StripedHyena
- [GENERator (2024)](generated/kb_curated/papers-pdf/generator_2024.pdf) — 6-mer generative DNA LM

### 🧠 Brain FMs (4 papers)
- [BrainLM (2024)](generated/kb_curated/papers-pdf/brainlm_2024.pdf) — ViT-MAE for fMRI
- [Brain-JEPA (2024)](generated/kb_curated/papers-pdf/brainjepa_2024.pdf) — JEPA for fMRI
- [Brain Harmony (2025)](generated/kb_curated/papers-pdf/brainharmony_2025.pdf) — sMRI+fMRI with TAPE
- [BrainMT (2025)](generated/kb_curated/papers-pdf/brainmt_2025.pdf) — Hybrid Mamba-Transformer

### 🏥 Multimodal Architectures (2 papers)
- [BAGEL (2025)](generated/kb_curated/papers-pdf/bagel_2025.pdf) — Unified multimodal pretraining
- [MoT (2025)](generated/kb_curated/papers-pdf/mot_2025.pdf) — Mixture-of-Transformers

📋 [View all paper card YAMLs →](https://github.com/allison-eunse/neuro-omics-kb/tree/main/kb/paper_cards)

---

## 📊 Data References

- **Governance & QC:** [data/governance_qc.md](data/governance_qc.md)
- **UKB data map & schemas:** [data/ukb_data_map.md](data/ukb_data_map.md)
- **Dataset manifest:** `kb/datasets/ukb_manifest_stub.yaml` (in-repo path; not published)
- **Curated external catalogs:** `kb/datasets/fms_medical_catalog.yaml` for the Awesome Foundation Models roundup
- **(Planned)** Developmental / neurodevelopmental cohort cards (e.g., Cha Hospital longitudinal cohort)

---

## 🗂️ KB Assets (YAML + Curated Sources)

- **Model cards:** `kb/model_cards/`
- **Paper cards:** `kb/paper_cards/`
- **Integration principles:** `docs/generated/kb_curated/integration_cards/`
- **Dataset cards:** `kb/datasets/`
- **Paper source files:** `docs/generated/kb_curated/papers-md/` (markdown summaries) and `docs/generated/kb_curated/papers-pdf/` (original PDFs)

---

## ⚙️ Experiment Configs

Templates in `configs/experiments/`:

- **01_cca_gene_smri.yaml** — CCA + permutation baseline
- **02_prediction_baselines.yaml** — Gene vs sMRI vs Fusion
- **03_logo_gene_attribution.yaml** — LOGO ΔAUC protocol

---

## 🚀 Quick Start Guide

### Baselines
- Z-score → residualize → per-modality 512-D projection → CCA + permutations → LR/GBDT (Gene, Brain, Fusion) → DeLong/bootstrap

### Confounds
- Age, sex, site/scanner, motion (FD), SES, genetic PCs

### Roadmap
- Late fusion first → two-tower contrastive → EI stacking → hub tokens/TAPE if needed

### Quick Test
- Run the CCA + permutation template first; it surfaces cross-modal structure before heavier prediction/fusion experiments.

!!! tip "Why CCA + Permutation?"
    CCA alone will always return non-zero canonical correlations, even on shuffled data. The permutation loop builds a null distribution (re-fitting CCA after shuffling one modality within the train fold) so we can report p-values and avoid over-interpreting noise—critical when cohorts share confounds like site or ancestry.

---

## 🛠️ KB Workflow

### Usage
1. Read paper card YAML for context
2. Check linked code walkthrough for implementation
3. Clone experiment config template
4. Fill dataset paths and parameters
5. Run and log outputs back to KB

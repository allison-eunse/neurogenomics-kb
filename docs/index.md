---
title: Neuro-Omics KB
status: active
updated: 2025-11-19
---

# 🧬🧠 Neuro-Omics Knowledge Base

<p align="center">
  <strong>A comprehensive documentation hub for genetics and brain foundation models and their multimodal integration</strong>
</p>

<p align="center">
  <a href="guide/kb_overview/">📖 KB Overview</a> •
  <a href="models/genetics/">🧬 Genetics Models</a> •
  <a href="models/brain/">🧠 Brain Models</a> •
  <a href="integration/integration_strategy/">🔗 Integration Guide</a> •
  <a href="https://github.com/allison-eunse/neuro-omics-kb">💻 GitHub</a>
</p>

---

!!! success "Welcome!"
    This knowledge base connects **genomics, brain imaging, and behavioral data** through foundation models and multimodal integration strategies. Whether you're analyzing UK Biobank data, implementing gene-brain fusion pipelines, or exploring developmental cohorts, you'll find structured documentation, ready-to-use recipes, and reproducible workflows here.

> **Maintained by** Allison Eun Se You | **Last updated** November 19, 2025 | **Status:** ✅ Documentation Complete

---

## 💡 Use Cases

<div class="grid cards" markdown>

-   :material-dna: **Genetics Research**

    ---

    Extract gene embeddings, analyze variant effects, predict phenotypes from DNA sequences

    [:octicons-arrow-right-24: Genetics Models](models/genetics/)

-   :material-brain: **Brain Imaging**

    ---

    Process fMRI/sMRI, extract neuroimaging features, harmonize multi-site data

    [:octicons-arrow-right-24: Brain Models](models/brain/)

-   :material-link-variant: **Multimodal Integration**

    ---

    Fuse gene + brain embeddings, gene-brain-behavior prediction, cross-modal alignment

    [:octicons-arrow-right-24: Integration Strategy](integration/integration_strategy/)

-   :material-flask: **Reproducible Research**

    ---

    Use validated pipelines, experiment configs, and quality gates for your cohorts

    [:octicons-arrow-right-24: Analysis Recipes](integration/analysis_recipes/cca_permutation/)

</div>

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
    2. Check [decision logs](decisions/2025-11-integration-plan.md)
    3. Review [integration strategy](integration/integration_strategy.md)

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

### General Multimodal
| Model | Type | Key Innovation | Links |
|-------|------|----------------|-------|
| 🔗 [BAGEL](models/multimodal/bagel.md) | Unified FM | MoT experts (understanding + generation) | [Paper](generated/kb_curated/papers-md/bagel_2025.md) • [Walkthrough](code_walkthroughs/bagel_walkthrough.md) |
| 🔗 [MoT](models/multimodal/mot.md) | Sparse transformer | Modality-aware sparsity (~55% FLOPs) | [Paper](generated/kb_curated/papers-md/mot_2025.md) • [Walkthrough](code_walkthroughs/mot_walkthrough.md) |

### Medical & Clinical
| Model | Domain | Specialization | Links |
|-------|--------|----------------|-------|
| 🏥 [M3FM](models/multimodal/m3fm.md) | Radiology | CXR/CT + bilingual reports (EN/CN) | [Paper](generated/kb_curated/papers-md/m3fm_2025.md) • [Walkthrough](code_walkthroughs/m3fm_walkthrough.md) |
| 🏥 [Me-LLaMA](models/multimodal/me_llama.md) | Medical LLM | Continual pretrained LLaMA (129B tokens) | [Paper](generated/kb_curated/papers-md/me_llama_2024.md) • [Walkthrough](code_walkthroughs/melamma_walkthrough.md) |
| 🏥 [TITAN](models/multimodal/titan.md) | Pathology | Gigapixel whole-slide imaging | [Paper](generated/kb_curated/papers-md/titan_2025.md) • [Walkthrough](code_walkthroughs/titan_walkthrough.md) |

📖 **Explore more:**  
[Multimodal Models Overview](models/multimodal/) • [Multimodal Architectures Guide](integration/multimodal_architectures.md) • [Design Patterns](integration/design_patterns.md)

---

## 📚 Research Papers

<details>
<summary><strong>22 Structured Paper Summaries</strong> — Click to expand</summary>

### 🧬 Genetics Foundation Models
| Paper | Year | Key Contribution |
|-------|------|------------------|
| [Caduceus](generated/kb_curated/papers-md/caduceus_2024.md) | 2024 | RC-equivariant bidirectional DNA FM |
| [Evo 2](generated/kb_curated/papers-md/evo2_2024.md) | 2024 | 1M context StripedHyena architecture |
| [GENERator](generated/kb_curated/papers-md/generator_2024.md) | 2024 | 6-mer generative DNA language model |

### 🧠 Brain Foundation Models
| Paper | Year | Key Contribution |
|-------|------|------------------|
| [BrainLM](generated/kb_curated/papers-md/brainlm_2024.md) | 2024 | ViT-MAE for fMRI with site robustness |
| [Brain-JEPA](generated/kb_curated/papers-md/brainjepa_2024.md) | 2024 | Joint-embedding predictive architecture |
| [Brain Harmony](generated/kb_curated/papers-md/brainharmony_2025.md) | 2025 | sMRI+fMRI fusion with TAPE tokens |
| [BrainMT](generated/kb_curated/papers-md/brainmt_2025.md) | 2025 | Hybrid Mamba-Transformer for efficiency |

### 🔗 Multimodal Integration
| Paper | Year | Key Contribution |
|-------|------|------------------|
| [BAGEL](generated/kb_curated/papers-md/bagel_2025.md) | 2025 | Unified understanding + generation with MoT |
| [MoT](generated/kb_curated/papers-md/mot_2025.md) | 2025 | Modality-aware sparse transformers |
| [M3FM](generated/kb_curated/papers-md/m3fm_2025.md) | 2025 | Multilingual medical vision-language |
| [Me-LLaMA](generated/kb_curated/papers-md/me_llama_2024.md) | 2024 | Medical LLM with continual pretraining |
| [TITAN](generated/kb_curated/papers-md/titan_2025.md) | 2025 | Whole-slide pathology vision-language |
| [Multimodal FMs](generated/kb_curated/papers-md/mmfm_2025.md) | 2025 | Survey of multimodal architectures |

### 🧪 Integration Methods & Evaluation
| Paper | Year | Key Contribution |
|-------|------|------------------|
| [Ensemble Integration (Li 2022)](generated/kb_curated/papers-md/ensemble_integration_li2022.md) | 2022 | Late fusion rationale and best practices |
| [Oncology Multimodal (Waqas 2024)](generated/kb_curated/papers-md/oncology_multimodal_waqas2024.md) | 2024 | Confounds and evaluation protocols |
| [Yoon BioKDD 2025](generated/kb_curated/papers-md/yoon_biokdd2025.md) | 2025 | Gene embeddings + LOGO attribution |

### 📊 Genomics & Population Methods
| Paper | Key Contribution |
|-------|------------------|
| [GWAS Diverse Populations](generated/kb_curated/papers-md/gwas_diverse_populations.md) | Ancestry control and bias mitigation |
| [PRS Guide](generated/kb_curated/papers-md/prs_guide.md) | Polygenic risk score methodology |

📋 **Full archive:** [Paper Cards (YAML)](https://github.com/allison-eunse/neuro-omics-kb/tree/main/kb/paper_cards) • [Paper PDFs](https://github.com/allison-eunse/neuro-omics-kb/tree/main/docs/generated/kb_curated/papers-pdf)

</details>

---

## 📊 Data & Schemas

| Resource | Description | Link |
|----------|-------------|------|
| 🏥 **UKB Data Map** | Field mappings, cohort definitions | [View](data/ukb_data_map.md) |
| ✅ **Governance & QC** | Quality control protocols, IRB guidelines | [View](data/governance_qc.md) |
| 🔑 **Subject Keys** | ID management and anonymization | [View](data/subject_keys.md) |
| 📋 **Schemas** | Data format specifications | [View](data/schemas.md) |
| 📦 **FMS-Medical Catalog** | 100+ medical FM references | [View](models/multimodal/fms_medical.md) |

!!! info "Planned Additions"
    Developmental cohort cards (Cha Hospital longitudinal studies) and additional neurodevelopmental datasets coming soon.

---

## 🗂️ KB Assets

<div class="grid cards" markdown>

-   :material-file-document: **Model Cards**

    ---

    YAML metadata for 16 foundation models (genetics, brain, multimodal)

    [:octicons-arrow-right-24: Browse on GitHub](https://github.com/allison-eunse/neuro-omics-kb/tree/main/kb/model_cards)

-   :material-book-open-page-variant: **Paper Cards**

    ---

    Structured summaries of 15+ key papers with integration hooks

    [:octicons-arrow-right-24: Browse on GitHub](https://github.com/allison-eunse/neuro-omics-kb/tree/main/kb/paper_cards)

-   :material-database: **Dataset Cards**

    ---

    Data source specifications for UKB, HCP, and benchmarks

    [:octicons-arrow-right-24: Browse on GitHub](https://github.com/allison-eunse/neuro-omics-kb/tree/main/kb/datasets)

-   :material-link-variant: **Integration Cards**

    ---

    Cross-modal fusion patterns and actionable guidance

    [:octicons-arrow-right-24: Browse on GitHub](https://github.com/allison-eunse/neuro-omics-kb/tree/main/kb/integration_cards)

</div>

---

## ⚙️ Experiment Configs

Ready-to-use analysis templates with validation schemas:

| Template | Purpose | Key Features |
|----------|---------|--------------|
| 📊 **01_cca_gene_smri** | CCA + permutation baseline | Cross-modal null distributions, p-values |
| 🎯 **02_prediction_baselines** | Gene vs Brain vs Fusion | LR/GBDT comparison, DeLong tests |
| 🧬 **03_logo_gene_attribution** | LOGO ΔAUC protocol | Leave-one-gene-out attribution |

[:octicons-arrow-right-24: Browse all configs on GitHub](https://github.com/allison-eunse/neuro-omics-kb/tree/main/configs/experiments)

---

## 🚀 Quick Start Guide

### Standard Pipeline

```mermaid
graph LR
    A[Raw Data] --> B[Z-score normalization]
    B --> C[Residualize confounds]
    C --> D[512-D projection]
    D --> E{Analysis Type}
    E -->|Structure| F[CCA + permutations]
    E -->|Prediction| G[LR/GBDT fusion]
    F --> H[Statistical tests]
    G --> H
    H --> I[Results + validation]
```

### Essential Controls

!!! warning "Always Residualize"
    **Confounds to control:**
    - Age, sex, site/scanner
    - Motion (mean FD for fMRI)
    - SES, genetic PCs
    - Batch effects

### Integration Roadmap

```
Late Fusion (baseline)
    ↓ If fusion wins significantly
Two-Tower Contrastive
    ↓ If gains plateau
EI Stacking / Hub Tokens
    ↓ Last resort
Full Early Fusion (TAPE-style)
```

!!! tip "Start with CCA + Permutation"
    CCA always returns non-zero correlations, even on shuffled data. The permutation test builds a **null distribution** by re-fitting after within-fold shuffling, giving you p-values to avoid over-interpreting noise—critical when sites share confounds.

---

## 🛠️ Typical Workflow

1. **📖 Explore** — Browse model cards and paper summaries
2. **🔍 Select** — Choose appropriate FMs for your modalities
3. **⚙️ Configure** — Clone experiment config template
4. **▶️ Run** — Extract embeddings and run analysis
5. **✅ Validate** — Use quality gates (`manage_kb.py`)
6. **📝 Document** — Log results back to KB

**Need help?** Check the [KB Overview](guide/kb_overview.md) or explore [Code Walkthroughs](code_walkthroughs/index.md)

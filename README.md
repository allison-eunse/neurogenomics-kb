# 🧬🧠 Neuro-Omics Knowledge Base

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://allison-eunse.github.io/neuro-omics-kb/)
[![Models](https://img.shields.io/badge/models-13%20FMs-green)](#foundation-models)
[![Paper Cards](https://img.shields.io/badge/papers-33-orange)](#research-papers)
[![Integration Cards](https://img.shields.io/badge/integration_cards-3-purple)](#integration-strategies)

> **A comprehensive documentation hub for genetics and brain foundation models and their multimodal integration.**

[📖 Read the Docs](https://allison-eunse.github.io/neuro-omics-kb/) | [🚀 Quick Start](https://allison-eunse.github.io/neuro-omics-kb/#getting-started) | [💡 Use Cases](https://allison-eunse.github.io/neuro-omics-kb/#use-cases)

---

## What is this?

A **documentation-first knowledge base** for researchers working with:
- 🧬 **Genetic foundation models** (Caduceus, DNABERT-2, Evo2, GENERator)
- 🧠 **Brain imaging models** (BrainLM, Brain-JEPA, BrainMT, Brain Harmony, SwiFT)
- 🏥 **Multimodal/Clinical models** (BAGEL, MoT, M3FM, Me-LLaMA, TITAN, FMS-Medical)
- 🔗 **Integration strategies** for gene-brain-behavior-language analysis

**Scope:** Documentation, metadata cards, and integration patterns — **not** model implementation code.

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/allison-eunse/neuro-omics-kb.git
cd neuro-omics-kb
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. View documentation locally
mkdocs serve
# Visit http://localhost:8000

# 3. Validate metadata cards
python scripts/manage_kb.py validate models
```

**New to foundation models?** Start with:
1. 📖 [KB Overview](https://allison-eunse.github.io/neuro-omics-kb/guide/kb_overview/) - Understand the structure
2. 🧬 [Genetics Models Overview](https://allison-eunse.github.io/neuro-omics-kb/models/genetics/index.html) - DNA sequence models
3. 🧠 [Brain Models Overview](https://allison-eunse.github.io/neuro-omics-kb/models/brain/index.html) - Neuroimaging models
4. 🔗 [Integration Strategy](https://allison-eunse.github.io/neuro-omics-kb/integration/integration_strategy/) - How to combine modalities

---

## 💡 Use Cases

**This KB supports research across multiple modalities:**

- 🧬 **Genetics Research** - Extract gene embeddings, analyze variant effects, predict phenotypes from DNA sequences
- 🧠 **Brain Imaging** - Process fMRI/sMRI, extract neuroimaging features, harmonize multi-site data
- 🔗 **Multimodal Integration** - Fuse gene + brain embeddings, gene-brain-behavior prediction, cross-modal alignment
- 🏥 **Multimodal & Clinical Models** - Use unified architectures (BAGEL, MoT, M3FM), process medical imaging with clinical text (TITAN, M3FM), leverage medical LLMs (Me-LLaMA)
- 🧪 **Reproducible Research** - Use validated pipelines, experiment configs, and quality gates for your cohorts

**Example workflows:**
- Gene-brain association discovery using WES + sMRI with CCA
- fMRI embedding extraction with BrainLM for MDD prediction
- Leave-one-gene-out (LOGO) attribution for gene importance
- Multimodal fusion for clinical decision support

---

## 📦 What's Inside

<details open>
<summary><b>📚 Documentation (docs/)</b></summary>

- **Code Walkthroughs** - Step-by-step guides for 15 foundation models with consistent formatting
  - 🧬 **Genetics** (4): Caduceus, DNABERT-2, GENERator, Evo 2
  - 🧠 **Brain** (5): BrainLM, Brain-JEPA, Brain Harmony, BrainMT, SwiFT
  - 🏥 **Multimodal/Clinical** (6): BAGEL, MoT, M3FM, Me-LLaMA, TITAN, FMS-Medical catalog
- **Integration Playbooks** - Multimodal fusion strategies (late fusion → contrastive → TAPE)
- **Data Schemas** - UK Biobank, HCP, developmental cohorts
- **Decision Logs** - Architectural choices and research rationale
- **Curated Papers** - PDFs + Markdown summaries in `docs/generated/kb_curated/`

</details>

<details>
<summary><b>🏷️ Metadata Cards (kb/)</b></summary>

- **Model Cards** (`model_cards/*.yaml`) - 15 model cards (13 FMs + 2 ARPA-H planning cards) with architecture specs, embedding recipes, integration hooks
- **Dataset Cards** (`datasets/*.yaml`) - Sample sizes, QC thresholds, access requirements
- **Paper Cards** (`paper_cards/*.yaml`) - 20 research papers with structured takeaways
- **Integration Cards** (`integration_cards/*.yaml`) - Embedding strategies, harmonization methods, preprocessing pipelines

[Browse all cards →](./kb/)

</details>

<details>
<summary><b>🔧 Tools & Scripts</b></summary>

- `scripts/manage_kb.py` - Validate YAML cards, query embedding strategies
- `scripts/codex_gate.py` - Quality gate for automated workflows
- `scripts/fetch_external_repos.sh` - Sync upstream model repositories

</details>

<details>
<summary><b>⚙️ Experiment Configs</b></summary>

Ready-to-run YAML templates in `configs/experiments/`:
- `01_cca_gene_smri.yaml` - CCA + permutation baseline
- `02_prediction_baselines.yaml` - Gene vs Brain vs Fusion
- `03_logo_gene_attribution.yaml` - Gene attribution protocol

</details>

---

## 🎯 Foundation Models

### Genetics Models
| Model | Best for | Context | Documentation |
|-------|----------|---------|---------------|
| 🧬 [Caduceus](https://allison-eunse.github.io/neuro-omics-kb/models/genetics/caduceus/) | RC-equivariant gene embeddings | DNA sequences | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/caduceus_walkthrough/) |
| 🧬 [DNABERT-2](https://allison-eunse.github.io/neuro-omics-kb/models/genetics/dnabert2/) | Cross-species transfer | BPE tokenization | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/dnabert2_walkthrough/) |
| 🧬 [Evo 2](https://allison-eunse.github.io/neuro-omics-kb/models/genetics/evo2/) | Ultra-long regulatory regions | 1M context | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/evo2_walkthrough/) |
| 🧬 [GENERator](https://allison-eunse.github.io/neuro-omics-kb/models/genetics/generator/) | Generative modeling | 6-mer LM | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/generator_walkthrough/) |

### Brain Models
| Model | Modality | Best for | Documentation |
|-------|----------|----------|---------------|
| 🧠 [BrainLM](https://allison-eunse.github.io/neuro-omics-kb/models/brain/brainlm/) | fMRI | Site-robust embeddings | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/brainlm_walkthrough/) |
| 🧠 [Brain-JEPA](https://allison-eunse.github.io/neuro-omics-kb/models/brain/brainjepa/) | fMRI | Lower-latency option | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/brainjepa_walkthrough/) |
| 🧠 [Brain Harmony](https://allison-eunse.github.io/neuro-omics-kb/models/brain/brainharmony/) | sMRI + fMRI | Multi-modal fusion | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/brainharmony_walkthrough/) |
| 🧠 [BrainMT](https://allison-eunse.github.io/neuro-omics-kb/models/brain/brainmt/) | sMRI/fMRI | Mamba efficiency | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/brainmt_walkthrough/) |
| 🧠 [SwiFT](https://allison-eunse.github.io/neuro-omics-kb/models/brain/swift/) | fMRI | Hierarchical spatiotemporal | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/swift_walkthrough/) |

### Multimodal & Clinical Models
| Model | Modalities | Best for | Documentation |
|-------|-----------|----------|---------------|
| 🏥 [BAGEL](https://allison-eunse.github.io/neuro-omics-kb/models/multimodal/bagel/) | Vision + Text + Video | Unified multimodal FM with MoT | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/bagel_walkthrough/) |
| 🏥 [MoT](https://allison-eunse.github.io/neuro-omics-kb/models/multimodal/mot/) | Text + Images + Speech | Sparse mixture-of-transformers | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/mot_walkthrough/) |
| 🏥 [M3FM](https://allison-eunse.github.io/neuro-omics-kb/models/multimodal/m3fm/) | CXR + Text | Multilingual medical reports | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/m3fm_walkthrough/) |
| 🏥 [Me-LLaMA](https://allison-eunse.github.io/neuro-omics-kb/models/multimodal/me_llama/) | Medical Text | LLM for clinical reasoning | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/melamma_walkthrough/) |
| 🏥 [TITAN](https://allison-eunse.github.io/neuro-omics-kb/models/multimodal/titan/) | Histopathology | Whole-slide image analysis | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/titan_walkthrough/) |
| 🏥 [FMS-Medical](https://allison-eunse.github.io/neuro-omics-kb/models/multimodal/fms_medical/) | Clinical Multi-modal | Medical foundation models catalog | [Walkthrough](https://allison-eunse.github.io/neuro-omics-kb/code_walkthroughs/fms_medical_walkthrough/) |

---

## 📋 Research Papers

**33 structured paper cards** documenting:

- 🧬 **Genetics FMs** (5): Caduceus, DNABERT-2, Evo2, GENERator, HyenaDNA
- 🧠 **Brain FMs** (5): BrainLM, Brain-JEPA, Brain Harmony, BrainMT, SwiFT
- 🏥 **Multimodal/Clinical FMs** (6): BAGEL, MoT, M3FM, Me-LLaMA, TITAN, Flamingo
- 🔗 **Integration & Methods** (11): Ensemble integration, Multimodal FMs survey, MM-LLM imaging, Oncology review, Yoon BioKDD, RC-equivariant networks, RC consistency for DNA LMs, Systems & algorithms for multi-hybrid LMs, Brain MRI bias unlearning, Brain multisite harmonization (MURD), Site unlearning (Dinsdale)
- 🧬 **Genomics & Population** (2): GWAS diverse populations, PRS guide
- 📊 **Tabular Baseline** (1): TabPFN
- 📚 **General** (2): Representation learning, Foundation models overview

[View all paper cards →](./kb/paper_cards/) | [Browse summaries →](https://allison-eunse.github.io/neuro-omics-kb/generated/kb_curated/papers-md/)

---

## 🔗 Integration Strategies

**3 comprehensive integration cards** synthesizing multimodal patterns:

- 🎯 **[Ensemble Integration](https://allison-eunse.github.io/neuro-omics-kb/models/integrations/ensemble_integration/)** - Model stacking, averaging, and meta-learning for late fusion
- 🏥 **[Oncology Multimodal Review](https://allison-eunse.github.io/neuro-omics-kb/models/integrations/oncology_multimodal_review/)** - Early/intermediate/late fusion taxonomy from cancer research
- 🎨 **[Multimodal FM Patterns](https://allison-eunse.github.io/neuro-omics-kb/models/integrations/multimodal_fm_patterns/)** - Architectural patterns from BAGEL, MoT, M3FM for Brain-Omics Models

[View integration design patterns →](https://allison-eunse.github.io/neuro-omics-kb/integration/design_patterns/) | [Multimodal architectures →](https://allison-eunse.github.io/neuro-omics-kb/integration/multimodal_architectures/)

---

## 🔗 Reference Commands

### Trace Embedding Strategies
```bash
# Show the full sMRI baseline recipe
python scripts/manage_kb.py ops strategy smri_free_surfer_pca512_v1

# Inspect harmonization metadata (e.g., MURD)
python scripts/manage_kb.py ops harmonization murd_t1_t2
```

### Validate YAML Cards
```bash
python scripts/manage_kb.py validate models
python scripts/manage_kb.py validate datasets
```

### Codex Quality Gate
```bash
# Cycle 1 – quick sanity before giving Codex control
python scripts/codex_gate.py --mode fast --label cycle1 --since origin/main

# Cycle 2 – full sweep before handing work back
python scripts/codex_gate.py --mode full --label cycle2 --since HEAD~1
```

---

## 🌍 Role in Larger Ecosystems

In addition to standalone neurogenomics analyses, **neuro-omics-kb** serves as the **documentation layer** for multimodal brain–omics–LLM foundation model efforts, including ARPA-H–style **Brain-Omics Model (BOM)** initiatives.

### ARPA-H Brain-Omics Model (BOM) Alignment

This KB provides the **foundation for escalating from late fusion → contrastive learning → unified multimodal architectures**:

- **Phase 1 (Current):** Late fusion baselines with genetics + brain FMs
  - Tools: CCA+permutation, prediction baselines, partial correlations
  - Models: Caduceus/DNABERT-2 (genetics) + BrainLM/SwiFT (brain)
  
- **Phase 2 (Near-term):** Two-tower contrastive alignment
  - Patterns: InfoNCE, frozen encoders, small projectors
  - Reference: M3FM, oncology multimodal review
  
- **Phase 3 (Long-term):** Unified Brain-Omics Models
  - Architectures: MoT-style sparse transformers, BAGEL-style unified decoders
  - Integration: Gene-brain-behavior-language tokens with LLM as semantic hub

[Read Integration Plan →](https://allison-eunse.github.io/neuro-omics-kb/decisions/2025-11-integration-plan/) | [View Design Patterns →](https://allison-eunse.github.io/neuro-omics-kb/integration/design_patterns/)

---

## 🔗 Related Repositories

- **Model Implementations**: See links in individual model cards

---

## 📧 Contact

**Maintainer**: Allison Eun Se You  
**Purpose**: Knowledge base for neuro-omics foundation model research  
**Scope**: Documentation, metadata, integration strategies, and references to upstream implementations

---

**Note:** This is a documentation-first KB. Implementation code lives in the upstream repositories referenced throughout `external_repos/` (a mix of tracked snapshots and fetch-on-demand placeholders).

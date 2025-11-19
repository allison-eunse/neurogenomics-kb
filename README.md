# 🧬🧠 Neuro-Omics Knowledge Base

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://allison-eunse.github.io/neuro-omics-kb/)
[![Models](https://img.shields.io/badge/models-9-green)](#foundation-models)
[![Paper Cards](https://img.shields.io/badge/papers-14-orange)](#research-papers)
[![License](https://img.shields.io/badge/license-MIT-purple)]()

> **A comprehensive documentation hub for genetics and brain foundation models and their multimodal integration.**

[📖 Read the Docs](https://allison-eunse.github.io/neuro-omics-kb/) | [🚀 Quick Start](#quick-start) | [💡 Use Cases](#use-cases) | [🤝 Contributing](CONTRIBUTING.md)

---

## What is this?

A **documentation-first knowledge base** for researchers working with:
- 🧬 **Genetic foundation models** (Caduceus, DNABERT-2, Evo2, GENERator)
- 🧠 **Brain imaging models** (BrainLM, Brain-JEPA, BrainMT, Brain Harmony, SwiFT)
- 🔗 **Multimodal integration** strategies for gene-brain-behaviour analysis

**Scope:** Documentation, metadata cards, and integration strategies — **not** model implementation code.

**Key cohorts:** UK Biobank, HCP, developmental/neurodevelopmental populations

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
2. 🧬 [Genetics Models Overview](https://allison-eunse.github.io/neuro-omics-kb/models/genetics/) - DNA sequence models
3. 🧠 [Brain Models Overview](https://allison-eunse.github.io/neuro-omics-kb/models/brain/) - Neuroimaging models
4. 🔗 [Integration Strategy](https://allison-eunse.github.io/neuro-omics-kb/integration/integration_strategy/) - How to combine modalities

---

## 💡 Use Cases

**This KB helps you:**

✅ **Understand foundation models** - Detailed walkthroughs for 9+ models with integration hooks  
✅ **Design multimodal studies** - CCA, late fusion, contrastive learning recipes  
✅ **Reproduce analyses** - Versioned embedding strategies, harmonization methods, experiment configs  
✅ **Navigate datasets** - Structured cards for UK Biobank, HCP, developmental cohorts  
✅ **Track decisions** - Decision logs documenting why certain approaches were chosen  

**Example workflows:**
- Gene-brain association discovery using WES + sMRI
- fMRI embedding extraction with BrainLM for MDD prediction
- Leave-one-gene-out (LOGO) attribution for gene importance

---

## 📦 What's Inside

<details open>
<summary><b>📚 Documentation (docs/)</b></summary>

- **Code Walkthroughs** - Step-by-step guides for 15 foundation models
  - 🧬 Genomics: Caduceus, DNABERT-2, GENERator, Evo 2
  - 🧠 Brain: BrainLM, Brain-JEPA, Brain Harmony, BrainMT, SwiFT
  - 🏥 Multimodal/Clinical: M3FM, Me-LLaMA, TITAN, BAGEL, MoT, FMS-Medical catalog
- **Integration Playbooks** - Multimodal fusion strategies (late fusion → contrastive → TAPE)
- **Data Schemas** - UK Biobank, HCP, developmental cohorts
- **Decision Logs** - Architectural choices and research rationale
- **Curated Papers** - PDFs + Markdown summaries in `docs/generated/kb_curated/`

</details>

<details>
<summary><b>🏷️ Metadata Cards (kb/)</b></summary>

- **Model Cards** (`model_cards/*.yaml`) - 9 FMs with architecture specs, embedding recipes, integration hooks
- **Dataset Cards** (`datasets/*.yaml`) - Sample sizes, QC thresholds, access requirements
- **Paper Cards** (`paper_cards/*.yaml`) - 14 research papers with structured takeaways
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

---

## 📋 Research Papers

**14 structured paper cards** documenting:

- 🔗 **Integration principles** (5): Late fusion, multimodal oncology, MDD genes, PRS, GWAS
- 🧬 **Genetics FMs** (3): Caduceus, Evo2, GENERator
- 🧠 **Brain FMs** (4): BrainLM, Brain-JEPA, Brain Harmony, BrainMT
- 🏥 **Multimodal architectures** (2): BAGEL, MoT

[View all paper cards →](./kb/paper_cards/)

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

In addition to standalone neurogenomics analyses, **neuro-omics-kb** serves as the **neurogenomics documentation layer** inside larger multimodal brain–omics–LLM FM efforts (e.g., ARPA-H–style **Brain-Omics Model (BOM)**). It documents models, datasets, and integration recipes so that **gene–brain–behaviour FMs**—spanning adult and developmental cohorts, MRI/fMRI, EEG/EPhys, genetics, behavioural/developmental assessments, and language—can be scaled, compared, and reproduced across cohorts and projects.

---

## 🔗 Related Repositories

- **PDF/Markdown Converter**: [pdf-md-ai-summaries](https://github.com/allison-eunse/pdf-md-ai-summaries)
- **Model Implementations**: See links in individual model cards
- **Datasets**: UK Biobank (restricted), HCP, OpenGenome2

---

## 📧 Contact

**Maintainer**: Allison Eun Se You  
**Purpose**: Knowledge base for neuro-omics foundation model research  
**Scope**: Documentation, metadata, integration strategies (no implementation)

---

**Note:** This is a **knowledge base only** - no implementation code. For actual model training/inference, refer to the `external_repos/` directories or the original model repositories.

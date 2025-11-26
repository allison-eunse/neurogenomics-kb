# 🧬 Genetics Foundation Models

> **DNA sequence foundation models for genomic representation learning**

This section documents the **genetics foundation models** used to extract gene-level embeddings from raw genomic sequences (DNA/RNA) for downstream integration with brain imaging, behavioral phenotypes, and clinical outcomes.

---

## 📋 Overview

All genetics FMs documented here:

- **Operate on nucleotide sequences** (A, C, G, T) rather than pre-computed variant calls or SNP arrays
- **Support gene-level embeddings** via forward/reverse-complement (RC) averaging and pooling strategies
- **Enable interpretability** through attribution methods like LOGO ΔAUC
- **Are pretrained on large genomic corpora** (human reference genomes, multi-species datasets, or RefSeq)

---

## 🎯 Model Registry

| Model | Architecture | Key Feature | Integration Role | Documentation |
|-------|-------------|-------------|------------------|---------------|
| 🧬 [**Caduceus**](caduceus.md) | Mamba (BiMamba) + RC-equivariance | Strand-robust, efficient long-context | Primary gene encoder for UK Biobank WES | [Walkthrough](../../code_walkthroughs/caduceus_walkthrough.md) |
| 🧬 [**DNABERT-2**](dnabert2.md) | BERT (multi-species) | BPE tokenization, cross-species transfer | Alternative gene encoder; comparison baseline | [Walkthrough](../../code_walkthroughs/dnabert2_walkthrough.md) |
| 🧬 [**Evo 2**](evo2.md) | StripedHyena (1M context) | Ultra-long-range dependencies | Exploratory; regulatory region capture | [Walkthrough](../../code_walkthroughs/evo2_walkthrough.md) |
| 🧬 [**GENERator**](generator.md) | Generative 6-mer LM | Generative modeling, sequence design | Reference for generative vs discriminative | [Walkthrough](../../code_walkthroughs/generator_walkthrough.md) |
| 🧬 [**HyenaDNA**](hyenadna.md) | Hyena implicit convolutions (1M context) | Single-nucleotide, ultra-long genomic modeling | Conceptual long-context genomics reference | [Architecture walkthrough](../../code_walkthroughs/hyena_walkthrough.md) |

---

## 🔄 Usage Workflow

1. **Extract sequences** from reference genome (hg38) for target genes
2. **Tokenize** using model-specific vocabularies (k-mers, BPE, or single-nucleotide)
3. **Embed** forward and reverse-complement sequences
4. **Pool** to gene-level representation (mean/CLS depending on model)
5. **Project** to 512-D for cross-modal alignment with brain embeddings

---

## 🔑 Key Considerations

### RC-equivariance
DNA has no inherent directionality; models like **Caduceus** enforce BiMamba RC-equivariance to avoid strand bias. For non-equivariant models, manually average forward and RC embeddings.

### Variant handling
Foundation models operate on **reference sequences by default**. To incorporate subject-specific variants:

- Patch reference with VCF alleles
- Re-embed variant sequences
- Compare ΔAUC between reference and variant embeddings (exploratory)

### Attribution
Use **LOGO (Leave-One-Gene-Out)** ΔAUC to assess which genes contribute most to downstream prediction tasks (e.g., MDD risk, cognitive scores). See [Yoon et al. BioKDD 2025](https://raw.githubusercontent.com/allison-eunse/neuro-omics-kb/main/docs/generated/kb_curated/papers-pdf/yoon_biokdd2025.pdf) for protocol details.

---

## 🔗 Integration Targets

Genetics embeddings are integrated with:

- **sMRI** IDPs (structural phenotypes) via CCA, late fusion, or contrastive alignment
- **fMRI** embeddings (e.g., BrainLM, Brain-JEPA) for gene–brain–behaviour triangulation
- **Behavioral phenotypes** (cognitive scores, psychiatric diagnoses) via multimodal prediction

**Learn more:**
- [Integration Strategy](../../integration/integration_strategy.md) - Fusion protocols
- [Modality Features: Genomics](../../integration/modality_features/genomics.md) - Preprocessing specs

---

## 📦 Source Repositories

<details>
<summary><b>Click to view all source repositories</b></summary>

All genetics FM source code lives in `external_repos/`:

| Model | Local Path | GitHub Repository |
|-------|------------|-------------------|
| Caduceus | `external_repos/caduceus/` | [kuleshov-group/caduceus](https://github.com/kuleshov-group/caduceus) |
| DNABERT-2 | `external_repos/dnabert2/` | [Zhihan1996/DNABERT2](https://github.com/Zhihan1996/DNABERT2) |
| Evo 2 | `external_repos/evo2/` | [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2) |
| GENERator | `external_repos/generator/` | [GenerTeam/GENERator](https://github.com/GenerTeam/GENERator) |

Each model page includes:
- ✅ Detailed code walkthrough in `docs/code_walkthroughs/`
- ✅ Structured YAML card in `kb/model_cards/`
- ✅ Integration recipes and preprocessing specs

</details>

---

## 🚀 Next Steps

- ✅ Validate gene embedding reproducibility across cohorts (UK Biobank WES, Cha Hospital panel sequencing)
- ✅ Benchmark LOGO ΔAUC stability under different embedding projection dimensions (256, 512, 1024)
- 🔬 Explore regulatory region embeddings (enhancers, promoters) with long-context models like Evo 2

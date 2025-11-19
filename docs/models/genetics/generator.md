---
title: GENERator — Model Card
status: active
updated: 2025-11-19
---

# GENERator

## Overview

**Type:** Generative DNA language model  
**Architecture:** 6-mer-based autoregressive transformer  
**Modality:** Nucleotide sequences (DNA)  
**Primary use:** Generative modeling and sequence design (with discriminative embedding extraction)

## Purpose & Design Philosophy

GENERator is a **generative DNA language model** trained on RefSeq and other genomic corpora using 6-mer tokenization. While primarily designed for sequence generation and design tasks (e.g., synthetic promoter optimization), its learned representations can be extracted for discriminative tasks like gene embedding and downstream fusion.

**Key innovation:** 6-mer vocabulary balances computational tractability with sufficient resolution to capture regulatory motifs and codon structure.

## Architecture Highlights

- **Backbone:** Autoregressive transformer (GPT-style)
- **Tokenization:** 6-mer overlapping windows (4096-token vocabulary)
- **Pretraining:** Next-token prediction on human RefSeq + genomic corpora
- **Generative objective:** Likelihood maximization for sequence generation
- **Output:** Generative logits (design mode) or hidden states (embedding mode)

## Integration Strategy

### For Neuro-Omics KB

**Embedding recipe:** `genetics_gene_fm_pca512_v1` (GENERator variant)
- Extract gene sequences from hg38 reference genome
- Tokenize with 6-mer overlapping windows
- Forward pass → extract **hidden states** (not generative logits)
- **RC handling:** GENERator not RC-equivariant → average forward/RC embeddings
- Mean pool over gene length → gene-level vector
- Concatenate target gene set
- Project to 512-D via PCA
- Residualize: age, sex, ancestry PCs, batch

**Fusion targets:**
- **Gene-brain alignment:** Late fusion with brain FM embeddings
- **Generative vs. discriminative:** Compare GENERator embeddings to Caduceus/DNABERT-2
- **Sequence design (exploratory):** Generate synthetic regulatory elements with desired properties

### For ARPA-H Brain-Omics Models

GENERator demonstrates **generative modeling** for biological sequences:
- Hidden states from generative models can serve as discriminative features
- Generative capability enables counterfactual analysis ("what if this gene sequence changed?")
- 6-mer tokenization preserves codon structure for coding sequence analysis
- Blueprint for generative components in multimodal Brain-Omics Model (BOM)

## Embedding Extraction Workflow

```bash
# Discriminative mode (embeddings)
# 1. Extract gene sequences (hg38 reference)
# 2. Tokenize with 6-mer overlapping windows
# 3. Load pretrained GENERator checkpoint
# 4. Forward pass → extract hidden states (not output logits)
# 5. RC-average: embed(seq) and embed(reverse_complement(seq))
# 6. Mean pool over tokens → gene embedding
# 7. Log: token_vocabulary, pooling_layer (e.g., layer -1)

# Generative mode (sequence design)
# 1. Define target properties (e.g., GC content, expression level)
# 2. Sample from GENERator with conditioning
# 3. Validate generated sequences via wet-lab or in-silico assays
```

## Strengths & Limitations

### Strengths
- **Generative capability:** Can design novel sequences (regulatory elements, synthetic genes)
- **6-mer vocabulary:** Preserves codon structure, captures motifs
- **Hidden states useful:** Discriminative embeddings competitive with specialized models
- **Interpretable:** Generative likelihoods inform sequence quality

### Limitations
- **Not RC-equivariant:** Requires manual forward/RC averaging
- **Generative objective:** Optimized for likelihood, not discriminative tasks
- **Checkpoint availability:** Fewer public weights vs. DNABERT-2
- **6-mer limitations:** May miss patterns spanning >6 bases (compare to BPE or longer k-mers)

## When to Use GENERator

✅ **Use when:**
- Interested in generative modeling and sequence design
- Want to compare generative vs. discriminative embeddings
- Need 6-mer vocabulary (codon-aware analysis)
- Exploring counterfactual sequence perturbations

⚠️ **Consider alternatives:**
- **Caduceus:** For discriminative tasks with RC-equivariance
- **DNABERT-2:** BPE tokenization, stronger discriminative benchmarks
- **Evo2:** For ultra-long regulatory contexts

## Reference Materials

**Primary sources:**
- **Paper:** [GENERator (2024)](../../generated/kb_curated/papers-pdf/generator_2024.pdf)
- **Code walkthrough:** [GENERator walkthrough](../../code_walkthroughs/generator_walkthrough.md)
- **YAML card:** `kb/model_cards/generator.yaml`
- **Paper card:** `kb/paper_cards/generator_2024.yaml`

**Integration recipes:**
- [Modality Features: Genomics](../../integration/modality_features/genomics.md)
- [Integration Strategy](../../integration/integration_strategy.md)

**Source repository:**
- **Local:** `external_repos/generator/`
- **GitHub:** [GenerTeam/GENERator](https://github.com/GenerTeam/GENERator)

## Next Steps in Our Pipeline

1. **Discriminative benchmark:** Compare GENERator vs. Caduceus/DNABERT-2 on same gene set
2. **Generative pilot:** Design synthetic promoters, test expression predictions
3. **Counterfactual analysis:** Perturb gene sequences, measure embedding Δ
4. **6-mer analysis:** Visualize learned k-mer representations
5. **ARPA-H vision:** Explore generative components for Brain-Omics Model (BOM)

## Engineering Notes

- **Extract hidden states, not logits** for discriminative embeddings
- Always **RC-average** forward and reverse-complement embeddings
- Log **layer used** for extraction (typically last layer before output)
- 6-mer tokenization is **deterministic** but frame-dependent (start position matters)
- When generating sequences, validate via independent predictors (avoid model collapse)

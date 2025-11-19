---
title: Caduceus — Model Card
status: active
updated: 2025-11-19
---

# Caduceus

## Overview

**Type:** RC-equivariant DNA foundation model  
**Architecture:** BiMamba (bidirectional Mamba) + Hyena  
**Modality:** Nucleotide sequences (DNA/RNA)  
**Primary use:** Strand-robust gene-level embeddings for multimodal fusion

## Purpose & Design Philosophy

Caduceus enforces **reverse-complement (RC) equivariance** through bidirectional Mamba/Hyena layers, ensuring embeddings are invariant to DNA strand orientation. This addresses a fundamental biological constraint: DNA has no inherent directionality, yet many language model architectures introduce strand bias. Caduceus learns sequence grammar while respecting this symmetry.

**Key innovation:** RC-equivariance as architectural constraint (not post-hoc averaging) → more parameter-efficient and biologically principled.

## Architecture Highlights

- **Backbone:** BiMamba (Mamba blocks with bidirectional RC scanning) or RC-augmented Hyena
- **RC enforcement:** Built into attention/convolution layers
- **Input:** Raw nucleotide sequences (A, C, G, T) with k-mer or single-base tokenization
- **Pretraining:** Masked language modeling on large genomic corpora (human + multi-species)
- **Output:** Per-position embeddings → gene-level via mean pooling

## Integration Strategy

### For Neuro-Omics KB

**Embedding recipe:** `genetics_gene_fm_pca512_v1` (Caduceus variant)
- Extract gene sequences from reference genome (hg38)
- Tokenize with Caduceus vocabulary (typically 4-mer or 6-mer)
- Forward pass → per-nucleotide embeddings
- **RC hygiene:** Caduceus natively RC-equivariant, but verify with sanity check (forward == RC)
- Mean pool over gene length → gene-level vector
- Concatenate target gene set (e.g., 38 MDD genes from Yoon et al.)
- Project to 512-D via PCA
- Residualize: age, sex, ancestry PCs (1-10), batch

**Fusion targets:**
- **Gene-brain CCA:** Align with BrainLM/Brain-JEPA embeddings
- **LOGO attribution:** Leave-one-gene-out ΔAUC for gene importance (Yoon et al. protocol)
- **Variant impact:** Compare reference vs. subject-specific sequences (exploratory)

### For ARPA-H Brain-Omics Models

Caduceus provides **strand-robust genetic representations** for Brain-Omics systems:
- RC-equivariance critical when sequences are sampled from forward or reverse strands
- Gene embeddings can be projected into shared LLM/VLM spaces for cross-modal reasoning
- Efficient Mamba architecture scales to whole-genome or regulatory region analysis
- Natural encoder for "genetic modality" in unified multimodal Brain-Omics Model (BOM)

## Embedding Extraction Workflow

```bash
# 1. Extract gene sequences from hg38 reference (GENCODE annotations)
# 2. Tokenize with Caduceus vocabulary (e.g., 6-mer overlapping)
# 3. Load pretrained Caduceus checkpoint
# 4. Forward pass → per-position embeddings
# 5. Verify RC equivariance (optional but recommended):
#    embed(seq) ≈ embed(reverse_complement(seq))
# 6. Mean pool over gene → gene-level vector
# 7. Concatenate gene set → subject genotype embedding
# 8. Log: gene_list, reference_version, embedding_strategy_id
```

## Strengths & Limitations

### Strengths
- **RC-equivariant by design:** No manual averaging needed
- **Parameter efficient:** Mamba/Hyena scale better than full attention for long sequences
- **Strong benchmarks:** Competitive performance on regulatory element prediction, variant effect
- **Interpretable:** Attention/conv patterns respect biological constraints

### Limitations
- **Requires RC-aware tokenization:** Some vocabularies break RC symmetry (use carefully)
- **Limited to reference sequences:** Variant handling requires re-embedding (computationally expensive)
- **Checkpoint availability:** Fewer pretrained scales vs. DNABERT-2 or ESM-style models
- **K-mer choice matters:** Different tokenizations yield different embedding quality

## When to Use Caduceus

✅ **Use when:**
- Need strand-robust gene embeddings for UKB/Cha Hospital genetics
- Prioritizing parameter efficiency for long sequences (>10kb genes)
- Want architectural RC-equivariance (not post-hoc correction)
- Implementing LOGO attribution (Yoon et al. protocol)

⚠️ **Consider alternatives:**
- **DNABERT-2:** BPE tokenization, more public checkpoints, cross-species pretraining
- **Evo2:** Ultra-long context (1M tokens) for regulatory regions
- **GENERator:** Generative modeling if sequence design is goal

## Reference Materials

### Knowledge Base Resources

**Curated materials in this KB:**
- **Paper summary & notes (PDF):** [Caduceus (2024)](../../generated/kb_curated/papers-pdf/caduceus_2024.pdf)
- **Paper card (YAML):** `kb/paper_cards/caduceus_2024.yaml`
- **Code walkthrough:** [Caduceus walkthrough](../../code_walkthroughs/caduceus_walkthrough.md)
- **Model card (YAML):** `kb/model_cards/caduceus.yaml`

**Integration recipes:**
- [Modality Features: Genomics](../../integration/modality_features/genomics.md)
- [Integration Strategy](../../integration/integration_strategy.md)
- [CCA + Permutation](../../integration/analysis_recipes/cca_permutation.md)
- [LOGO Attribution](https://github.com/allison-eunse/neuro-omics-kb/blob/main/configs/experiments/03_logo_gene_attribution.yaml) (experiment config)

### Original Sources

**Source code repositories:**
- **Local copy:** `external_repos/caduceus/`
- **Official GitHub:** [kuleshov-group/caduceus](https://github.com/kuleshov-group/caduceus)

**Original paper:**
- **Title:** "Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling"
- **Authors:** Schiff et al.
- **Published:** arXiv preprint, March 2024
- **Link:** [arXiv:2403.03234](https://arxiv.org/abs/2403.03234)
- **PDF (local):** [caduceus_2024.pdf](../../generated/kb_curated/papers-pdf/caduceus_2024.pdf)

## Next Steps in Our Pipeline

1. **UKB WES extraction:** Embed 38 MDD genes + cognition-related genes from UK Biobank
2. **RC verification:** Sanity check embed(seq) == embed(RC(seq)) on test genes
3. **Gene-brain CCA:** Align Caduceus embeddings with BrainLM fMRI vectors
4. **LOGO protocol:** Implement leave-one-gene-out ΔAUC (Yoon et al. BioKDD'25)
5. **Variant exploration:** Test impact of subject-specific SNPs on embeddings (pilot)

## Engineering Notes

- Always **log k-mer size** and **tokenization strategy** in metadata
- Verify RC-equivariance on held-out genes before scaling to full cohort
- When comparing to DNABERT-2, use same gene set and reference version
- For attribution: LOGO requires nested CV to avoid leakage (see Yoon et al. protocol)

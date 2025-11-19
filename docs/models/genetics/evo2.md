---
title: Evo 2 — Model Card
status: active
updated: 2025-11-19
---

# Evo 2

## Overview

**Type:** Ultra-long-context DNA foundation model  
**Architecture:** StripedHyena 2 (Hyena + attention)  
**Modality:** Nucleotide sequences (DNA/RNA)  
**Primary use:** Regulatory region embeddings with 1M token context

## Purpose & Design Philosophy

Evo2 extends DNA foundation models to **1 million token contexts** using StripedHyena 2 architecture (hybrid Hyena operators + attention layers). This enables modeling entire genes with full regulatory context (promoters, enhancers, 3D loop anchors) in a single forward pass, capturing long-range genomic interactions that shorter-context models miss.

**Key innovation:** 1M context via sub-quadratic Hyena operators → whole-locus modeling including distal regulatory elements.

## Architecture Highlights

- **Backbone:** StripedHyena 2 (alternating Hyena convolution + multi-head attention)
- **Context length:** 1,048,576 tokens (~1 Mb of genomic sequence)
- **Tokenization:** Single-base or 2-mer/4-mer (preserves fine resolution)
- **Pretraining:** Masked LM on human + multi-species genomes
- **Output:** Per-position embeddings → region/gene pooling

## Integration Strategy

### For Neuro-Omics KB

**Embedding recipe:** `genetics_regulatory_evo2_v1` (exploratory)
- Extract extended gene loci (gene + 100kb upstream/downstream for regulatory context)
- Tokenize with Evo2 vocabulary (typically single-base or 2-mer)
- Forward pass → per-position embeddings for full locus
- **RC handling:** Evo2 not explicitly RC-equivariant → average forward/RC embeddings
- Pool over gene CDS → gene embedding
- Optionally extract regulatory region embeddings (promoter, enhancers) separately
- Project to 512-D via PCA
- Residualize: age, sex, ancestry PCs, batch

**Fusion targets:**
- **Gene expression prediction:** Regulatory context improves gene-phenotype links
- **Enhancer-gene mapping:** Identify distal elements affecting brain-expressed genes
- **3D genome modeling:** Capture loop anchors and TAD boundaries (exploratory)

### For ARPA-H Brain-Omics Models

Evo2 enables **whole-locus genetic representations** for Brain-Omics systems:
- 1M context captures regulatory grammar spanning hundreds of kilobases
- Critical for brain-specific enhancers distant from target genes
- Can embed entire pathways or multi-gene clusters in single pass
- Blueprint for ultra-long-context multimodal architectures (e.g., long-range EEG patterns)

## Embedding Extraction Workflow

```bash
# 1. Extract extended loci (gene ± 100kb from hg38)
# 2. Tokenize with Evo2 single-base or k-mer vocabulary
# 3. Load pretrained Evo2 checkpoint
# 4. Forward pass (may require chunking if >1M tokens)
# 5. Extract embeddings for:
#    - Gene CDS (coding sequence)
#    - Promoter (-2kb to TSS)
#    - Predicted enhancers (if annotated)
# 6. RC-average forward + reverse-complement
# 7. Pool each region → separate vectors or concatenate
# 8. Log: context_length, regulatory_elements_included
```

## Strengths & Limitations

### Strengths
- **Ultra-long context:** 1M tokens captures distal regulatory elements
- **Whole-locus modeling:** No need to manually select regulatory windows
- **Sub-quadratic scaling:** Hyena operators enable long context without full attention cost
- **Regulatory grammar:** Can learn enhancer-promoter interactions end-to-end

### Limitations
- **Massive memory footprint:** 1M context requires high-memory GPUs (80GB+ A100/H100)
- **Slower inference:** Even with Hyena, 1M tokens slower than short-context models
- **Overkill for coding sequences:** Most genes <10kb don't need 1M context
- **Checkpoint availability:** Fewer public weights vs. DNABERT-2/Caduceus

## When to Use Evo2

✅ **Use when:**
- Need regulatory context for brain-specific gene expression
- Studying long-range enhancer-promoter interactions
- Have sufficient compute (80GB+ GPU, large batch sizes)
- Exploring 3D genome structure embeddings

⚠️ **Defer until:**
- Caduceus/DNABERT-2 baselines complete
- Regulatory element analysis becomes critical
- GPU resources available for long-context experiments

⚠️ **Consider alternatives:**
- **Caduceus:** For coding sequences without regulatory context
- **DNABERT-2:** For standard gene embeddings with manageable compute
- **GENERator:** If generative modeling is priority

## Reference Materials

### Knowledge Base Resources

**Curated materials in this KB:**
- **Paper summary & notes (PDF):** [Evo2 (2024)](../../generated/kb_curated/papers-pdf/evo2_2024.pdf)
- **Code walkthrough:** [Evo2 walkthrough](../../code_walkthroughs/evo2_walkthrough.md)
- **Model card (YAML):** `kb/model_cards/evo2.yaml`
- **Paper card (YAML):** `kb/paper_cards/evo2_2024.yaml`

**Integration recipes:**
- [Modality Features: Genomics](../../integration/modality_features/genomics.md)
- [Integration Strategy](../../integration/integration_strategy.md)

### Original Sources

**Source code repositories:**
- **Local copy:** `external_repos/evo2/`
- **Official GitHub:** [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2)

**Original paper:**
- **Title:** "Genome modeling and design across all domains of life with Evo 2"
- **Authors:** Arc Institute Team
- **Published:** bioRxiv preprint, February 2025
- **Link:** [bioRxiv:2025.02.18.638918](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1)
- **PDF (local):** [evo2_2024.pdf](../../generated/kb_curated/papers-pdf/evo2_2024.pdf)

## Next Steps in Our Pipeline

1. **Pilot study:** Embed 5-10 brain-expressed genes with known distal enhancers
2. **Context ablation:** Test 10kb vs. 100kb vs. 1M context for gene-brain CCA
3. **Memory profiling:** Document GPU requirements and chunking strategies
4. **Enhancer-gene links:** Compare Evo2 regulatory embeddings vs. eQTL databases
5. **ARPA-H vision:** Explore Evo2-style long context for other modalities (EEG, longitudinal)

## Engineering Notes

- **GPU requirements:** 80GB+ A100 or H100 for full 1M context
- Chunk long sequences if needed; aggregate chunk embeddings carefully
- Log **context length used** (may be <1M for most genes)
- RC-averaging doubles compute; consider caching forward embeddings
- When comparing to short-context models, isolate regulatory contribution via ablation

---
title: Brain Harmony — Model Card
status: active
updated: 2025-11-19
---

# Brain Harmony

## Overview

**Type:** Multi-modal brain foundation model  
**Architecture:** ViT + TAPE (Temporal Adaptive Patch Embedding)  
**Modalities:** sMRI + fMRI (unified)  
**Primary use:** Cross-modal brain embeddings for heterogeneous cohorts

## Purpose & Design Philosophy

Brain Harmony addresses a critical challenge in multi-site neuroimaging: **heterogeneous TRs and scanning protocols**. By introducing TAPE (Temporal Adaptive Patch Embedding), the model resizes temporal tokens to a fixed duration τ, enabling unified processing of fMRI data with variable repetition times. Hub tokens fuse sMRI and fMRI modalities into a shared representation space.

**Key innovation:** TAPE + hub tokens allow robust multimodal fusion even when different sites use different TR/scanner configurations.

## Architecture Highlights

- **Backbone:** Vision Transformer with TAPE for fMRI, standard patches for sMRI
- **TAPE mechanism:** Resizes temporal patches to fixed τ duration regardless of TR
- **Hub tokens:** Cross-modal attention for sMRI ↔ fMRI fusion
- **Input:** T1w structural scans + parcel time series
- **Output:** Unified multimodal subject embeddings

## Integration Strategy

### For Neuro-Omics KB

**Embedding recipe:** `multimodal_brain_harmony_v1`
- Extract both sMRI and fMRI features through shared encoder
- Hub tokens aggregate cross-modal information
- Project to 512-D unified representation
- Residualize: age, sex, site, scanner, ICV (sMRI), mean FD (fMRI)

**Fusion targets:**
- **Gene-brain-behavior triangulation:** Single unified brain vector + genomics
- **Multi-site robustness:** Critical for UKB + Cha Hospital + ABCD combinations
- **Developmental trajectories:** Handle TR changes across pediatric age ranges

### For ARPA-H Brain-Omics Models

Brain Harmony exemplifies **modality-adaptive fusion** for Brain-Omics systems:
- TAPE-style mechanisms can extend to other time-varying modalities (EEG, longitudinal behavior)
- Hub tokens provide blueprint for cross-modal attention in gene-brain-language models
- TR heterogeneity handling essential for federated Brain-Omics Model (BOM) deployment

## Embedding Extraction Workflow

```bash
# 1. Preprocess sMRI → FreeSurfer / volumetric tensor
# 2. Preprocess fMRI → parcellate + retain TR metadata
# 3. Load pretrained Brain Harmony checkpoint
# 4. Forward pass with TAPE temporal adaptation
# 5. Extract hub token embeddings (not individual modality tokens)
# 6. Project to 512-D if needed
# 7. Log embedding_strategy ID + TR range in metadata
```

## Strengths & Limitations

### Strengths
- **TR heterogeneity handling:** TAPE critical for multi-site/longitudinal studies
- **Multi-modal fusion:** Native sMRI+fMRI joint embeddings
- **Hub token architecture:** Flexible attention mechanism for modality integration
- **Practical engineering:** Addresses real-world scanning protocol variations

### Limitations
- **Higher complexity:** TAPE + hub tokens increase training/inference cost
- **Engineering overhead:** More complex than single-modality encoders
- **Limited public checkpoints:** Newer model, fewer pretrained weights available
- **Overkill for homogeneous cohorts:** If TR is fixed, simpler models may suffice

## When to Use Brain Harmony

✅ **Use when:**
- Combining UKB (TR=0.72s) + HCP (TR=0.72s) + Cha Hospital (TR=TBD) + ABCD (TR=0.8s)
- Need both structural and functional information in single embedding
- Site/scanner heterogeneity limits other approaches
- Preparing for ARPA-H-style federated multimodal systems

⚠️ **Consider alternatives:**
- **Late fusion (BrainLM + FreeSurfer):** Simpler baseline if TR is homogeneous
- **BrainMT:** If temporal modeling more critical than structural integration
- **SwiFT:** For 4D volumetric approaches without explicit parcellation

## Reference Materials

**Primary sources:**
- **Paper:** [Brain Harmony (2025)](../../generated/kb_curated/papers-pdf/brainharmony_2025.pdf)
- **Code walkthrough:** [Brain Harmony walkthrough](../../code_walkthroughs/brainharmony_walkthrough.md)
- **YAML card:** `kb/model_cards/brainharmony.yaml`
- **Paper card:** `kb/paper_cards/brainharmony_2025.yaml`

**Integration recipes:**
- [Modality Features: sMRI](../../integration/modality_features/smri.md)
- [Modality Features: fMRI](../../integration/modality_features/fmri.md)
- [Integration Strategy](../../integration/integration_strategy.md)

**Source repository:**
- **Local:** `external_repos/brainharmony/`
- **GitHub:** [hzlab/Brain-Harmony](https://github.com/hzlab/Brain-Harmony)

## Next Steps in Our Pipeline

1. **TR profiling:** Document TR distributions across UKB/Cha Hospital/ABCD
2. **Baseline comparison:** Brain Harmony vs. late fusion of BrainLM+FreeSurfer
3. **Hub token analysis:** Visualize what cross-modal patterns hub tokens capture
4. **Gene-multimodal-brain CCA:** Test whether unified embeddings improve genetics alignment
5. **ARPA-H scalability:** Evaluate TAPE mechanism for EEG time-varying modalities

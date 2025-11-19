---
title: BrainLM — Model Card
status: active
updated: 2025-11-19
---

# BrainLM

## Overview

**Type:** Self-supervised foundation model for fMRI  
**Architecture:** Vision Transformer with Masked Autoencoding (ViT-MAE)  
**Modality:** Functional MRI (parcel time series)  
**Primary use:** Subject-level embeddings for downstream prediction tasks

## Purpose & Design Philosophy

BrainLM applies masked autoencoding to fMRI parcel time series, learning site-invariant brain representations through large-scale multi-cohort pretraining (UK Biobank + HCP). The model reconstructs masked parcels across time, forcing the encoder to capture functional relationships and temporal dynamics without relying on task-specific supervision.

**Key innovation:** Site-robust pretraining enables strong linear probe performance and generalization across diverse cohorts.

## Architecture Highlights

- **Backbone:** ViT-MAE with spatial-temporal masking
- **Input:** Parcel time series (e.g., Schaefer-400 @ TR=0.72s)
- **Pretraining:** Mask random parcels/timepoints → reconstruct from latent tokens
- **Output:** Subject-level embeddings via mean pooling over latent tokens

## Integration Strategy

### For Neuro-Omics KB

**Embedding recipe:** `rsfmri_brainlm_segments_v1`
- Extract latent embeddings from pretrained encoder
- Mean pool over time/tokens → subject vector
- Project to 512-D for cross-modal alignment
- Residualize: age, sex, site, mean FD, tSNR

**Fusion targets:**
- **Gene-brain associations:** Late fusion with Caduceus/DNABERT-2 embeddings
- **Behavioral prediction:** MDD, fluid intelligence, cognitive composites
- **Developmental trajectories:** Longitudinal cohorts (Cha Hospital, ABCD)

### For ARPA-H Brain-Omics Models

BrainLM serves as a **brain modality encoder** in larger multimodal systems:
- Embeddings can be projected into shared LLM/VLM spaces for cross-modal reasoning
- Site-robust features critical for federated/multi-institution Brain-Omics Models
- Natural baseline before escalating to multimodal encoders (Brain Harmony, BrainMT)

## Embedding Extraction Workflow

```bash
# 1. Preprocess fMRI → parcellate (Schaefer-400)
# 2. Load pretrained BrainLM checkpoint
# 3. Extract latent tokens (no masking during inference)
# 4. Pool to subject vector
# 5. Apply harmonization (ComBat/MURD) if needed
# 6. Log embedding strategy ID in experiment config
```

## Strengths & Limitations

### Strengths
- **Multi-site robustness:** Pretraining on UKB+HCP reduces site effects
- **Strong baselines:** High linear probe accuracy on cognitive/behavioral tasks
- **Computational efficiency:** ViT inference faster than recurrent/SSM alternatives
- **Well-documented:** Extensive benchmarks vs. classical FC approaches

### Limitations
- **Requires parcellation:** No raw 4D volume support (unlike SwiFT/BrainMT)
- **Fixed TR assumption:** Variable TR cohorts need TAPE-style adaptation
- **Embedding interpretability:** Latent space less directly tied to functional networks than FC matrices

## When to Use BrainLM

✅ **Use when:**
- Starting fMRI integration baselines (Option B in Nov 2025 plan)
- Need site-robust features across UKB/HCP/developmental cohorts
- Want efficient inference for large-N experiments

⚠️ **Consider alternatives:**
- **Brain-JEPA:** Lower latency, better semantic consistency claims
- **Brain Harmony:** Multi-modal sMRI+fMRI fusion with TAPE for TR heterogeneity
- **BrainMT:** Long-range temporal dependencies via Mamba blocks
- **SwiFT:** 4D volume input without explicit parcellation

## Reference Materials

**Primary sources:**
- **Paper:** [BrainLM (2024)](../../generated/kb_curated/papers-pdf/brainlm_2024.pdf)
- **Code walkthrough:** [BrainLM walkthrough](../../code_walkthroughs/brainlm_walkthrough.md)
- **YAML card:** `kb/model_cards/brainlm.yaml` (detailed architecture specs)
- **Paper card:** `kb/paper_cards/brainlm_2024.yaml` (structured takeaways)

**Integration recipes:**
- [Modality Features: fMRI](../../integration/modality_features/fmri.md)
- [Integration Strategy](../../integration/integration_strategy.md)
- [CCA + Permutation Recipe](../../integration/analysis_recipes/cca_permutation.md)

**Source repository:**
- **Local:** `external_repos/brainlm/`
- **GitHub:** [vandijklab/BrainLM](https://github.com/vandijklab/BrainLM)

## Next Steps in Our Pipeline

1. **Validate extraction:** Ensure consistent embeddings across UKB/Cha Hospital cohorts
2. **Benchmark stability:** Test across different parcellation schemes (Schaefer 100/200/400)
3. **Gene-brain CCA:** Align BrainLM embeddings with Caduceus gene vectors
4. **Fusion experiments:** Compare late fusion vs. two-tower contrastive alignment
5. **Developmental extension:** Adapt to pediatric fMRI (shorter scans, higher motion)

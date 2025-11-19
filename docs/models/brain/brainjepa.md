---
title: Brain-JEPA — Model Card
status: active
updated: 2025-11-19
---

# Brain-JEPA

## Overview

**Type:** Joint-Embedding Predictive Architecture for fMRI  
**Architecture:** JEPA with functional gradient positioning  
**Modality:** Functional MRI (parcel time series)  
**Primary use:** Semantic-consistent subject embeddings for zero-shot and linear probing

## Purpose & Design Philosophy

Brain-JEPA extends JEPA (Joint-Embedding Predictive Architecture) to fMRI by learning latent representations that predict masked brain regions without pixel-level reconstruction. The model emphasizes **semantic consistency** across brain states by using functional gradient positioning and spatiotemporal masking strategies (Cross-ROI, Cross-Time).

**Key innovation:** Avoids reconstruction loss collapse; achieves better linear probe performance than MAE-based approaches on reported benchmarks.

## Architecture Highlights

- **Backbone:** JEPA encoder-predictor with functional gradient positional encoding
- **Input:** Parcel time series (ROI × timepoints)
- **Pretraining:** Predict latent representations of masked regions/timeframes
- **Masking:** Cross-ROI (spatial) and Cross-Time (temporal) strategies
- **Output:** Token latents → pooled to compact subject vectors

## Integration Strategy

### For Neuro-Omics KB

**Embedding recipe:** `rsfmri_brainjepa_roi_v1`
- Extract token latents from pretrained encoder (no reconstruction decoder)
- Pool latent tokens → subject-level embedding
- Project to 512-D for downstream tasks
- Residualize: age, sex, site, mean FD

**Fusion targets:**
- **Gene-brain alignment:** Late fusion with genomic embeddings (Caduceus, Evo2)
- **Behavioral prediction:** Cognitive scores, psychiatric diagnoses
- **Zero-shot transfer:** Leverage semantic consistency for unseen tasks

### For ARPA-H Brain-Omics Models

Brain-JEPA provides **lower-latency fMRI encoding** compared to full autoencoding:
- No reconstruction decoder → faster inference for large-scale screening
- Semantic latents align well with language/vision embeddings in multimodal hubs
- Functional gradient positioning preserves anatomical relationships for cross-modal reasoning

## Embedding Extraction Workflow

```bash
# 1. Preprocess fMRI → parcellate (standard atlas)
# 2. Load pretrained Brain-JEPA encoder (not predictor/decoder)
# 3. Forward pass → extract token latents
# 4. Pool (mean/attention) → subject embedding
# 5. Optional: Apply harmonization before projection
# 6. Log embedding_strategy ID: rsfmri_brainjepa_roi_v1
```

## Strengths & Limitations

### Strengths
- **Better linear probing:** Reported improvements over MAE on cognitive/behavioral tasks
- **Lower inference cost:** No reconstruction decoder needed at embedding extraction time
- **Semantic consistency:** Latent predictions enforce functional coherence
- **Interpretability:** Functional gradient positioning maintains anatomical structure

### Limitations
- **Heavier engineering:** JEPA training more complex than standard MAE
- **Less mature ecosystem:** Fewer public checkpoints vs. BrainLM
- **Requires careful masking:** Cross-ROI/Time strategies need domain expertise
- **Limited long-context claims:** Not explicitly designed for ultra-long temporal dependencies

## When to Use Brain-JEPA

✅ **Use when:**
- Need semantic consistency for zero-shot/few-shot tasks
- Want faster inference than full autoencoding models
- Prioritize linear probe performance over reconstruction fidelity

⚠️ **Consider alternatives:**
- **BrainLM:** More mature, extensive benchmarks, simpler architecture
- **BrainMT:** For long-range temporal modeling with Mamba blocks
- **Brain Harmony:** Multi-modal sMRI+fMRI fusion
- **SwiFT:** 4D volume input without parcellation

## Reference Materials

**Primary sources:**
- **Paper:** [Brain-JEPA (2024)](../../generated/kb_curated/papers-pdf/brainjepa_2024.pdf)
- **Code walkthrough:** [Brain-JEPA walkthrough](../../code_walkthroughs/brainjepa_walkthrough.md)
- **YAML card:** `kb/model_cards/brainjepa.yaml`
- **Paper card:** `kb/paper_cards/brainjepa_2024.yaml`

**Integration recipes:**
- [Modality Features: fMRI](../../integration/modality_features/fmri.md)
- [Integration Strategy](../../integration/integration_strategy.md)
- [Design Patterns](../../integration/design_patterns.md)

**Source repository:**
- **Local:** `external_repos/brainjepa/`
- **GitHub:** [janklees/brainjepa](https://github.com/janklees/brainjepa)

## Next Steps in Our Pipeline

1. **Benchmark vs. BrainLM:** Compare linear probe performance on UKB cognitive tasks
2. **Latency profiling:** Quantify inference speedup vs. full MAE reconstruction
3. **Gene-brain fusion:** Test whether semantic latents improve CCA with genomic features
4. **Zero-shot evaluation:** Assess transfer to Cha Hospital developmental cohort
5. **Multimodal alignment:** Explore projection into shared LLM embedding space

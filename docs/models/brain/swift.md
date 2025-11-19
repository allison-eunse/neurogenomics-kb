---
title: SwiFT — Model Card
status: active
updated: 2025-11-19
---

# SwiFT

## Overview

**Type:** Spatiotemporal foundation model for fMRI  
**Architecture:** Swin Transformer (hierarchical windows)  
**Modality:** Functional MRI (4D volumes)  
**Primary use:** Direct 4D volume encoding without explicit parcellation

## Purpose & Design Philosophy

SwiFT (Swin Transformer for fMRI Time series) applies hierarchical windowed attention to 4D fMRI volumes, eliminating the need for explicit parcellation while capturing spatiotemporal patterns across multiple scales. The model processes raw BOLD signals through cascaded Swin blocks, enabling direct learning from volumetric data.

**Key innovation:** Sequence-free 4D modeling with hierarchical attention windows preserves fine-grained spatial structure while capturing temporal dynamics.

## Architecture Highlights

- **Backbone:** 4D Swin Transformer with shifted windows
- **Input:** Raw BOLD volumes (X × Y × Z × T)
- **Windowing:** Hierarchical 4D patches with local/global attention
- **No parcellation:** Learns spatial structure end-to-end
- **Output:** Subject-level embeddings via global pooling or CLS token

## Integration Strategy

### For Neuro-Omics KB

**Embedding recipe:** `rsfmri_swift_segments_v1`
- Process 4D volumes through Swin blocks (typically 20-frame segments)
- Extract final layer representations
- Pool across spatial-temporal dimensions → subject vector
- Project to 512-D for cross-modal alignment
- Residualize: age, sex, site, mean FD

**Fusion targets:**
- **Gene-brain associations:** When fine-grained spatial patterns matter
- **Atlasing-free analysis:** Avoid parcellation scheme dependence
- **Multi-resolution modeling:** Capture both local and global brain dynamics

### For ARPA-H Brain-Omics Models

SwiFT's **hierarchical 4D processing** offers advantages for Brain-Omics systems:
- No parcellation bias → better cross-site generalization
- Multi-scale attention aligns with hierarchical biological organization
- 4D paradigm extensible to other volumetric time series (perfusion imaging, DCE-MRI)
- Can serve as blueprint for spatiotemporal EEG source reconstruction

## Embedding Extraction Workflow

```bash
# 1. Preprocess fMRI → motion correction, normalization (no parcellation)
# 2. Segment into overlapping 4D windows (e.g., 20-frame chunks)
# 3. Load pretrained SwiFT checkpoint
# 4. Forward pass through Swin blocks
# 5. Extract global representation (CLS token or spatial average)
# 6. Aggregate across segments → subject embedding
# 7. Log: window_size, stride, preprocessing_pipeline_id
```

## Strengths & Limitations

### Strengths
- **No parcellation required:** Learns spatial structure end-to-end
- **Multi-scale processing:** Hierarchical windows capture local and global patterns
- **Strong performance:** Reported competitive results vs. parcellation-based methods
- **Parcellation-agnostic:** No bias from atlas choice (Schaefer vs. AAL vs. Gordon)

### Limitations
- **Computational cost:** 4D convolutions and windowed attention memory-intensive
- **Longer training:** Hierarchical architecture requires more epochs to converge
- **Preprocessing critical:** Motion and spatial normalization quality directly impact performance
- **GPU memory:** Full 4D volumes with fine temporal resolution may exceed typical GPU limits

## When to Use SwiFT

✅ **Use when:**
- Want to avoid parcellation scheme dependence
- Need fine-grained spatial analysis (subcortical structures, small nuclei)
- Have sufficient compute for 4D volume processing
- Exploring multi-resolution spatiotemporal patterns

⚠️ **Consider alternatives:**
- **BrainLM/Brain-JEPA:** If parcellation acceptable and want faster baselines
- **BrainMT:** For longer temporal contexts with lower memory footprint
- **Brain Harmony:** Multi-modal sMRI+fMRI fusion with TAPE

## Reference Materials

**Primary sources:**
- **Paper:** Pending full KB curation — see [upstream paper](https://doi.org/TBD) and repo README
- **Code walkthrough:** [SwiFT walkthrough](../../code_walkthroughs/swift_walkthrough.md)
- **YAML card:** `kb/model_cards/swift.yaml`
- **Paper card:** TBD (add to `kb/paper_cards/swift_2024.yaml` after curation)

**Integration recipes:**
- [Modality Features: fMRI](../../integration/modality_features/fmri.md)
- [Integration Strategy](../../integration/integration_strategy.md)
- [Preprocessing Pipelines](../../generated/kb_curated/integration_cards/rsfmri_preprocessing_pipelines.md)

**Source repository:**
- **Local:** `external_repos/swift/`
- **GitHub:** [Transconnectome/SwiFT](https://github.com/Transconnectome/SwiFT)

## Next Steps in Our Pipeline

1. **Parcellation comparison:** SwiFT vs. BrainLM (Schaefer-400) on same UKB cognitive tasks
2. **Memory profiling:** Document GPU requirements across different volume resolutions
3. **Preprocessing sensitivity:** Test robustness to motion correction/spatial normalization choices
4. **Gene-brain fusion:** Evaluate whether 4D embeddings improve genetics alignment
5. **Developmental adaptation:** Assess performance on pediatric datasets with smaller brain volumes

## Engineering Notes

- Segment long scans into overlapping windows to fit GPU memory
- Log **window size**, **stride**, and **overlap** for reproducibility
- Spatial normalization quality critical — consider using MURD/ComBat preprocessing
- When comparing to parcellation-based models, ensure fair preprocessing parity

---
title: TITAN — Model Card
status: active
updated: 2025-11-19
---

# TITAN (Transformer-based Image and Text Alignment Network)

## Overview

**Type:** Whole-Slide Pathology Foundation Model  
**Architecture:** Slide-level Vision Transformer with vision-language alignment  
**Modality:** Whole-slide histopathology images (WSIs) + pathology reports  
**Primary use:** Slide-level feature extraction for diagnosis, prognosis, retrieval, and report generation

## Purpose & Design Philosophy

TITAN is a slide-level foundation model for digital pathology that transforms **gigapixel whole-slide images** into general-purpose feature representations supporting diagnosis, biomarker prediction, survival analysis, rare disease retrieval, and report generation. Instead of operating on raw pixels, TITAN builds on pre-extracted patch embeddings and scales self-supervised learning to entire slides using vision transformers with long-context positional encodings. The model is pretrained in three stages: vision-only SSL, ROI-level caption alignment, and slide-level report alignment.

**Key innovation:** Multi-scale hierarchical architecture processes gigapixel pathology images end-to-end with vision-language alignment, achieving strong zero-shot and few-shot performance on rare diseases.

## Architecture Highlights

- **Three-stage pretraining:**
  - **Stage 1 (TITANV):** Vision-only SSL on WSI feature grids with iBOT-style masked prediction
  - **Stage 2:** ROI-level contrastive alignment with synthetic captions (PathChat)
  - **Stage 3:** Slide-level alignment with pathology reports via CoCa-style objectives
- **Input representation:** 2D grid of patch embeddings (from CONCH v1.5 encoder) + [CLS] token
- **Positional encoding:** Long-range encodings adapted to large 2D grids (>10⁴ patches)
- **Scale:** Pretrained on 335k WSIs across 20 organ types + 182k pathology reports
- **Tasks:** Subtyping, biomarker prediction, survival, retrieval, zero-shot classification, report generation

## Integration Strategy

### For Neuro-Omics KB

TITAN provides **hierarchical vision-language patterns**:

**Key lessons for brain imaging:**
- **Multi-scale processing:** Hierarchical approach applicable to multi-resolution brain imaging (T1, T2, fMRI)
- **Patch-to-whole aggregation:** TITAN's patch → slide pipeline informs voxel → brain → subject aggregation
- **Vision-language alignment:** Contrastive learning patterns transferable to brain scans + radiology reports
- **Zero-shot rare disease:** Critical for uncommon neurological phenotypes with <100 cases

**Potential adaptation for neuroimaging:**
```
Brain MRI voxels → Patch embeddings (BrainLM, SwiFT)
                → 3D grid of features
                → Vision transformer with long-context encoding
                → Contrastive alignment with radiology reports
                → Zero-shot diagnosis + report generation
```

### For ARPA-H Brain-Omics Model (BOM)

TITAN demonstrates **whole-system feature extraction**:

```
Gene variants → Regional embeddings
               ↓
Brain volumes → Multi-scale features (voxel → region → whole-brain)
               ↓
               Vision-language alignment
               ↓
Clinical predictions + report generation
```

**Transfer insights:**
- **Long-context modeling:** Process entire brain volumes without cropping/downsampling
- **Rare phenotype retrieval:** TITAN's retrieval success informs rare genetic disorder diagnosis
- **Few-shot learning:** Strong performance with minimal labels—critical for rare neurological conditions
- **Synthetic caption generation:** PathChat patterns applicable to brain ROI descriptions

## Embedding Extraction Workflow

If adapting TITAN for neuroimaging:

```bash
# 1. Extract patch-level features from brain scans
#    - Use BrainLM or SwiFT as patch encoder (analogous to CONCH)
# 2. Arrange patches into 3D grid preserving spatial layout
# 3. Apply vision transformer with long-context positional encoding
# 4. Stage 1: Self-supervised pretraining on brain volumes
# 5. Stage 2: ROI-level alignment with synthetic captions
# 6. Stage 3: Whole-scan alignment with radiology reports
# 7. Extract embeddings for downstream tasks
```

**For neuro-omics KB:**
- **Hierarchical features:** Multi-scale brain representations
- **Report alignment:** Connect brain scans with clinical text
- **Zero-shot transfer:** Apply to new cohorts without labeled data

## Strengths & Limitations

### Strengths
- **Gigapixel-scale processing:** Handles entire WSIs (>10⁴ patches) end-to-end
- **Vision-language alignment:** Supports zero-shot classification and report generation
- **Strong few-shot performance:** Excels with limited labeled data
- **Rare disease retrieval:** Validated on diagnostically challenging cases
- **Multi-scale pretraining:** Vision-only + ROI-level + slide-level stages

### Limitations
- **Pathology-specific:** Trained on histopathology, not neuroimaging
- **Requires powerful patch encoder:** Depends on CONCH v1.5 quality
- **Compute intensive:** Large-scale WSI pretraining expensive
- **Limited to 2D spatial context:** Does not natively handle 3D/4D neuroimaging sequences

## When to Use TITAN

✅ **Use as reference when:**
- Designing hierarchical vision models for brain imaging
- Building vision-language alignment for medical imaging + reports
- Implementing zero-shot rare disease classification
- Scaling models to gigapixel/high-resolution inputs

⚠️ **Do not use directly for:**
- Neuroimaging (trained on pathology, not brain scans)
- 3D/4D temporal sequences (designed for 2D spatial grids)
- Production diagnosis (requires clinical validation)

⚠️ **Consider alternatives:**
- **BrainLM/SwiFT:** For neuroimaging-specific feature extraction
- **M3FM:** For CLIP-style alignment with medical reports
- **BAGEL:** For unified understanding + generation across modalities

## Reference Materials

**Primary sources:**
- **Paper:** [TITAN (2025)](../../generated/kb_curated/papers-md/titan_2025.md) — Nature Medicine
- **Code walkthrough:** [TITAN walkthrough](../../code_walkthroughs/titan_walkthrough.md)
- **YAML card:** [kb/model_cards/titan.yaml](https://github.com/allison-eunse/neuro-omics-kb/blob/main/kb/model_cards/titan.yaml)

**Integration recipes:**
- [Multimodal Architectures](../../integration/multimodal_architectures.md)
- [Design Patterns](../../integration/design_patterns.md) — Hierarchical vision-language
- [Integration Strategy](../../integration/integration_strategy.md)

**Source repository:**
- **Local:** `external_repos/titan/`
- **GitHub:** [mahmoodlab/TITAN](https://github.com/mahmoodlab/TITAN)

## Next Steps in Our Pipeline

1. **Hierarchical architecture study:** Extract multi-scale patterns for brain imaging
2. **Vision-language adaptation:** Implement brain scan + report contrastive learning
3. **Zero-shot rare phenotypes:** Evaluate on uncommon neurological disorders
4. **3D/4D extension:** Adapt long-context encoding to temporal fMRI sequences
5. **Few-shot learning:** Test with limited labels on Cha Hospital pediatric cohorts

## Engineering Notes

- TITAN's **three-stage pretraining** (vision → ROI captions → reports) provides a clear template
- **Long-context positional encodings** critical for processing entire brain volumes
- **PathChat synthetic captions** demonstrate value of synthetic data for vision-language alignment
- **Rare disease retrieval** evaluation pattern applicable to rare genetic neurological disorders


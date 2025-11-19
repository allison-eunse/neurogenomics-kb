---
title: BrainMT — Model Card
status: active
updated: 2025-11-19
---

# BrainMT

## Overview

**Type:** Hybrid State Space + Transformer for fMRI  
**Architecture:** Mamba blocks + Multi-Head Self-Attention (Hybrid SSM-Transformer)  
**Modality:** Functional MRI (3D volumes or parcels)  
**Primary use:** Long-range temporal dependency modeling with computational efficiency

## Purpose & Design Philosophy

BrainMT fuses **bidirectional Mamba blocks** (State Space Models with temporal-first scanning) with **Transformer attention** to model long-range fMRI dependencies more efficiently than pure transformers. The architecture targets multitask learning across fluid intelligence regression, sex classification, and harmonization tasks on UKB/HCP cohorts.

**Key innovation:** Mamba's sub-quadratic complexity enables processing longer temporal sequences (≥200 frames) without the memory explosion of full attention.

## Architecture Highlights

- **Hybrid blocks:** Bidirectional Mamba (temporal scanning) + MHSA (global attention)
- **Patch embedding:** 3D Conv → flatten → linear projection
- **Temporal modeling:** Mamba handles sequence dependencies; attention captures global structure
- **Multitask heads:** Shared encoder → task-specific prediction heads
- **Training:** Requires fused CUDA kernels (Mamba-ssm library)

## Integration Strategy

### For Neuro-Omics KB

**Embedding recipe:** `rsfmri_brainmt_segments_v1`
- Extract embeddings from shared encoder (before task heads)
- Mean pool over sequence length → subject vector
- Project to 512-D for downstream fusion
- Residualize: age, sex, site, mean FD
- **Metadata requirement:** Log sequence length (BrainMT performance depends on context ≥200)

**Fusion targets:**
- **Long-context gene-brain alignment:** When temporal dynamics critical (e.g., task fMRI)
- **Developmental trajectories:** Pediatric longitudinal fMRI with evolving patterns
- **Multitask prediction:** Joint cognitive + diagnostic tasks

### For ARPA-H Brain-Omics Models

BrainMT demonstrates **efficient long-context modeling** for multimodal systems:
- Mamba architecture adaptable to other sequential modalities (EEG, longitudinal assessments)
- Hybrid SSM-Transformer paradigm balances efficiency vs. expressiveness
- Multitask framework aligns with Brain-Omics Model (BOM) joint training over diverse phenotypes

## Embedding Extraction Workflow

```bash
# 1. Preprocess fMRI → 3D volumes or parcels (≥200 frames preferred)
# 2. Load pretrained BrainMT checkpoint
# 3. Forward through encoder (Mamba blocks + MHSA layers)
# 4. Extract pre-head embeddings (not task-specific outputs)
# 5. Pool to subject-level vector
# 6. Log: sequence_length, mamba_config, embedding_strategy_id
```

## Strengths & Limitations

### Strengths
- **Efficient long-context:** Mamba scales sub-quadratically vs. full attention
- **Multitask learning:** Single encoder serves multiple downstream tasks
- **Hybrid architecture:** Balances local temporal patterns (Mamba) + global structure (attention)
- **Benchmarked on UKB/HCP:** Published results on fluid intelligence and sex classification

### Limitations
- **Heavy dependencies:** Requires Mamba-ssm CUDA kernels (custom build)
- **Training complexity:** Hybrid architecture harder to debug than pure ViT
- **Checkpoint availability:** Fewer public pretrained weights vs. BrainLM
- **Overkill for short sequences:** <200 frames may not fully leverage Mamba's strengths

## When to Use BrainMT

✅ **Use when:**
- Need long-context modeling (task fMRI, naturalistic viewing)
- Multitask setting with shared encoder across cognitive/diagnostic tasks
- Want efficiency gains over pure Transformer for ≥200 frame sequences
- Exploring SSM architectures for neuro-omics applications

⚠️ **Defer until:**
- BrainLM/Brain-JEPA baselines exhausted (per Nov 2025 integration plan)
- Engineering resources available for custom kernel setup
- Sufficient GPU memory for hybrid block training/inference

⚠️ **Consider alternatives:**
- **BrainLM:** Simpler baseline, more mature ecosystem
- **Brain-JEPA:** Faster inference, better for semantic consistency
- **SwiFT:** 4D volumes without explicit sequence modeling
- **Brain Harmony:** Multi-modal sMRI+fMRI fusion

## Reference Materials

### Knowledge Base Resources

**Curated materials in this KB:**
- **Paper summary & notes (PDF):** [BrainMT (2025)](../../generated/kb_curated/papers-pdf/brainmt_2025.pdf)
- **Code walkthrough:** [BrainMT walkthrough](../../code_walkthroughs/brainmt_walkthrough.md)
- **Model card (YAML):** `kb/model_cards/brainmt.yaml`
- **Paper card (YAML):** `kb/paper_cards/brainmt_2025.yaml`

**Integration recipes:**
- [Modality Features: fMRI](../../integration/modality_features/fmri.md)
- [Integration Strategy](../../integration/integration_strategy.md)
- [Design Patterns](../../integration/design_patterns.md)

### Original Sources

**Source code repositories:**
- **Local copy:** `external_repos/brainmt/`
- **Official GitHub:** [arunkumar-kannan/brainmt-fmri](https://github.com/arunkumar-kannan/brainmt-fmri)

**Original paper:**
- **Title:** "BrainMT: A Hybrid Mamba-Transformer Architecture for Modeling Long-Range Dependencies in Functional MRI Data"
- **Authors:** Kannan, Arunkumar; Lindquist, Martin A.; Caffo, Brian
- **Published:** Conference proceedings (SpringerLink), September 2025, pp. 150-160
- **Link:** [SpringerLink](https://dl.acm.org/doi/10.1007/978-3-032-05162-2_15)
- **PDF (local):** [brainmt_2025.pdf](../../generated/kb_curated/papers-pdf/brainmt_2025.pdf)

## Next Steps in Our Pipeline

1. **Baseline comparisons:** BrainMT vs. BrainLM on UKB cognitive tasks (same train/test splits)
2. **Sequence length ablation:** Test performance vs. context length (100, 200, 400 frames)
3. **Gene-brain alignment:** Evaluate whether long-context embeddings improve genetics CCA
4. **Developmental extension:** Adapt to pediatric fMRI (Cha Hospital, ABCD)
5. **SSM exploration:** Investigate Mamba-style architectures for EEG/EPhys modalities

## Engineering Notes

- Capture **masking ratio** and **sequence length** in metadata for reproducibility
- Multitask heads are task-specific; extract **shared encoder embeddings** for fusion
- When exporting weights, ensure Mamba kernel version compatibility across systems

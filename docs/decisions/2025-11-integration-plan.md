---
title: Integration Baseline Plan (Nov 2025)
status: active
updated: 2025-11-19
---

# Integration Baseline Plan

This document defines the **phased escalation strategy** for gene-brain-behavior integration, informed by foundation model research, oncology multimodal reviews, and ARPA-H Brain-Omics Model (BOM) vision.

## 🔵 Phase 1: Late Fusion Baselines (Current)

**Principle**: Prefer late integration first under heterogeneous semantics.

- **Sources**: Ensemble Integration (Li et al. 2022), Oncology multimodal review (2024)
- **Inference**: Preserve modality-specific signal; avoid premature joint spaces
- **Implementation**:
  - Concatenate compact per-modality features (genetics embeddings + sMRI PCA + fMRI FC)
  - Train LR and GBDT baselines with stratified CV
  - Z-score + residualize per feature vs covariates
  - AUROC/AUPRC with CIs; DeLong/bootstrap for differences

**Analysis Recipes**:

- **CCA + permutation**: Test gene-brain associations before heavy fusion
  - 1,000 permutations on residualized, standardized inputs
  - Record canonical correlations, significance, and stability
- **Prediction baselines**: Gene-only vs Brain-only vs Late fusion
  - Establish unimodal baselines before multimodal claims
- **Partial correlations**: Logistic regression with covariates
  - Control for age, sex, site, scanner before interpreting effects

**Modality Sequencing**:

- Start with sMRI ROIs (FreeSurfer PCA embeddings)
- Add fMRI as FC vectors (BrainLM/SwiFT embeddings)
- Later consider brain FM latents (BrainMT, Brain Harmony)

**Genetics Embedding Hygiene**:

- **RC-equivariance**: Average forward + reverse-complement embeddings (Caduceus)
- **Deterministic tokenization**: Use consistent k-mer/BPE strategies (DNABERT-2, GENERator)
- **Gene attribution**: LOGO (leave-one-gene-out) ΔAUC with Wilcoxon + FDR (Yoon BioKDD'25)

## 🟢 Phase 2: Two-Tower Contrastive (Near-term)

**Trigger**: Late fusion shows consistent gene-brain signal (CCA p<0.001, prediction improvement >5% AUROC)

**Architecture Pattern**:

- **Frozen encoders**: Genetics FM (Caduceus/DNABERT-2) + Brain FM (BrainLM/SwiFT)
- **Small projectors**: 256→128→64 for each modality
- **Contrastive loss**: InfoNCE with temperature tuning
- **Reference models**: M3FM (vision-language for medical imaging), oncology two-tower review

**What to Monitor**:

- Embedding alignment quality (cosine similarity distributions)
- Downstream task performance vs. late fusion
- Computational cost (FLOPs, GPU memory)

**When to Escalate to Phase 3**:

- Two-tower alignment consistently outperforms late fusion by >10% AUROC
- Need for cross-modal reasoning (e.g., "which genes explain this brain pattern?")
- Multi-site/TR heterogeneity requires joint harmonization

## 🔴 Phase 3: Unified Multimodal Architectures (Long-term)

**Vision**: ARPA-H-style Brain-Omics Model (BOM) — unified transformer processing gene-brain-behavior-language tokens

**Architecture Options**:

### Option A: Mixture-of-Transformers (MoT)

- **Pattern**: Modality-specific FFNs + shared global attention
- **Advantages**: 
  - 55% FLOPs of dense baseline
  - Modality-aware sparsity (genetics_ffn, brain_ffn, behavior_ffn)
  - Stable scaling to 7B+ parameters
- **Reference**: MoT paper (2025), BAGEL paper (2025)
- **Use case**: Large cohorts (UK Biobank N=500k+), compute-constrained environments

### Option B: Unified Decoder (BAGEL-style)

- **Pattern**: Single decoder-only transformer with interleaved modality tokens
- **Advantages**:
  - Emergent cross-modal reasoning
  - Supports both understanding (gene-brain association) and generation (clinical report)
  - Mixture-of-experts for task-specific specialization
- **Reference**: BAGEL paper (2025)
- **Use case**: Gene-brain-behavior-language unification with LLM as semantic hub

### Option C: LLM-as-Bridge

- **Pattern**: Project genetics/brain embeddings into LLM token space (Me-LLaMA-style)
- **Advantages**:
  - Natural language queries over multimodal neuro-omics data
  - Leverage pretrained medical LLMs for domain knowledge
  - Explain genetic risk in natural language
- **Reference**: Me-LLaMA (2024), M3FM (2025)
- **Use case**: Clinical decision support, patient-facing genetic counseling

## Decision Table

| Signal Strength | Computational Budget | Primary Goal | Recommended Pattern |
|----------------|---------------------|-------------|-------------------|
| Weak (CCA p>0.01) | Any | Establish baseline | Late Fusion |
| Moderate (CCA p<0.001, ΔAUROC<5%) | Low | Gene-brain association | Late Fusion + CCA |
| Strong (ΔAUROC>5%) | Medium | Multimodal prediction | Two-Tower Contrastive |
| Very Strong (ΔAUROC>10%) | High | Cross-modal reasoning | MoT or Unified Decoder |
| Strong + Language | High | Clinical integration | LLM-as-Bridge |

## Current Implementation Status

✅ **Phase 1 Complete**:
- Analysis recipes documented (CCA, prediction, partial correlations)
- Modality features specified (genomics, sMRI, fMRI)
- Embedding policies defined (naming, PCA dims)
- Integration cards: Ensemble integration, Oncology review

🚧 **Phase 2 Prep**:
- Integration card: Multimodal FM patterns (synthesizes BAGEL, MoT, M3FM, Me-LLaMA, TITAN)
- Multimodal architectures doc (detailed BAGEL/MoT/M3FM/Me-LLaMA/TITAN patterns)
- Design patterns doc (late fusion, two-tower, MoT, BOM escalation logic)

⏳ **Phase 3 Future**:
- Awaiting Phase 1 validation on UK Biobank + genetics embeddings
- Will pilot two-tower if CCA+permutation shows p<0.001 canonical correlations
- BOM architecture selection depends on computational resources and cohort size

## Key References

- **Ensemble Integration**: [Li et al. 2022](../generated/kb_curated/papers-md/ensemble_integration_li2022.md)
- **Oncology Multimodal Review**: [Waqas et al. 2024](../generated/kb_curated/papers-md/oncology_multimodal_waqas2024.md)
- **Multimodal FM Survey**: [Gupta et al. 2025](../generated/kb_curated/papers-md/mmfm_2025.md)
- **BAGEL**: [arXiv:2505.14683](../generated/kb_curated/papers-md/bagel_2025.md)
- **MoT**: [arXiv:2411.04996](../generated/kb_curated/papers-md/mot_2025.md)
- **M3FM**: [ai-in-health/M3FM](../generated/kb_curated/papers-md/m3fm_2025.md)
- **Me-LLaMA**: [BIDS-Xu-Lab/Me-LLaMA](../generated/kb_curated/papers-md/me_llama_2024.md)

## Next Steps

1. **Validate Phase 1** on UK Biobank WES + sMRI/fMRI
2. **Monitor trigger conditions** for Phase 2 escalation
3. **Pilot two-tower** if late fusion shows >5% AUROC improvement
4. **Document decisions** in this log as we progress through phases

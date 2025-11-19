---
title: Multimodal Architecture Patterns
status: active
updated: 2025-11-19
---

# Multimodal Architecture Patterns for Brain-Omics Models

This document catalogs **architectural patterns from multimodal foundation models** that inform the design of ARPA-H-style Brain-Omics Model (BOM) systems. These models demonstrate how to fuse heterogeneous modalities (vision, language, time series, structured data) at scale—lessons directly applicable to gene–brain–behavior–language integration.

## Overview

**Purpose:** Extract design principles from state-of-the-art multimodal FMs to guide Neuro-Omics KB integration strategies as they escalate from late fusion → two-tower contrastive → unified multimodal architectures.

**Scope:** Medical/clinical multimodal FMs, unified vision-language-speech models, and sparse multimodal transformers.

---

## 1. BAGEL — Unified Multimodal Foundation Model

### Architecture Summary

**Model:** BAGEL (Emerging Properties in Unified Multimodal Pretraining)  
**Paper:** [arXiv:2505.14683](https://arxiv.org/abs/2505.14683) | **Card:** `kb/paper_cards/bagel_2025.yaml`

- **Backbone:** Qwen2.5 decoder-only transformer (7B active, 14B total with MoT experts)
- **Modalities:** Text, images, video, web data
- **Architecture:** Mixture-of-Transformer-Experts (MoT) with separate experts for understanding vs. generation
- **Visual encoding:** SigLIP2-style ViT encoder for understanding
- **Visual generation:** FLUX VAE + rectified-flow diffusion conditioned on transformer states
- **Training:** Trillions of interleaved multimodal tokens with reasoning-oriented curation

### Key Design Patterns

✅ **Unified decoder-only architecture:** Single transformer processes all modalities as token sequences  
✅ **Mixture-of-experts (MoT):** Separate experts for understanding (comprehension) vs. generation tasks  
✅ **Interleaved data:** Reasoning-oriented multimodal corpus with natural task diversity  
✅ **Emergent capabilities:** Complex reasoning, free-form manipulation, 3D understanding from unified pretraining

### Implications for Brain-Omics Models

**Direct applications:**
- **Gene-brain-language unification:** Treat genetics (nucleotide tokens), brain (parcel tokens), and behavior (structured tokens) as additional modalities alongside text
- **MoT for neuro-omics:** Separate experts for discriminative (gene-brain association) vs. generative (report generation, counterfactual prediction) tasks
- **Interleaved corpus design:** Create multimodal corpus pairing genetic variants + brain scans + cognitive assessments + clinical narratives

**Escalation path:**
1. Late fusion baselines (current)
2. Two-tower contrastive (gene encoder ↔ brain encoder)
3. MoT-style unified architecture where genetics/brain/behavior tokens share decoder with modality-specific experts

**Reference materials:**
- [BAGEL walkthrough](../code_walkthroughs/bagel_walkthrough.md)
- [BAGEL paper card](../generated/kb_curated/papers-md/bagel_2025.md)

---

## 2. MoT — Mixture-of-Transformers

### Architecture Summary

**Model:** Mixture-of-Transformers (Sparse and Scalable for Multi-Modal FMs)  
**Paper:** [arXiv:2411.04996](https://arxiv.org/abs/2411.04996) | **Card:** `kb/paper_cards/mot_2025.yaml`

- **Backbone:** Sparse multimodal transformer with modality-aware FFNs/attention
- **Modalities:** Text, images, speech
- **Sparsity mechanism:** Separate FFN/attention projections per modality; shared global self-attention
- **Settings:** Chameleon-style autoregressive + Transfusion-style diffusion
- **Efficiency:** ~55.8% FLOPs of dense baseline, similar or better performance

### Key Design Patterns

✅ **Modality-aware sparsity:** Decouple non-embedding parameters by modality  
✅ **Shared global attention:** All tokens interact via self-attention (no routing)  
✅ **Drop-in replacement:** Compatible with existing dense transformer architectures  
✅ **Stable scaling:** Maintains performance across model sizes (1B → 7B → 30B)

### Implications for Brain-Omics Models

**Direct applications:**
- **Per-modality FFNs:** Separate feed-forward networks for genetics, brain MRI, fMRI, EEG, behavior tokens
- **Shared attention:** Global self-attention over all modalities captures cross-modal dependencies
- **Compute efficiency:** Critical for scaling to large cohorts (UK Biobank N=500k+)

**Integration with Neuro-Omics KB:**
- Implement modality-specific projectors (genetics_ffn, brain_ffn, behavior_ffn)
- Retain shared attention over concatenated gene+brain+behavior tokens
- Compare vs. learned MoE routing (simpler, more interpretable)

**Reference materials:**
- [MoT walkthrough](../code_walkthroughs/mot_walkthrough.md)
- [MoT paper card](../generated/kb_curated/papers-md/mot_2025.md)

---

## 3. M3FM — Multilingual Medical Model

### Architecture Summary

**Model:** M3FM (Multilingual Chest X-ray Report Generator)  
**Repo:** [ai-in-health/M3FM](https://github.com/ai-in-health/M3FM) | **Card:** `kb/model_cards/m3fm.yaml`

- **Backbone:** Multilingual CLIP encoder + relational-memory Transformer decoder
- **Modalities:** Chest X-ray images, bilingual text (English/Chinese)
- **Architecture:** Two-tower (vision encoder + language decoder) with relational memory
- **Decoder:** Language selection via BOS token (1=English, 2=Chinese)
- **Training:** COV-CTR COVID-era CXR dataset with multilingual reports

### Key Design Patterns

✅ **Two-tower fusion:** Vision encoder outputs → cross-attention in language decoder  
✅ **Language-aware generation:** Single decoder handles multiple languages via BOS conditioning  
✅ **Relational memory:** Augmented attention for capturing long-range report dependencies  
✅ **Medical domain adaptation:** CLIP text embeddings projected for medical terminology

### Implications for Brain-Omics Models

**Direct applications:**
- **Brain-omics-to-language:** Project brain/genetics embeddings into CLIP-like space → generate clinical narratives
- **Bilingual reporting:** Extend to English/Korean for Cha Hospital developmental cohorts
- **Relational memory for clinical context:** Track longitudinal patient history across visits

**Integration strategy:**
- Use M3FM-style two-tower for brain scan → clinical report generation
- Adapt relational memory for multi-visit longitudinal modeling
- Explore gene embedding → language generation (explain genetic risk in natural language)

**Reference materials:**
- [M3FM walkthrough](../code_walkthroughs/m3fm_walkthrough.md)
- [M3FM model card](../model_cards/m3fm.yaml)

---

## 4. Me-LLaMA — Medical LLM

### Architecture Summary

**Model:** Me-LLaMA (Medical LLaMA)  
**Repo:** [BIDS-Xu-Lab/Me-LLaMA](https://github.com/BIDS-Xu-Lab/Me-LLaMA) | **Card:** `kb/model_cards/me_llama.yaml`

- **Backbone:** LLaMA-2/3 (13B/70B) with continual pretraining + LoRA instruction tuning
- **Modality:** Medical text (biomedical literature, clinical notes, guidelines)
- **Pretraining ratio:** 15:1:4 (biomedical : clinical : general)
- **Training:** 129B medical tokens + 214K instruction samples
- **Evaluation:** 12+ medical QA/NLP tasks with prompt templates

### Key Design Patterns

✅ **Continual pretraining:** Adapt general LLM to medical domain with curated corpus  
✅ **LoRA instruction tuning:** Parameter-efficient adaptation for clinical reasoning  
✅ **Prompt engineering:** Modality-specific prompts for different clinical tasks  
✅ **Evaluation harness:** Structured benchmarking across medical NLP tasks

### Implications for Brain-Omics Models

**Direct applications:**
- **Neuro-omics LLM:** Continual pretrain LLaMA on neuroscience literature + genetics papers + clinical neurology notes
- **Instruction tuning for clinical tasks:** Adapt for cognitive assessment interpretation, genetic counseling, neuroimaging report generation
- **Prompt templates:** Create standardized prompts for gene-brain-behavior reasoning

**As semantic bridge in BOM:**
- Me-LLaMA-style medical LLM serves as **semantic hub** for Brain-Omics Model
- Project genetics/brain/EEG embeddings into LLM token space for cross-modal reasoning
- Enable natural language queries over multimodal neuro-omics data

**Reference materials:**
- [Me-LLaMA walkthrough](../code_walkthroughs/melamma_walkthrough.md)
- [Me-LLaMA model card](../model_cards/me_llama.yaml)

---

## 5. TITAN — Whole-Slide Image FM

### Architecture Summary

**Model:** TITAN (Transformer for Integrative Tissue Analysis)  
**Repo:** [mahmoodlab/TITAN](https://github.com/mahmoodlab/TITAN) | **Card:** `kb/model_cards/titan.yaml`

- **Backbone:** Slide-level transformer with multi-scale patch aggregation
- **Modality:** Whole-slide histopathology images
- **Architecture:** Hierarchical attention over gigapixel images (millions of patches)
- **Applications:** Cancer diagnosis, survival prediction, treatment response

### Key Design Patterns

✅ **Multi-scale patch processing:** Handle gigapixel images via hierarchical aggregation  
✅ **Attention-based pooling:** Learn to aggregate informative regions  
✅ **Slide-level embeddings:** Compress millions of patches → fixed-size vectors  
✅ **Task-specific heads:** Shared encoder for multiple downstream tasks

### Implications for Brain-Omics Models

**Direct applications:**
- **Brain MRI analogy:** Whole-brain 3D volumes → hierarchical patch aggregation (similar to TITAN's slide processing)
- **Multi-scale fusion:** Combine region-level (parcels) and voxel-level (fine-grained) brain features
- **Histology + genetics:** If histopathology data available (e.g., brain tissue banks), TITAN-style processing + genetics fusion

**Integration with Neuro-Omics KB:**
- Adapt TITAN's multi-scale attention for 3D MRI volumes
- Use TITAN-style patch aggregation for whole-brain sMRI + fMRI fusion
- Explore cross-modal attention: pathology patches ↔ genetic variants

**Reference materials:**
- [TITAN walkthrough](../code_walkthroughs/titan_walkthrough.md)
- [TITAN model card](../model_cards/titan.yaml)

---

## 6. FMS-Medical Catalog

### Resource Summary

**Catalog:** Awesome Foundation Models for Advancing Healthcare  
**Repo:** [YutingHe-list/Awesome-Foundation-Models](https://github.com/YutingHe-list/Awesome-Foundation-Models-for-Advancing-Healthcare)

- **Scope:** 200+ medical foundation models across modalities (text, vision, multimodal, protein, genomics, clinical time series)
- **Organization:** Bilingual (English/Chinese) with taxonomy by modality and task
- **Usage:** Reference catalog for discovering relevant medical FMs

### Key Resources

✅ **Medical vision FMs:** CXR, CT, MRI, histopathology encoders  
✅ **Medical LLMs:** Clinical text understanding and generation models  
✅ **Genomics/proteomics FMs:** Sequence models for molecular biology  
✅ **Multimodal FMs:** Vision-language models for radiology, pathology reports

### Implications for Brain-Omics Models

**Discovery and benchmarking:**
- Identify relevant medical imaging FMs for brain scan processing
- Find medical LLMs for clinical narrative generation
- Discover multimodal architectures to adapt for gene-brain-behavior fusion

**Reference for ARPA-H integration:**
- Survey multimodal medical FMs to inform BOM architecture choices
- Benchmark against medical FM baselines (e.g., CXR report generation → adapt for neuroimaging)

**Reference materials:**
- [FMS-Medical walkthrough](../code_walkthroughs/fms_medical_walkthrough.md)
- [FMS-Medical catalog YAML](../datasets/fms_medical_catalog.yaml)

---

## Integration Roadmap for Neuro-Omics KB

### Phase 1: Late Fusion Baselines (Current)
- **Models:** Separate encoders (Caduceus, BrainLM, FreeSurfer ROIs)
- **Fusion:** Concatenate embeddings → LR/GBDT prediction
- **Evaluation:** CCA + permutation, AUROC/AUPRC, DeLong tests

### Phase 2: Two-Tower Contrastive
- **Architecture:** Frozen gene encoder ↔ frozen brain encoder with learnable projectors
- **Loss:** InfoNCE or similar contrastive objective
- **Inspiration:** CLIP-style alignment (M3FM two-tower paradigm)

### Phase 3: MoT-Style Sparse Integration
- **Architecture:** Shared self-attention over gene+brain+behavior tokens
- **Sparsity:** Modality-specific FFNs (genetics_ffn, brain_ffn, behavior_ffn)
- **Inspiration:** MoT paper (arXiv:2411.04996)

### Phase 4: Unified Brain-Omics Model (BOM)
- **Architecture:** BAGEL-style decoder-only with MoT experts
- **Modalities:** Genetics (nucleotide tokens) + brain (parcel/voxel tokens) + behavior (structured tokens) + language (text tokens)
- **Semantic bridge:** Me-LLaMA-style medical LLM as central hub
- **Training:** Interleaved multimodal corpus (genetic variants + brain scans + cognitive assessments + clinical notes)

---

## Next Steps

1. **Complete Phase 1 baselines** (CCA + prediction on UKB gene-brain data)
2. **Pilot two-tower contrastive** (gene-brain alignment with frozen encoders)
3. **Explore MoT-style sparsity** (modality-specific FFNs vs. full early fusion)
4. **Design ARPA-H BOM architecture** (unified multimodal transformer with neuro-omics tokens)
5. **Curate interleaved corpus** (multimodal neuro-omics data for unified pretraining)

---

## Reference Index

**Walkthrough documents:**
- [BAGEL walkthrough](../code_walkthroughs/bagel_walkthrough.md)
- [MoT walkthrough](../code_walkthroughs/mot_walkthrough.md)
- [M3FM walkthrough](../code_walkthroughs/m3fm_walkthrough.md)
- [Me-LLaMA walkthrough](../code_walkthroughs/melamma_walkthrough.md)
- [TITAN walkthrough](../code_walkthroughs/titan_walkthrough.md)
- [FMS-Medical walkthrough](../code_walkthroughs/fms_medical_walkthrough.md)

**Paper/model cards:**
- `kb/paper_cards/bagel_2025.yaml`
- `kb/paper_cards/mot_2025.yaml`
- `kb/model_cards/m3fm.yaml`
- `kb/model_cards/me_llama.yaml`
- `kb/model_cards/titan.yaml`
- `kb/datasets/fms_medical_catalog.yaml`

**Integration recipes:**
- [Integration Strategy](integration_strategy.md)
- [Design Patterns](design_patterns.md)
- [CCA + Permutation](analysis_recipes/cca_permutation.md)


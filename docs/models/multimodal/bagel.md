---
title: BAGEL — Model Card
status: active
updated: 2025-11-19
---

# BAGEL

## Overview

**Type:** Unified Multimodal Foundation Model  
**Architecture:** Qwen2 MoT decoder + SigLIP-NaViT encoder + FLUX VAE  
**Modality:** Text, images, video, web content (interleaved sequences)  
**Primary use:** Unified multimodal understanding and generation with emergent reasoning capabilities

## Purpose & Design Philosophy

BAGEL (Bottleneck-free Architecture for Generation and Education-rich Learning) is an open-source unified multimodal foundation model that performs both **understanding and generation** across text, images, and video within a single architecture. Built on a Qwen2.5-derived decoder with Mixture-of-Transformers (MoT) experts—one for understanding, one for generation—BAGEL processes trillions of interleaved tokens to achieve emergent capabilities like free-form visual manipulation, 3D understanding, and world navigation.

**Key innovation:** Unlike models with separate understanding/generation modules, BAGEL uses shared self-attention across a unified token sequence, allowing tight coupling between reasoning and generation without architectural bottlenecks.

## Architecture Highlights

- **MoT structure:** Two experts (understanding + generation) share the same token sequence via common self-attention
- **Visual encoders:** 
  - SigLIP2-so400m/14 with NaViT for understanding (native aspect ratios)
  - FLUX VAE for generation (latent tokens, 8× downsampled, 16 channels)
- **Backbone:** Qwen2.5 decoder with RMSNorm, SwiGLU, RoPE, GQA, QK-Norm
- **Training objective:** Next-token prediction for text; rectified flow diffusion for visual tokens
- **Scale:** 7B active parameters, 14B total; trained on trillions of interleaved multimodal tokens

## Integration Strategy

### For Neuro-Omics KB

BAGEL provides architectural templates for multimodal integration:

**Key lessons for gene-brain-behavior fusion:**
- **Unified sequences:** How to process heterogeneous modalities (genes, brain scans, behavior) in one forward pass
- **Expert specialization:** MoT pattern adaptable to "genomics expert" + "brain expert" + shared attention
- **Interleaved data:** Training on mixed sequences improves cross-modal reasoning

**Not directly used in KB pipeline** (no neuroscience pretraining), but informs:
- Design patterns for late-stage multimodal fusion (see [Design Patterns](../../integration/design_patterns.md))
- LLM-as-semantic-bridge architectures for ARPA-H BOM
- Evaluation strategies for emergent multimodal capabilities

### For ARPA-H Brain-Omics Model (BOM)

BAGEL demonstrates how to build **bottleneck-free unified models**:

```
Gene embeddings    → |
                     | Shared self-attention over unified sequence
Brain embeddings   → |     ↓
                     | Expert routing (MoT-style)
Clinical text      → |     ↓
                     | Understanding + generation heads
Behavioral data    → |
```

**Transfer insights:**
- **Emergent reasoning:** BAGEL shows that scaling interleaved data produces complex reasoning—applicable to gene-brain-behavior associations
- **CFG for generation:** Classifier-free guidance patterns transferable to conditional brain image synthesis
- **Long-context modeling:** NaiveCache streaming inference applicable to longitudinal neuroimaging sequences

## Embedding Extraction Workflow

BAGEL is not used for embedding extraction in the Neuro-Omics KB (domain mismatch), but if adapting for clinical imaging + reports:

```bash
# 1. Prepare interleaved sequences (image patches + text tokens)
# 2. Load pretrained BAGEL checkpoint
# 3. Forward through shared self-attention + MoT experts
# 4. Extract pre-head embeddings (not task-specific outputs)
# 5. Pool to subject-level vectors for downstream fusion
```

**For clinical extension:** See [M3FM](m3fm.md) for medical imaging integration patterns.

## Strengths & Limitations

### Strengths
- **Unified architecture:** Single model for understanding + generation without bottlenecks
- **Emergent capabilities:** Free-form manipulation, multiview synthesis, world navigation
- **Open-source:** Full code, checkpoints, and quantized inference (NF4, INT8)
- **Scalable:** FSDP training, packed sequences, MFU telemetry for large-scale runs

### Limitations
- **Compute intensive:** Training requires substantial resources (trillions of tokens)
- **General domain:** Not specialized for neuroscience or genomics
- **Deployment costs:** 7B–14B parameters require high-memory GPUs (12–80 GB)
- **Data requirements:** Interleaved multimodal corpora hard to curate for domain-specific tasks

## When to Use BAGEL

✅ **Use as reference when:**
- Designing unified multimodal architectures for neuro-omics
- Exploring MoT-style expert routing for gene + brain modalities
- Building LLM-guided clinical report generation from brain imaging

⚠️ **Do not use directly for:**
- Neuroimaging embedding extraction (use BrainLM, SwiFT, etc.)
- Genetic sequence modeling (use Caduceus, Evo2, etc.)
- Production clinical workflows (general model, not clinically validated)

⚠️ **Consider alternatives:**
- **M3FM:** For medical imaging + text with CLIP-style alignment
- **BrainMT:** For neuroimaging with efficient long-context modeling
- **Caduceus + BrainLM fusion:** For gene-brain integration with domain-specific FMs

## Reference Materials

### Knowledge Base Resources

**Curated materials in this KB:**
- **Paper summary & notes (PDF):** [BAGEL (2025)](../../generated/kb_curated/papers-pdf/bagel_2025.pdf)
- **Code walkthrough:** [BAGEL walkthrough](../../code_walkthroughs/bagel_walkthrough.md)
- **Model card (YAML):** `kb/model_cards/bagel.yaml` (if exists)
- **Paper card (YAML):** `kb/paper_cards/bagel_2025.yaml`

**Integration recipes:**
- [Multimodal Architectures](../../integration/multimodal_architectures.md)
- [Design Patterns](../../integration/design_patterns.md)
- [Integration Strategy](../../integration/integration_strategy.md)

### Original Sources

**Source code repositories:**
- **Local copy:** `external_repos/bagel/`
- **Official GitHub:** [ChaofanTao/BAGEL](https://github.com/ChaofanTao/BAGEL)

**Original paper:**
- **Title:** "Emerging Properties in Unified Multimodal Pretraining"
- **Authors:** Deng, Chaorui; Zhu, Deyao; Li, Kunchang; Gou, Chenhui; Li, Feng; Wang, Zeyu; Zhong, Shu; Yu, Weihao; Nie, Xiaonan; Song, Ziang; Shi, Guang; Fan, Haoqi
- **Published:** arXiv preprint, 2025
- **Link:** [arXiv:2505.14683](https://arxiv.org/abs/2505.14683)
- **PDF (local):** [bagel_2025.pdf](../../generated/kb_curated/papers-pdf/bagel_2025.pdf)

## Next Steps in Our Pipeline

1. **Architecture study:** Extract MoT patterns for potential gene-brain expert routing
2. **Interleaved data design:** Inform how to structure mixed gene + brain + behavior sequences
3. **LLM integration:** Study CFG and generation strategies for clinical report synthesis
4. **Evaluation framework:** Adapt IntelligentBench patterns for neuro-omics emergent capabilities
5. **Clinical extension:** Combine BAGEL insights with M3FM for brain imaging + clinical text

## Engineering Notes

- BAGEL uses **packed sequences** with modality-specific indices—applicable to gene + brain token mixing
- **CFG contexts** (text/image guidance) are plain dicts—easy to extend to clinical conditioning
- **Quantization** (NF4, INT8) provides deployment patterns for resource-constrained clinical settings
- **FSDP + EMA** training pipeline applicable to large-scale neuro-omics model training


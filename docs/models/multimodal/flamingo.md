---
title: Flamingo — Model Card
status: active
updated: 2025-11-26
---

# Flamingo

## Overview

**Type:** Visual language model (VLM)  
**Architecture:** Perceiver-augmented vision encoder + gated cross-attention over a causal LM  
**Modality:** Images, videos, text (interleaved)  
**Primary use:** Few-shot multimodal understanding and text generation

## Purpose & Design Philosophy

Flamingo extends large language models to the visual domain by **bridging frozen vision and language
backbones** with a lightweight Perceiver Resampler and gated cross-attention layers. Instead of
fine-tuning on each downstream task, Flamingo is trained once on large-scale web multimodal data and
then adapted via in-context examples, mirroring GPT-3-style few-shot prompting for text.^[See
arXiv:2204.14198](https://arxiv.org/abs/2204.14198)

**Key idea:** keep powerful pretrained vision and language models intact, and add minimal, well-
behaved connectors that enable multimodal reasoning without catastrophic forgetting.

## Architecture Highlights

- **Vision encoder:** Pretrained CLIP/OpenCLIP-style ViT (e.g., ViT-L/14) for images or video frames.
- **Perceiver Resampler:** Converts variable-resolution feature maps into a fixed set of visual tokens
  via cross-attention from learnable latent queries.
- **Language model:** Pretrained causal LM (e.g., Chinchilla/MPT/RedPajama), largely frozen.
- **GATED XATTN-DENSE layers:** Inserted between LM blocks to cross-attend from text tokens to visual
  tokens, with tanh-gated residuals for stable training.
- **Interleaved inputs:** Sequences of images/videos and text with `<image>` and `<|endofchunk|>`
  markers; image-causal masking ensures each text span only sees its associated images.

For implementation details, see the OpenFlamingo factory and Flamingo wrapper in the code
walkthrough.

## Integration Strategy

### For Neuro-Omics KB

Flamingo is **not a primary model** in current neuro-omics experiments but serves as a design
reference for:

- **Scan-conditioned report generation:** Replace the CLIP encoder with brain encoders (BrainLM,
  BrainMT, Brain Harmony) so fMRI/sMRI tokens play the role of image tokens.
- **Multimodal adapters:** Reuse the Perceiver Resampler concept for compressing high-dimensional
  brain features into a fixed number of tokens.
- **LLM semantic bridge:** Use Flamingo-style gated cross-attention to inject brain/genetics
  embeddings into language models (see `kb/model_cards/llm_semantic_bridge.yaml`).

### For ARPA-H Brain-Omics Models

Flamingo illustrates how to:

- Keep **foundation encoders frozen** while adding small multimodal connectors.
- Structure **interleaved multimodal sequences** that include context examples followed by a query.
- Build **few-shot-capable architectures** without task-specific heads for every benchmark.

These patterns carry over to Brain–Omics–LLM stacks that must reason jointly over genetics, brain
imaging, and clinical text.

## Reference Materials

### Knowledge Base Resources

- **Paper summary:** `docs/generated/kb_curated/papers-md/flamingo_2022.md`
- **Paper card (YAML):** `kb/paper_cards/flamingo_2022.yaml`
- **Code walkthrough:** `docs/code_walkthroughs/flamingo_walkthrough.md`
- **Model card (YAML):** `kb/model_cards/flamingo.yaml`

### Original Sources

- **Official implementation:** [OpenFlamingo GitHub](https://github.com/mlfoundations/open_flamingo)
- **Paper:** *Flamingo: a Visual Language Model for Few-Shot Learning* (NeurIPS 2022)^[arXiv:2204.14198](https://arxiv.org/abs/2204.14198)



---
title: HyenaDNA — Model Card
status: active
updated: 2025-11-26
---

# HyenaDNA

## Overview

**Type:** Long-context DNA foundation model  
**Architecture:** Decoder-only Hyena operators (implicit convolutions)  
**Modality:** Nucleotide sequences (DNA)  
**Primary use (conceptual in KB):** Reference architecture for 1M-token genomic modeling

## Purpose & Design Philosophy

HyenaDNA demonstrates that **sub-quadratic sequence operators** can scale genomic language models to
1M-token contexts at single-nucleotide resolution, breaking the context-length barrier imposed by
quadratic attention while preserving fine-grained variant information.^[See
arXiv:2306.15794](https://arxiv.org/abs/2306.15794) It is trained as a next-nucleotide predictor on
the human reference genome and evaluated on standard regulatory element benchmarks, showing that
carefully designed implicit convolutions can match or exceed attention-based DNA LMs with far fewer
parameters and data.

## Architecture Highlights

- **Operators:** Hyena implicit convolutions with data-controlled gating (no self-attention).
- **Context length:** Up to 1,000,000 tokens (1Mbp) with character-level tokenization.
- **Training tricks:** Sequence-length warm-up schedule, gradient checkpointing for ultralong
  inputs, soft prompts for downstream adaptation.
- **Outputs:** Per-position logits/embeddings suitable for downstream pooling (gene, enhancer,
  window-level features).

HyenaDNA is not currently vendored as code in this KB; instead, the generic StripedHyena codebase in
`external_repos/hyena/` is used for architectural walkthroughs.

## Integration Strategy

### For Neuro-Omics KB

HyenaDNA is tracked as a **long-context genomics reference**:

- Informs the design of ultra-long-context pipelines built around Evo 2 (StripedHyena 2) for
  regulatory-region and whole-locus embeddings.
- Motivates experimenting with 100kb–1Mbp windows when studying distal regulatory effects on
  brain-related genes.
- Suggests that sequence-length warm-up and soft prompting should be standard recipes when
  introducing Hyena/StripedHyena operators into neuro-omics models.

Concrete embeddings in this KB currently use **Caduceus**, **DNABERT-2**, **Evo 2**, and
**GENERaTOR**; HyenaDNA is kept as a design anchor and potential future encoder once public
checkpoints and code stabilise.

## Reference Materials

### Knowledge Base Resources

- **Paper summary:** `docs/generated/kb_curated/papers-md/hyenadna_2023.md`
- **Paper card (YAML):** `kb/paper_cards/hyenadna_2023.yaml`
- **Model card (YAML):** `kb/model_cards/hyenadna.yaml`
- **Architecture walkthrough:** `docs/code_walkthroughs/hyena_walkthrough.md` (StripedHyena core)

### Original Sources

- **Paper:** *HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution*
  (NeurIPS 2023)^[arXiv:2306.15794](https://arxiv.org/abs/2306.15794)
- **Hyena / StripedHyena code:** see [StripedHyena GitHub](https://github.com/togethercomputer/stripedhyena) and
  related Hyena project repositories referenced in the paper.



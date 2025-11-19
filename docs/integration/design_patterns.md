---
title: Integration Design Patterns
status: active
updated: 2025-11-19
---

# Integration Design Patterns

> **Default strategy:** Late fusion via separate gene/brain heads (see [Integration Strategy](integration_strategy.md)). The patterns below document **escalation paths** for when baseline fusion demonstrates significant gains, with risks and implementation guidance noted.

## Overview

This document catalogs integration architectures from simplest (late fusion) to most complex (unified multimodal transformers). Each pattern includes:
- **Description:** How the pattern works
- **Use cases:** When to apply it
- **Risks:** What can go wrong
- **Examples:** Reference models demonstrating the pattern
- **Escalation criteria:** When to move to this pattern from simpler baselines

---

## Pattern 1: Late Fusion (Baseline)

### Description
Train separate encoders for each modality, extract fixed-size embeddings, concatenate, and pass to a simple classifier (LR, GBDT). No parameter sharing between modalities during training.

### Architecture
```
Genetics FM → gene_embed [512-D]  ──┐
                                     ├─→ concat [1024-D] → LR/GBDT → prediction
Brain FM    → brain_embed [512-D] ──┘
```

### Use Cases
- **Initial baselines:** Establish single-modality vs. fusion performance
- **Heterogeneous modalities:** Genetics (sequence) vs. brain (spatiotemporal) have different semantics
- **Small sample sizes:** Fewer parameters to tune than joint training

### Risks
- Suboptimal if modalities have strong cross-modal dependencies
- No learned alignment between modality spaces

### Implementation (Current)
- **Genetics:** Caduceus/DNABERT-2 → gene-level embeddings → PCA-512
- **Brain:** BrainLM/Brain-JEPA → subject embeddings → PCA-512
- **Fusion:** `np.concatenate([gene_embed, brain_embed], axis=-1)`
- **Classifier:** LogisticRegression (balanced, C=0.01) or LightGBM

### Escalation Criteria
✅ Move to Pattern 2 when: Fusion significantly outperforms max(Gene, Brain) with p < 0.05 (DeLong test)

---

## Pattern 2: Two-Tower Contrastive Alignment

### Description
Freeze pretrained modality encoders, add small learnable projectors, align via contrastive loss (InfoNCE). Creates shared embedding space without modifying foundation models.

### Architecture
```
Genetics FM (frozen) → gene_embed → projector_gene [256-D]  ──┐
                                                               ├─→ InfoNCE loss
Brain FM (frozen)    → brain_embed → projector_brain [256-D] ──┘
                                           ↓
                               aligned_space [256-D] → classifier
```

### Use Cases
- **Cross-modal retrieval:** Find genes associated with brain patterns
- **Zero-shot transfer:** Align modalities on one cohort, transfer to another
- **Foundation model preservation:** Keep pretrained weights frozen

### Risks
- Requires paired gene-brain samples (not all subjects have both modalities)
- Contrastive loss sensitive to negative sampling strategy
- May not capture complex non-linear interactions

### Examples
- **CLIP:** Image-text contrastive alignment (OpenAI)
- **M3FM:** Medical image-text two-tower fusion ([walkthrough](../code_walkthroughs/m3fm_walkthrough.md))

### Implementation Strategy
1. Freeze Caduceus and BrainLM checkpoints
2. Add 512→256 projectors (2-layer MLP with ReLU)
3. Sample positive pairs: (gene_i, brain_i) from same subject
4. Sample negatives: (gene_i, brain_j≠i) within batch
5. Optimize InfoNCE loss on train split
6. Extract aligned embeddings → downstream classifier

### Escalation Criteria
✅ Move to Pattern 3 when: Cross-modal retrieval tasks emerge or need end-to-end joint training

---

## Pattern 3: Early Fusion with Shared Encoder

### Description
Concatenate or interleave modality embeddings early, process through shared transformer layers. Enables cross-modal attention but requires careful preprocessing alignment.

### Architecture
```
gene_tokens [N_genes, D]  ──┐
                            ├─→ concat → Shared Transformer → pooled_embed → classifier
brain_tokens [N_parcels, D] ─┘
```

### Use Cases
- **Complex interactions:** When modalities have intricate dependencies (e.g., gene regulatory networks affecting brain circuits)
- **Multi-task learning:** Share encoder across prediction tasks (MDD, cognitive scores)
- **End-to-end optimization:** Allow gradients to flow through all layers

### Risks
- **Modality imbalance:** Dominant modality can suppress weaker one
- **Overfitting:** More parameters, higher risk with small N
- **Preprocessing coupling:** Requires consistent tokenization/normalization across modalities

### Examples
- **BAGEL:** Unified decoder over text+image+video tokens ([paper card](../generated/kb_curated/papers-md/bagel_2025.md))
- **Brain Harmony:** Joint sMRI+fMRI processing with hub tokens ([model card](../models/brain/brainharmony.md))

### Implementation Strategy
1. Tokenize both modalities to fixed dimensions
2. Add modality-specific positional encodings
3. Concatenate token sequences: `[gene_tok_1, ..., gene_tok_N, brain_tok_1, ..., brain_tok_M]`
4. Process through transformer layers with cross-modal attention
5. Pool final layer → classification head

### Escalation Criteria
✅ Move to Pattern 4 when: Need modality-specific parameter sets (avoid modality collapse)

---

## Pattern 4: Mixture-of-Transformers (MoT) Sparse Fusion

### Description
Shared self-attention over all modality tokens, but **separate FFNs and layer norms per modality**. Balances cross-modal attention with modality-specific processing.

### Architecture
```
gene_tokens + brain_tokens + behavior_tokens
    ↓
Shared Self-Attention (all tokens interact)
    ↓
Modality-specific FFN branches:
  ├─ genetics_ffn
  ├─ brain_ffn
  └─ behavior_ffn
    ↓
Pooled embedding → task heads
```

### Use Cases
- **Compute efficiency:** ~55% FLOPs vs. full dense multimodal transformer
- **Modality preservation:** Each modality retains specialized processing
- **Scalable fusion:** Handle 3+ modalities without parameter explosion

### Risks
- More complex architecture than dense baseline
- Requires careful initialization of per-modality parameters
- May underperform dense if modalities highly correlated

### Examples
- **MoT paper:** Sparse multimodal transformer ([arXiv:2411.04996](https://arxiv.org/abs/2411.04996), [card](../generated/kb_curated/papers-md/mot_2025.md))

### Implementation Strategy
1. Initialize shared attention layers (all modalities)
2. Create separate FFN/norm modules per modality (genetics_ffn, brain_ffn, behavior_ffn)
3. Forward pass: attention(all_tokens) → route_to_modality_ffn(token) → merge
4. Train end-to-end with task-specific heads

### Escalation Criteria
✅ Move to Pattern 5 when: Need generative capabilities (e.g., clinical report generation from gene-brain data)

---

## Pattern 5: Unified Brain-Omics Model (BOM)

### Description
Single decoder-only transformer with Mixture-of-Experts (MoE) processing **all modalities as token sequences**: genetics (nucleotide tokens), brain (parcel/voxel tokens), behavior (structured tokens), language (text tokens). Inspired by BAGEL/GPT-4o-style unified multimodal architectures.

### Architecture
```
Tokenize all modalities:
  - Genetics: nucleotide sequences → tokens
  - Brain MRI: 3D patches → tokens (ViT-style)
  - fMRI: parcel time series → tokens
  - EEG: channel × time → tokens
  - Behavior: structured data → embedding tokens
  - Language: text → BPE tokens

  ↓
Unified Decoder-Only Transformer (e.g., LLaMA-style)
  - Mixture-of-Experts (understanding vs. generation)
  - Cross-modal self-attention
  - Next-token prediction objective

  ↓
Downstream tasks:
  - Gene-brain association discovery
  - Clinical report generation
  - Counterfactual reasoning ("what if gene X was mutated?")
  - Cognitive decline prediction
```

### Use Cases
- **ARPA-H Brain-Omics Model (BOM):** Unified foundation model for neuro-omics
- **LLM as semantic bridge:** Language model embeddings as "lingua franca" for cross-modal reasoning
- **Generative tasks:** Report generation, sequence design, counterfactual prediction
- **Unified pretraining:** Single model handles all neuro-omics modalities

### Risks
- **Massive compute:** Requires trillions of tokens, large-scale infrastructure
- **Data curation:** Need high-quality interleaved multimodal corpus
- **Complexity:** Hardest to debug, longest training time
- **Evaluation:** Requires diverse benchmarks across modalities

### Examples
- **BAGEL:** Unified text+image+video+web model ([walkthrough](../code_walkthroughs/bagel_walkthrough.md))
- **GPT-4o:** Unified multimodal reasoning (proprietary)
- **Chameleon:** Text-image unified autoregressive model

### Implementation Strategy (Long-term Vision)

**Phase 1: Corpus Curation**
- Collect interleaved multimodal neuro-omics data:
  - Genetic variants + brain scans + cognitive assessments + clinical notes
  - Longitudinal trajectories (developmental, disease progression)
  - Multimodal annotations (gene function descriptions, brain region labels, symptom text)

**Phase 2: Tokenization**
- Genetics: Nucleotide sequences (A/C/G/T) or k-mer tokens
- Brain MRI: 3D patch tokens (ViT-style, 16³ patches)
- fMRI: Parcel time series → temporal tokens
- EEG: Channel-time matrices → spectral-spatial tokens
- Behavior: Structured scores → learned embeddings
- Language: Standard BPE/SentencePiece tokens

**Phase 3: Architecture**
- Decoder-only transformer (LLaMA/Qwen base)
- Mixture-of-Experts: Understanding vs. generation experts
- Modality-specific input embedders, shared transformer backbone
- Task-specific heads (classification, generation, retrieval)

**Phase 4: Training**
- Next-token prediction across all modalities
- Interleaved sequence objective (language → genetics → brain → language)
- Multitask loss: prediction + generation + contrastive

**Phase 5: Evaluation**
- Gene-brain association discovery (AUC, correlation)
- Clinical report generation (BLEU, METEOR, clinician ratings)
- Cognitive prediction (AUROC on MDD, fluid intelligence)
- Cross-modal retrieval (gene→brain, brain→phenotype)
- Counterfactual reasoning (GPT-4-judge evaluation)

### Escalation Criteria
✅ Implement BOM when: Phases 1-4 patterns exhausted, significant funding secured, compute infrastructure available

---

## ARPA-H Integration Roadmap

### Timeline

| Phase | Pattern | Status | Target Completion |
|-------|---------|--------|-------------------|
| **Phase 1** | Late fusion baselines | In progress | Nov 2025 |
| **Phase 2** | Two-tower contrastive | Pending | Q1 2026 |
| **Phase 3** | Early fusion / MoT | Pending | Q2 2026 |
| **Phase 4** | Unified BOM (pilot) | Planned | Q3-Q4 2026 |
| **Phase 5** | Scaled BOM deployment | Vision | 2027+ |

### Current Focus (Nov 2025)

**Active:**
- ✅ Late fusion: Gene (Caduceus) + Brain (BrainLM) → LR/GBDT
- ✅ CCA + permutation: Assess cross-modal correlation structure
- ✅ LOGO attribution: Gene-level importance (Yoon et al. protocol)

**Next steps:**
1. Complete late fusion baselines on UKB gene-brain data
2. Pilot two-tower contrastive alignment (frozen encoders)
3. Design interleaved corpus for Phase 3+ experiments

---

## Reference Materials

**Multimodal architecture examples:**
- [Multimodal Architectures Overview](multimodal_architectures.md) — Detailed patterns from BAGEL, MoT, M3FM, Me-LLaMA, TITAN

**Integration strategies:**
- [Integration Strategy](integration_strategy.md) — Preprocessing, harmonization, escalation criteria
- [CCA + Permutation Recipe](analysis_recipes/cca_permutation.md) — Statistical testing before fusion
- [Prediction Baselines](analysis_recipes/prediction_baselines.md) — Late fusion implementation

**Model documentation:**
- [Brain Models Overview](../models/brain/index.md)
- [Genetics Models Overview](../models/genetics/index.md)

**Decision logs:**
- [Integration Baseline Plan (Nov 2025)](../decisions/2025-11-integration-plan.md) — Why late fusion first

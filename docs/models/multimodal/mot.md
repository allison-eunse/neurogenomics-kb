---
title: MoT — Model Card
status: active
updated: 2025-11-19
---

# Mixture-of-Transformers (MoT)

## Overview

**Type:** Sparse Multimodal Transformer Architecture  
**Architecture:** Modality-aware sparse transformer with global self-attention  
**Modality:** Text, images, speech (unified token sequences)  
**Primary use:** Compute-efficient multimodal foundation models with 40–60% FLOP savings

## Purpose & Design Philosophy

Mixture-of-Transformers (MoT) introduces **modality-aware sparsity** to make large multimodal foundation models dramatically more efficient. Instead of a single dense transformer over all modalities, MoT decouples all non-embedding parameters (FFNs, attention projections, layer norms) by modality while keeping **global self-attention** over the full sequence. This structured sparsity matches dense baseline performance while using only 40–60% of pretraining FLOPs and significantly reduces wall-clock training time.

**Key innovation:** Rule-based routing by modality (not learned MoE routing) provides stability and simplicity while achieving substantial compute savings.

## Architecture Highlights

- **Sparsity mechanism:** Modality-aware parameter decoupling (separate FFNs, attention matrices, layer norms per modality)
- **Shared attention:** Full self-attention over mixed sequences—no routing-based attention sparsity
- **Parameter selection:** Token modality tag determines which parameter set to use
- **Compatibility:** Drop-in replacement for dense transformers in Chameleon and Transfusion architectures
- **Scaling:** Evaluated from 37M to 7B parameters across multiple settings

## Integration Strategy

### For Neuro-Omics KB

MoT provides **efficiency patterns** for gene-brain-behavior integration:

**Key lessons:**
- **Modality-specific processing:** Separate genomics FFN + brain FFN + shared attention over joint sequences
- **Compute savings:** 40–60% FLOP reduction applicable to large-scale neuro-omics pretraining
- **Stable training:** No MoE routing instability—deterministic modality selection
- **Implementation simplicity:** Easy to implement vs. complex load-balancing in MoE

**Application to KB pipeline:**
```python
# Pseudocode for neuro-omics MoT
for token in sequence:
    if token.modality == "gene":
        ffn_output = gene_ffn(attention_output)
    elif token.modality == "brain":
        ffn_output = brain_ffn(attention_output)
    elif token.modality == "behavior":
        ffn_output = behavior_ffn(attention_output)
    # Shared self-attention across all modalities
```

### For ARPA-H Brain-Omics Model (BOM)

MoT demonstrates **scalable multimodal architectures**:

```
Gene tokens   → |
                | Global self-attention (dense)
Brain tokens  → |     ↓
                | Modality-aware FFNs (sparse)
Text tokens   → |     ↓
                | Prediction heads
```

**Transfer insights:**
- **Efficiency-first design:** Critical for scaling to population-level datasets (UK Biobank, HCP)
- **Leave-one-modality-out:** MoT evaluation patterns inform ablation studies for gene-brain fusion
- **Hybrid models:** Combining MoT (modality sparsity) with MoE (expert routing) for complementary benefits
- **Systems optimization:** Wall-clock profiling applicable to neuro-omics training runs

## Embedding Extraction Workflow

MoT is an **architectural pattern**, not a standalone model, but if implementing for neuro-omics:

```bash
# 1. Tag tokens by modality (gene / brain / behavior)
# 2. Build MoT transformer with modality-specific FFNs
# 3. Forward through model (global attention + modality FFNs)
# 4. Extract embeddings before task-specific heads
# 5. Use for downstream fusion tasks
```

**For implementation:** See MoT paper code repository and adapt to neuro-omics modalities.

## Strengths & Limitations

### Strengths
- **Dramatic compute savings:** 40–60% FLOP reduction with matched performance
- **Training stability:** No MoE routing instability or load-balancing overhead
- **Implementation simplicity:** Rule-based routing easier than learned expert selection
- **Extensive evaluation:** Multiple settings (Chameleon, Transfusion), scales (37M–7B), and system profiling

### Limitations
- **Modality labels required:** Tokens must be pre-tagged by modality
- **Limited to tested modalities:** Text, images, speech—no structured data (tables, graphs, sequences)
- **No within-modality routing:** Single FFN per modality—no fine-grained specialization
- **Infrastructure-specific:** Results tied to specific training setups (AWS p4de, A100s)

## When to Use MoT

✅ **Use when:**
- Building large-scale multimodal models with limited compute budgets
- Want structured sparsity without MoE training complexity
- Need stable, deterministic routing by modality
- Scaling neuro-omics models to population-level datasets

⚠️ **Defer until:**
- Dense baselines established (per Nov 2025 integration plan)
- Modality boundaries clear (e.g., which brain features are "brain" vs "behavior")
- Engineering resources available for custom MoT implementation

⚠️ **Consider alternatives:**
- **Dense fusion:** Simpler baseline for initial gene-brain experiments
- **MoE architectures:** If need learned task-specific routing
- **Late fusion:** If modalities processed independently before combination

## Reference Materials

**Primary sources:**
- **Paper:** [MoT (2025)](../../generated/kb_curated/papers-md/mot_2025.md) — TMLR 2025
- **Code walkthrough:** [MoT walkthrough](../../code_walkthroughs/mot_walkthrough.md)
- **Paper card:** `kb/paper_cards/mot_2025.yaml`

**Integration recipes:**
- [Multimodal Architectures](../../integration/multimodal_architectures.md)
- [Design Patterns](../../integration/design_patterns.md)
- [Integration Strategy](../../integration/integration_strategy.md)

**Source repository:**
- **Local:** `external_repos/MoT/`
- **GitHub:** [Meta Mixture-of-Transformers](https://github.com/facebookresearch/MoT)

## Next Steps in Our Pipeline

1. **Architecture adaptation:** Design gene-brain-behavior MoT variant
2. **Efficiency benchmarking:** Compare MoT vs dense fusion on UKB cognitive tasks
3. **Ablation studies:** Implement leave-one-modality-out for gene-brain analysis
4. **Hybrid exploration:** Test MoT + MoE combination for neuro-omics
5. **Systems profiling:** Measure wall-clock and FLOP savings on KB training runs

## Engineering Notes

- MoT FLOPs match dense models with same parameter budget—**key for fair comparison**
- **Modality separation analysis** in paper informs how to design gene/brain/behavior boundaries
- **Hybrid MoT+MoE** results suggest complementary benefits for future neuro-omics architectures
- **Transfusion compatibility** shows MoT works with mixed objectives (autoregressive + diffusion)


---
title: M3FM — Model Card
status: active
updated: 2025-11-19
---

# M3FM (Multimodal, Multidomain, Multilingual Medical Foundation Model)

## Overview

**Type:** Medical Vision-Language Foundation Model  
**Architecture:** MultiMedCLIP (CLIP-style) + MultiMedLM (medical LLM)  
**Modality:** Chest X-ray, CT, radiology reports (English + Chinese)  
**Primary use:** Zero-shot medical report generation and disease diagnosis across domains and languages

## Purpose & Design Philosophy

M3FM is a medical foundation model designed for **zero-shot radiology report generation and diagnosis** across imaging modalities (CXR, CT) and languages (English, Chinese). The model learns a shared vision-language embedding space through contrastive learning (MultiMedCLIP), then trains a multilingual medical LLM (MultiMedLM) to generate reports and support diagnosis without labeled data in the target domain or language.

**Key innovation:** Single model handles multiple imaging modalities and languages through CLIP-style alignment + medical LLM, enabling deployment where labeled data is scarce.

## Architecture Highlights

- **Two-stage training:**
  - **Stage 1:** MultiMedCLIP aligns images (CXR, CT) with English text, and English-Chinese text pairs
  - **Stage 2:** MultiMedLM trained on multilingual corpora for report generation
- **Vision encoder:** CNN/ViT encoding CXR and CT to visual embeddings
- **Text encoder/decoder:** Transformer-based for English and Chinese reports
- **Alignment:** CLIP-like contrastive loss creates shared embedding space
- **Inference:** Zero-shot report generation via visual → text decoding through aligned space

## Integration Strategy

### For Neuro-Omics KB

M3FM provides **medical imaging integration patterns**:

**Key lessons for brain imaging + clinical text:**
- **Two-tower alignment:** Separate brain imaging encoder + clinical text encoder with contrastive loss
- **Zero-shot transfer:** Applicable to new cohorts (e.g., Cha Hospital) without labeled data
- **Multilingual support:** Extend brain-behavior models to non-English populations
- **Report generation:** Automate clinical summaries from neuroimaging

**Potential adaptation:**
```
Brain MRI/fMRI → Vision encoder (SwiFT/BrainLM) → |
                                                   | Contrastive alignment
Clinical notes → Text encoder (medical LLM)     → |
                                                   ↓
                                           Shared latent space
                                                   ↓
                                           Report generation LLM
```

### For ARPA-H Brain-Omics Model (BOM)

M3FM demonstrates **clinical translation patterns**:

```
Brain embeddings → |
                   | Two-tower contrastive alignment
Clinical text    → |     ↓
                   | Shared embedding space
Gene annotations → |     ↓
                   | Medical LLM for report generation
```

**Transfer insights:**
- **Zero-shot diagnosis:** Critical for rare neurological disorders with limited training data
- **Cross-domain generalization:** M3FM's CXR→CT transfer informs MRI→fMRI→CT transfers
- **Multilingual clinical AI:** Extend neuro-omics models to global cohorts
- **Few-shot learning:** Strong performance with minimal downstream labels

## Embedding Extraction Workflow

If adapting M3FM for brain imaging:

```bash
# 1. Train CLIP-style alignment on brain scans + clinical notes
# 2. Load pretrained brain FM (SwiFT, BrainLM) as vision encoder
# 3. Load medical LLM (Me-LLaMA, etc.) as text encoder
# 4. Contrastive training on paired brain-text data
# 5. Extract embeddings from shared space for downstream tasks
```

**For neuro-omics:**
- **Vision encoder:** SwiFT (fMRI) or BrainLM (3D volumes)
- **Text encoder:** Medical LLM pretrained on neurology literature
- **Alignment data:** Brain scans + radiology reports from UKB, HCP

## Strengths & Limitations

### Strengths
- **Genuine zero-shot:** Generates reports without labeled downstream data
- **Cross-domain + cross-language:** Single model handles CXR, CT, English, Chinese
- **Clinical validation:** Evaluated on 9 downstream datasets (COVID-19, TB, etc.)
- **Practical:** Leverages machine translation to bootstrap multilingual capabilities

### Limitations
- **Machine translation artifacts:** Reliance on MT for Chinese may introduce biases
- **Modality coverage:** Only CXR and CT—no MRI, ultrasound, pathology
- **Compute intensive:** Requires substantial resources for two-stage training
- **Evaluation gaps:** Standard metrics may not capture clinical safety

## When to Use M3FM

✅ **Use as reference when:**
- Building brain imaging + clinical text models
- Designing zero-shot transfer for new cohorts
- Implementing CLIP-style alignment for neuro-omics
- Supporting multilingual neuroimaging research

⚠️ **Do not use directly for:**
- Neuroimaging (trained on CXR/CT, not brain scans)
- Production clinical diagnosis (requires validation)
- Non-imaging modalities (no genetics support)

⚠️ **Consider alternatives:**
- **BAGEL/MoT:** For unified understanding + generation
- **TITAN:** For high-resolution pathology imaging
- **Me-LLaMA:** For medical LLM without imaging

## Reference Materials

### Knowledge Base Resources

**Curated materials in this KB:**
- **Paper summary & notes (PDF):** [M3FM (2025)](../../generated/kb_curated/papers-pdf/m3fm_2025.pdf)
- **Code walkthrough:** [M3FM walkthrough](../../code_walkthroughs/m3fm_walkthrough.md)
- **Model card (YAML):** `kb/model_cards/m3fm.yaml`
- **Paper card (YAML):** `kb/paper_cards/m3fm_2025.yaml`

**Integration recipes:**
- [Multimodal Architectures](../../integration/multimodal_architectures.md)
- [Design Patterns](../../integration/design_patterns.md) — Two-tower contrastive section
- [Integration Strategy](../../integration/integration_strategy.md)

### Original Sources

**Source code repositories:**
- **Local copy:** `external_repos/M3FM/`
- **Official GitHub:** [ai-in-health/M3FM](https://github.com/ai-in-health/M3FM)

**Original paper:**
- **Title:** "M3FM: A Multimodal, Multidomain, Multilingual Medical Foundation Model for Zero‑Shot Clinical Diagnosis"
- **Authors:** Liu, Fenglin; Li, Zheng; Yin, Qingyu; Huang, Jinfa; Luo, Jiebo; Thakur, Anshul; Branson, Kim; Schwab, Patrick; Yin, Bing; Wu, Xian; Zheng, Yefeng; Clifton, David A.
- **Published:** npj Digital Medicine, 2025
- **Link:** [Nature: s41746-024-01339-7](https://www.nature.com/articles/s41746-024-01339-7)
- **DOI:** [10.1038/s41746-024-01339-7](https://doi.org/10.1038/s41746-024-01339-7)
- **PDF (local):** [m3fm_2025.pdf](../../generated/kb_curated/papers-pdf/m3fm_2025.pdf)

## Next Steps in Our Pipeline

1. **CLIP adaptation:** Implement brain imaging + clinical text contrastive learning
2. **Zero-shot evaluation:** Test on new cohorts (Cha Hospital) without fine-tuning
3. **Multilingual extension:** Adapt to Korean clinical notes for Cha pediatric cohort
4. **Report generation:** Automate neuroimaging report synthesis from embeddings
5. **Diagnostic support:** Combine M3FM patterns with gene-brain fusion for clinical predictions

## Engineering Notes

- M3FM's **two-stage training** separates alignment from generation—applicable to neuro-omics
- **Contrastive learning** requires paired data—use UKB radiology reports + imaging
- **Machine translation** can bootstrap multilingual capabilities before human-labeled data available
- **Zero-shot evaluation** critical for rare neurological disorders with <100 cases


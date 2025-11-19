---
title: FMS-Medical Catalog — Resource Card
status: active
updated: 2025-11-19
---

# FMS-Medical: Foundation Models for Advancing Healthcare (Catalog)

## Overview

**Type:** Knowledge Base / Survey Resource  
**Format:** Curated repository of medical foundation models + datasets  
**Coverage:** Language (LFM), Vision (VFM), Bioinformatics (BFM), Multimodal (MFM)  
**Primary use:** Model and dataset discovery, benchmarking reference, literature review

## Purpose & Design Philosophy

FMS-Medical is an "awesome list" style knowledge base tracking foundation model research across healthcare modalities. Maintained as a GitHub repository with bilingual documentation (English + Chinese), it provides structured references to 100+ medical FM papers, datasets, tutorials, and related resources organized by modality and year. The resource is anchored by an IEEE Reviews in Biomedical Engineering survey paper and serves as a comprehensive entry point for medical AI research.

**Key value:** Centralized, actively maintained catalog of medical FMs and datasets—ideal for systematic literature review and baseline selection.

## Catalog Structure

### Model Taxonomies

| Category | Coverage | Use Case |
|----------|----------|----------|
| **LFM (Language)** | Medical LLMs, clinical NLP | Text understanding, report generation, QA |
| **VFM (Vision)** | Medical image encoders, segmentation | Radiology, pathology, ultrasound analysis |
| **BFM (Bioinformatics)** | Genomics, proteomics, drug discovery | Sequence modeling, variant interpretation |
| **MFM (Multimodal)** | Vision-language, integrated models | Unified diagnosis, multimodal reasoning |

### Dataset Catalogs

- **Text datasets:** Clinical notes, radiology reports, biomedical literature
- **Imaging datasets:** CXR, CT, MRI, pathology, ultrasound, retinal
- **Omics datasets:** Genomics, transcriptomics, proteomics
- **Multimodal datasets:** Image-text pairs, integrated EHR + imaging

## Integration Strategy

### For Neuro-Omics KB

FMS-Medical provides **dataset discovery and model benchmarking**:

**Key uses:**
- **Literature review:** Identify related work in medical AI for neuro-omics
- **Dataset selection:** Find imaging and genomics datasets for validation
- **Baseline comparison:** Track state-of-the-art methods for benchmarking
- **Survey reference:** Cite comprehensive medical FM survey

**Application to KB pipeline:**
```
FMS-Medical catalog
    ↓
Extract relevant entries:
    - Brain imaging models
    - Genomics foundation models
    - Multimodal medical datasets
    ↓
Populate KB model/dataset cards
    ↓
Benchmark neuro-omics methods against medical FM baselines
```

### For ARPA-H Brain-Omics Model (BOM)

FMS-Medical informs **medical AI landscape understanding**:

```
Survey medical FMs
    ↓
Identify integration patterns:
    - CLIP-style alignment (M3FM, TITAN)
    - MoT/MoE architectures
    - LLM continual pretraining (Me-LLaMA)
    ↓
Apply to BOM design
```

**Transfer insights:**
- **Comprehensive coverage:** Survey spans all medical FM modalities—identify gaps for neuro-omics
- **Dataset catalogs:** Find publicly available datasets for cross-validation
- **Benchmark references:** Track medical FM performance for comparison
- **Bilingual support:** Chinese documentation aids international collaboration

## Catalog Access Workflow

```bash
# 1. Clone or sync FMS-Medical repository
git clone https://github.com/YutingHe-list/Awesome-Foundation-Models-for-Advancing-Healthcare

# 2. Review README for model/dataset tables

# 3. Extract entries relevant to neuro-omics:
#    - VFM: Brain imaging models
#    - BFM: Genomics models
#    - MFM: Multimodal integration patterns

# 4. Populate KB YAML cards from extracted entries

# 5. Track updates via GitHub (actively maintained)
```

**For KB automation:**
- Parse README tables to generate candidate model/dataset cards
- Link to survey PDFs for detailed descriptions
- Monitor repository updates for new medical FMs

## Strengths & Limitations

### Strengths
- **Comprehensive coverage:** 100+ medical FMs across all modalities
- **Actively maintained:** Regular updates with publication news
- **Bilingual:** English + Chinese documentation
- **Well-organized:** Structured by modality, year, and venue
- **Survey anchor:** Peer-reviewed IEEE Reviews paper provides synthesis

### Limitations
- **Documentation-only:** No executable code or model weights
- **General medical focus:** Limited neuro-omics specific content
- **No unified schema:** Markdown tables, not structured YAML/JSON
- **Citation lag:** New models may take time to appear in catalog

## When to Use FMS-Medical

✅ **Use when:**
- Starting literature review on medical FMs
- Selecting baseline models for benchmarking
- Finding publicly available medical datasets
- Identifying integration patterns from related work
- Citing comprehensive medical FM surveys

⚠️ **Not a substitute for:**
- Model implementation code (links to external repos)
- Pretrained model weights (links to paper/HuggingFace)
- Executable benchmarking pipelines (reference only)

⚠️ **Complement with:**
- **Papers With Code:** For benchmark leaderboards
- **HuggingFace Model Hub:** For pretrained weights
- **Model-specific repos:** For implementation details

## Reference Materials

**Primary sources:**
- **Catalog:** [FMS-Medical GitHub](https://github.com/YutingHe-list/Awesome-Foundation-Models-for-Advancing-Healthcare)
- **Survey paper:** [IEEE Reviews in Biomedical Engineering (2024)](https://ieeexplore.ieee.org/document/10750441)
- **arXiv version:** [Foundation Models for Healthcare (2024)](https://arxiv.org/abs/2404.03264)
- **Code walkthrough:** [FMS-Medical walkthrough](../../code_walkthroughs/fms_medical_walkthrough.md)
- **KB dataset card:** [kb/datasets/fms_medical_catalog.yaml](https://github.com/allison-eunse/neuro-omics-kb/blob/main/kb/datasets/fms_medical_catalog.yaml)

**Integration recipes:**
- [Integration Strategy](../../integration/integration_strategy.md) — Dataset selection
- [KB Overview](../../guide/kb_overview.md) — Catalog integration patterns

**Source repository:**
- **Local:** `external_repos/fms-medical/`
- **GitHub:** [YutingHe-list/Awesome-Foundation-Models-for-Advancing-Healthcare](https://github.com/YutingHe-list/Awesome-Foundation-Models-for-Advancing-Healthcare)

## Next Steps in Our Pipeline

1. **Systematic extraction:** Parse FMS-Medical tables to populate KB model/dataset cards
2. **Gap analysis:** Identify medical FM capabilities missing from neuro-omics KB
3. **Benchmark selection:** Choose medical FM baselines for comparison experiments
4. **Dataset discovery:** Find publicly available datasets for cross-validation
5. **Literature tracking:** Monitor repository updates for new medical FM methods

## Engineering Notes

- FMS-Medical is **pure documentation**—no code dependencies
- **Bilingual PDFs** in `files/` directory useful for international teams
- **Citation metadata** in tables ready for automated KB card generation
- **Active maintenance**—check GitHub for recent updates before citing
- **Survey paper** provides narrative synthesis complementing tabular catalog


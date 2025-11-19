---
title: Me-LLaMA — Model Card
status: active
updated: 2025-11-19
---

# Me-LLaMA (Medical Large Language Model)

## Overview

**Type:** Medical Foundation Large Language Model  
**Architecture:** LLaMA-2 with continual pretraining + instruction tuning  
**Modality:** Text (biomedical literature + clinical notes)  
**Primary use:** Medical text understanding, generation, and clinical reasoning

## Purpose & Design Philosophy

Me-LLaMA is a family of open-source medical foundation LLMs (13B and 70B parameters) built by **continually pretraining LLaMA-2** on 129 billion tokens of biomedical literature and clinical notes, then instruction-tuning on 214k medical task examples. The model targets comprehensive medical text analysis across question answering, named entity recognition, relation extraction, classification, summarization, natural language inference, and complex clinical case reasoning.

**Key innovation:** Large-scale continual pretraining on diverse medical corpora (literature + clinical notes + general text) enables Me-LLaMA to match or exceed GPT-4 on several medical benchmarks while remaining fully open-source.

## Architecture Highlights

- **Backbone:** LLaMA-2 decoder-only transformers (13B and 70B parameters)
- **Continual pretraining:** 129B tokens from biomedical literature + clinical notes + general text
- **Instruction tuning:** 214k multi-task medical instructions covering 6+ task families
- **Model family:** Base models (Me-LLaMA-13B/70B) and chat models (Me-LLaMA-13B/70B-chat)
- **Evaluation:** 12 benchmarks + clinical case diagnosis vs open-source and commercial LLMs

## Integration Strategy

### For Neuro-Omics KB

Me-LLaMA provides **medical LLM integration patterns**:

**Key lessons for neuro-omics text integration:**
- **Continual pretraining:** How to inject domain knowledge into general LLMs
- **Clinical + literature mix:** Balance research articles with real-world clinical language
- **Instruction tuning:** Multi-task learning across diverse neuro-omics NLP tasks
- **Zero-shot transfer:** Applicable to new clinical scenarios without labeled data

**Potential adaptation for neuro-omics:**
```
General LLM (LLaMA-2) 
    ↓
Continual pretrain on:
    - Neuroscience literature (PubMed)
    - Genetics literature (dbGaP, ClinVar annotations)
    - Clinical neurology notes
    ↓
Instruction tune on:
    - Gene-disease QA
    - Brain phenotype description
    - Genetic counseling dialogs
    ↓
Neuro-Omics LLM
```

### For ARPA-H Brain-Omics Model (BOM)

Me-LLaMA demonstrates **LLM as semantic bridge**:

```
Gene embeddings   → |
                    | Feature extraction
Brain embeddings  → |     ↓
                    | Medical LLM (Me-LLaMA-style)
Clinical notes    → |     ↓
                    | Unified reasoning + report generation
```

**Transfer insights:**
- **Knowledge injection:** Add neuroscience + genetics knowledge to general LLMs
- **Clinical reasoning:** Complex case-based diagnosis applicable to neurological disorders
- **Multimodal bridge:** LLM connects structured embeddings (gene, brain) with unstructured text
- **Report generation:** Automate clinical summaries from multimodal neuro-omics inputs

## Embedding Extraction Workflow

If adapting Me-LLaMA for neuro-omics:

```bash
# 1. Collect neuroscience + genetics text corpora
#    - PubMed Central (neuroscience + genetics papers)
#    - Clinical neurology notes (de-identified)
#    - Genetic variant annotations (ClinVar, dbGaP)
# 2. Continual pretrain LLaMA-2 on domain corpora
# 3. Curate instruction-tuning dataset
#    - Gene-disease QA
#    - Brain phenotype classification
#    - Clinical case reasoning
# 4. Instruction tune and evaluate on medical NLP tasks
# 5. Use as semantic bridge for gene-brain-text integration
```

**For neuro-omics KB:**
- **Text encoder:** Extract embeddings from Me-LLaMA for clinical notes
- **Semantic alignment:** Align gene/brain embeddings with text embeddings
- **Report generation:** Generate clinical summaries from gene-brain predictions

## Strengths & Limitations

### Strengths
- **Large-scale continual pretraining:** 129B tokens from diverse medical sources
- **Open-source:** Fully released models, data, and code
- **Comprehensive evaluation:** 12 benchmarks + clinical case diagnosis
- **Competitive performance:** Matches or exceeds GPT-4 on several medical tasks

### Limitations
- **Text-only:** No vision or multimodal capabilities (unlike M3FM, TITAN)
- **Compute intensive:** Training requires >100k GPU hours
- **Clinical validation:** Strong benchmarks but limited real-world deployment data
- **Data access:** Clinical notes require institutional access and IRB approval

## When to Use Me-LLaMA

✅ **Use as reference when:**
- Building neuro-omics text understanding models
- Designing continual pretraining strategies for domain LLMs
- Creating instruction-tuning datasets for medical NLP
- Integrating LLMs as semantic bridges in multimodal systems

⚠️ **Do not use directly for:**
- Multimodal gene-brain integration (text-only model)
- Vision-language tasks (no image encoder)
- Production clinical diagnosis (requires validation)

⚠️ **Consider alternatives:**
- **M3FM:** For medical imaging + text with CLIP-style alignment
- **BAGEL:** For unified understanding + generation with vision
- **TITAN:** For whole-slide pathology with vision-language alignment

## Reference Materials

**Primary sources:**
- **Paper:** [Me-LLaMA (2024)](../../generated/kb_curated/papers-md/me_llama_2024.md) — Preprint
- **Code walkthrough:** [Me-LLaMA walkthrough](../../code_walkthroughs/melamma_walkthrough.md)
- **YAML card:** [kb/model_cards/me_llama.yaml](https://github.com/allison-eunse/neuro-omics-kb/blob/main/kb/model_cards/me_llama.yaml)

**Integration recipes:**
- [Multimodal Architectures](../../integration/multimodal_architectures.md)
- [Design Patterns](../../integration/design_patterns.md) — LLM as semantic bridge
- [Integration Strategy](../../integration/integration_strategy.md)

**Source repository:**
- **Local:** `external_repos/me-lamma/`
- **GitHub:** [BIDS-Xu-Lab/Me-LLaMA](https://github.com/BIDS-Xu-Lab/Me-LLaMA)

## Next Steps in Our Pipeline

1. **Domain corpus curation:** Collect neuroscience + genetics literature for continual pretraining
2. **Instruction dataset design:** Create neuro-omics QA, NER, RE task datasets
3. **Continual pretraining:** Adapt LLaMA-2 to neuroscience + genetics domains
4. **Semantic bridge integration:** Connect gene/brain embeddings with LLM text space
5. **Clinical report generation:** Automate neuroimaging + genetics summaries

## Engineering Notes

- Me-LLaMA's **mixture weighting** (general + biomedical + clinical) preserves broad language competence
- **Instruction tuning** on 214k examples spans 6 task families—applicable to neuro-omics NLP
- **Clinical case reasoning** evaluation critical for validating complex diagnostic capabilities
- **Open-source release** includes base and chat models—study instruction tuning strategies


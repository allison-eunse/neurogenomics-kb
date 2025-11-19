# FMS-Medical Curation Walkthrough

> **KB references:** Survey digest (pending) · [Integration strategy](../integration/integration_strategy.md) · [KB overview](../guide/kb_overview.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
`external_repos/fms-medical` is an “awesome list” style knowledge base that tracks foundation-model research across healthcare modalities—language (LFM), vision (VFM), bioinformatics (BFM), and multimodal (MFM)—plus dataset catalogs, tutorials, and bilingual survey PDFs. The maintainers keep the README current with publication news (e.g., IEEE Reviews acceptance) and provide both English and Chinese summaries (`files/HFM_Chinese.pdf`), making it a convenient seed for KB model/dataset cards.^[```1:38:/Users/allison/Projects/neuro-omics-kb/external_repos/fms-medical/README.md```]

## At-a-Glance
| Architecture | Params | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| Markdown knowledge graph: NEWS → Survey references → modality-specific method lists + dataset tables.^[```39:399:/Users/allison/Projects/neuro-omics-kb/external_repos/fms-medical/README.md```] | Organized by year (2020–2024) and modality; each entry stores venue, short description, and code/paper links—ready for transformation into KB YAML cards. | Highlights IEEE Reviews 2024 survey + arXiv companions; hosts bilingual PDF in `files/` for regional teams.^[```5:38:/Users/allison/Projects/neuro-omics-kb/external_repos/fms-medical/README.md```] | Pure Markdown + embedded images; contributions occur via pull requests (no runtime code). | Quick lookup for LFM/VFM/BFM/MFM models, dataset tables (text, imaging, omics, multimodal), lectures, blogs, and related awesome lists.^[```39:400:/Users/allison/Projects/neuro-omics-kb/external_repos/fms-medical/README.md```] | [github.com/YutingHe-list/Awesome-Foundation-Models-for-Advancing-Healthcare](https://github.com/YutingHe-list/Awesome-Foundation-Models-for-Advancing-Healthcare) |

### Repository Notes
- **Documentation-only.** No Python modules or requirements—syncing the README (and optional PDF) is sufficient to integrate the content into KB templates.
- **Bilingual assets.** `files/HFM_Chinese.pdf` mirrors the survey for Mandarin readers; cite it when translating KB summaries.^[```17:34:/Users/allison/Projects/neuro-omics-kb/external_repos/fms-medical/README.md```]
- **Citation-first.** Each entry lists papers/code, so KB automation scripts can parse the tables to populate `kb/model_cards`, `kb/datasets`, or `docs/generated` references.

## Key Components

### Survey Metadata & NEWS Banner
Top-of-file announcements capture publication milestones, acceptance venues, and contact information. These lines can drive KB changelogs or curated timelines.

```5:34:/Users/allison/Projects/neuro-omics-kb/external_repos/fms-medical/README.md
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[NEWS.20241115] **Our survey [paper](https://ieeexplore.ieee.org/document/10750441) has been accepted by IEEE Reviews in Biomedical Engineering (IF: 17.2).**

[NEWS.20240405] **The related survey [paper](https://arxiv.org/abs/2404.03264) has been released.**

[NOTE] **If you have any questions, please don't hesitate to [contact us](mailto:yuting.he4@case.edu).** 
```

### Modality Method Registries (LFM/VFM/BFM/MFM)
Each section groups methods by year, venue, and modality. Capturing these rows lets the KB auto-generate candidate model cards or integration experiments.

```82:210:/Users/allison/Projects/neuro-omics-kb/external_repos/fms-medical/README.md
## LFM methods
**2024**
- [AAAI] Zhongjing: Enhancing the chinese medical capabilities of large language model through expert feedback and realworld multi-turn dialogue. [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29907) [[Code]](https://github.com/SupritYoung/Zhongjing)
- [NeurIPS] MDAgents: An adaptive collaboration of LLMs for medical decision-making. [[Paper]](https://arxiv.org/abs/2404.15155) [[Code]](https://github.com/mitmedialab/MDAgents)
...
## VFM methods
**2024**
- [arXiv] USFM: A universal ultrasound foundation model generalized to tasks and organs towards label efficient image analysis. [[paper]](https://arxiv.org/html/2401.00153v2) 
```

### Dataset Catalogs
Separate tables detail datasets per modality (text, imaging, multimodal). These rows map neatly onto `kb/datasets/*.yaml` and help ensure coverage across integrative experiments.

```339:399:/Users/allison/Projects/neuro-omics-kb/external_repos/fms-medical/README.md
## Datasets
### LFM datasets
|                           Dataset  Name                               | Text Types  |            Scale           |    Task    |                       Link                             |
| :-------------------------------------------------------------------: | :-------: | :------------------------: | :--------: | :----------------------------------------------------: |
|[PubMed](https://pubmed.ncbi.nlm.nih.gov/download/) | Literature | 18B tokens |  Language modeling |[*](https://pubmed.ncbi.nlm.nih.gov/download/)|
|[MedC-I](https://arxiv.org/abs/2304.14454)| Literature | 79.2B tokens |  Dialogue |[*](https://huggingface.co/datasets/axiong/pmc_llama_instructions)|
...
|[CMeKG-8K](https://www.mdpi.com/2078-2489/11/4/186)| Dialogue | 8K instances |  Dialogue |[*](https://github.com/WENGSYX/CMKG)|
```

### Other Resources (Lectures/Blogs/Awesome lists)
The README also aggregates tutorials, blogs, and related awesome repositories under “Other Resources,” which can seed KB “Further reading” sections or onboarding material.

```53:74:/Users/allison/Projects/neuro-omics-kb/external_repos/fms-medical/README.md
- [Other Resources](#other-resources)
  - [Lectures and tutorials](#lectures-and-tutorials)
  - [Blogs](#blogs)
  - [Related awesome repositories](#related-awesome-repositories)
```

## Integration Hooks (Curation ↔ KB)
- **Automate card creation.** Parse the Markdown tables (e.g., via pandoc or custom scripts) to prefill `kb/model_cards` with metadata (venue, year, links), ensuring coverage parity across modalities.
- **Align dataset registries.** Map the README dataset entries to `kb/datasets/*.yaml` and tag each with modality + task so integration plans can quickly reference scale/availability.
- **Leverage bilingual PDFs.** When publishing KB summaries for non-English audiences, cite `files/HFM_Chinese.pdf` to keep translations aligned with the upstream survey and avoid redundant localization work.


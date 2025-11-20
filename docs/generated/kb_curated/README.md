---
title: AI Summaries and KB
status: draft
updated: 2025-11-20
---

# AI Summaries → KB Cards

Use the PDF⇄MD repo (`~/Projects/pdf<->md;ai-summaries/`) for extraction + summarization, then land the curated assets back in this KB using the three-layer structure below.

## Storage layout

```
docs/generated/kb_curated/
├── datasets/                 ← dataset summary templates / intermediate assets
├── papers-pdf/               ← Layer 2 source PDFs copied in for reference
├── papers-md/                ← Layer 2 technical notebooks (medium MD)
└── templates/                ← helper templates (shared)
```

- **Layer 1 (Citation & context, short)** → `kb/paper_cards/<slug>_YYYY.yaml`
- **Layer 2 (Technical notebook, medium)** → `docs/generated/kb_curated/papers-md/<slug>.md`
- **Layer 2 assets (PDFs)** → `docs/generated/kb_curated/papers-pdf/<slug>.pdf`
- **Layer 3 (Hooks into KB)** → bottom section of each Layer 2 MD + YAML registries in `kb/integration_cards/*.yaml` or narrative integration cards in `docs/models/integrations/*.md`

## Workflow

1. **Convert & summarize outside the KB**
   - `pdf_to_markdown.py input/*.pdf → build/<slug>.md`
   - `summary_generator.py build/<slug>.md → build/<slug>_summary.md`
2. **Copy curated assets into this repo**
   - `cp ~/Projects/pdf<->md;ai-summaries/input/<slug>.pdf docs/generated/kb_curated/papers-pdf/`
   - `cp ~/Projects/pdf<->md;ai-summaries/build/<slug>_summary.md docs/generated/kb_curated/papers-md/<slug>.md`
3. **Fill the Layer 2 template**
   - Follow `docs/generated/kb_curated/papers-md/template.md` for sections (problem/tasks, datasets, methods, results, limitations, hooks).
4. **Update the Layer 1 YAML card**
   - Keep summaries to 3–5 sentences, set `summary_md_path` and `local_pdf_path`, and list every KB doc/config the paper informs.
5. **Regenerate/extend guidance cards if needed**
   - For integration principles (e.g., Li 2022, Waqas 2024) update the YAML registries in `kb/integration_cards/*.yaml` and narrative cards under `docs/models/integrations/`.

## Why this matters

- Keeps PDFs + long notes co-located with the KB for future RAG/indexing.
- Ensures YAML cards stay concise and machine-friendly while MDs hold the richer narrative + explicit “hooks into our KB”.
- Makes it trivial to trace a config (e.g., `configs/experiments/02_prediction_baselines.yaml`) back to the exact papers + notes that motivated each requirement.

## Original source index

All PDFs/MDs under `docs/generated/kb_curated/` mirror the outputs generated in `../pdf<->md;ai-summaries/outputs`. Use the table below to jump from a KB summary back to the published source.

| Reference | KB assets | Original publication |
| --- | --- | --- |
| TabPFN (Nature 2025) | — (summary pending) | [Nature article](https://www.nature.com/articles/s41586-024-08328-6) · [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) |
| Brain Harmony (2025) | [PDF](papers-pdf/brainharmony_2025.pdf) · [MD](papers-md/brainharmony_2025.md) | [arXiv:2509.24693](https://arxiv.org/abs/2509.24693) |
| Brain-JEPA (2024) | [PDF](papers-pdf/brainjepa_2024.pdf) · [MD](papers-md/brainjepa_2024.md) | [arXiv:2409.19407](https://arxiv.org/abs/2409.19407) |
| BrainLM (2024) | [PDF](papers-pdf/brainlm_2024.pdf) · [MD](papers-md/brainlm_2024.md) | [OpenReview RwI7ZEfR27](https://openreview.net/forum?id=RwI7ZEfR27) |
| BrainMT (2025) | [PDF](papers-pdf/brainmt_2025.pdf) · [MD](papers-md/brainmt_2025.md) | [LNCS DOI 10.1007/978-3-032-05162-2_15](https://dl.acm.org/doi/10.1007/978-3-032-05162-2_15) |
| Caduceus (2024) | [PDF](papers-pdf/caduceus_2024.pdf) · [MD](papers-md/caduceus_2024.md) | [arXiv:2403.03234](https://arxiv.org/abs/2403.03234) |
| Evo 2 (2025) | [PDF](papers-pdf/evo2_2024.pdf) · [MD](papers-md/evo2_2024.md) | [bioRxiv 10.1101/2025.02.18.638918](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1) |
| GENERaTOR (2024) | [PDF](papers-pdf/generator_2024.pdf) · [MD](papers-md/generator_2024.md) | [arXiv:2502.07272](https://arxiv.org/abs/2502.07272) |
| Me-LLaMA medical LLMs (2025) | [PDF](papers-pdf/me_llama_2024.pdf) · [MD](papers-md/me_llama_2024.md) | [npj Digital Medicine 8, 141 (2025)](https://www.nature.com/articles/s41746-025-01533-1) |
| Multimodal LLMs in radiology (KJR 2025) | [PDF](papers-pdf/mm_llm_imaging_2025.pdf) · [MD](papers-md/mm_llm_imaging_2025.md) | [Korean Journal of Radiology, DOI 10.3348/kjr.2025.0599](https://kjronline.org/DOIx.php?id=10.3348/kjr.2025.0599) |
| Medical MMFMs in diagnosis & treatment (2025) | [PDF](papers-pdf/mmfm_2025.pdf) · [MD](papers-md/mmfm_2025.md) | [Artificial Intelligence in Medicine, ISSN S0933365725002003](https://www.sciencedirect.com/science/article/pii/S0933365725002003) |
| Foundation models for advancing healthcare (2024) | [PDF](papers-pdf/fm_general_2024.pdf) · [MD](papers-md/fm_general_2024.md) | [npj Digital Medicine (2024)](https://www.nature.com/articles/s41746-024-01339-7) |
| TITAN pathology FM (2025) | [PDF](papers-pdf/titan_2025.pdf) · [MD](papers-md/titan_2025.md) | [Nature Medicine 31, 1–13 (2025)](https://www.nature.com/articles/s41591-025-03982-3) |
| Ensemble Integration (Li 2022) | [PDF](papers-pdf/ensemble_integration_li2022.pdf) · [MD](papers-md/ensemble_integration_li2022.md) | Bioinformatics Advances 2022 (late-integration EI framework) |
| Oncology multimodal review (2024) | [PDF](papers-pdf/oncology_multimodal_waqas2024.pdf) · [MD](papers-md/oncology_multimodal_waqas2024.md) | [PubMed 39118787](https://pubmed.ncbi.nlm.nih.gov/39118787/) |
| GWAS in diverse populations (2019) | [PDF](papers-pdf/gwas_diverse_populations.pdf) · [MD](papers-md/gwas_diverse_populations.md) | [PubMed 36158455](https://pubmed.ncbi.nlm.nih.gov/36158455/) |
| PRS guide (2019) | [PDF](papers-pdf/prs_guide.pdf) · [MD](papers-md/prs_guide.md) | [PubMed 31607513](https://pubmed.ncbi.nlm.nih.gov/31607513/) |
| Yoon et al. BIOKDD'25 (MDD embeddings) | [PDF](papers-pdf/yoon_biokdd2025.pdf) · [MD](papers-md/yoon_biokdd2025.md) | [bioRxiv 10.1101/2025.02.18.638918](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full.pdf) |
| Mixture-of-Transformers (2025) | [PDF](papers-pdf/mot_2025.pdf) · [MD](papers-md/mot_2025.md) | [arXiv:2411.04996](https://arxiv.org/abs/2411.04996) |
| BAGEL unified multimodal FM (2025) | [PDF](papers-pdf/bagel_2025.pdf) · [MD](papers-md/bagel_2025.md) | [arXiv:2505.14683](https://arxiv.org/abs/2505.14683) |

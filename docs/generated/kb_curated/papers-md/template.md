---
title: Paper Slug — Descriptive Title
status: draft
updated: 2025-11-18
tags: [paper-notes]
---

# Paper Title (Year) — Short Tagline

## 1. Problem & Tasks

- What is being predicted/estimated?
- Benchmarks/tasks, evaluation settings (classification, regression, survival, etc.).
- How this connects to genetics × MRI integration (if applicable).

## 2. Datasets

- Cohorts, sample sizes, populations, modalities.
- Preprocessing specifics (QC, parcellations, sequencing protocols, etc.).
- Splits: train/val/test definitions, CV type, site-aware grouping, seeds.

## 3. Model / Method Details

### 3.1 Architecture / Method

- Backbone, context length, tokenization, RC-equivariance, hub tokens, etc.
- Input/output formats, objectives, losses, notable tricks.

### 3.2 Confound Handling & Evaluation Discipline

- Residualization, stratification, harmonization, missingness handling.
- Metrics (AUROC/AUPRC/Brier/calibration), statistical tests (DeLong, bootstrap, permutations, FDR).

## 4. Results & Tables

- Key numbers vs baselines (include effect sizes and uncertainty if reported).
- Ablation results or thresholds that matter for reuse.

## 5. Limitations & Cautions

- Author-listed limitations plus any reuse caveats you notice.
- Cohort bias, preprocessing quirks, compute constraints, licensing limits.

---

## 6. Hooks into Neuro-Omics KB

**Relevant KB assets**

- `kb/paper_cards/<slug>_YYYY.yaml`
- `kb/model_cards/...` (if applicable)
- `kb/datasets/...` (if applicable)
- `docs/generated/kb_curated/integration_cards/<related>.md`

**Configs / recipes informed**

- `configs/experiments/...`
- `docs/integration/analysis_recipes/...`
- `docs/integration/integration_strategy.md`

**Concrete guidance for our project**

- Fusion pattern / modality sequencing choices this paper justifies.
- Specific covariates, statistical tests, or dataset fields we adopted.
- Any LOGO/PRS/GWAS/CCA protocol parameters we ported over.

> Copy this file to `docs/generated/kb_curated/papers-md/<slug>.md` and fill in the sections. Keep Layer 2 MDs canonical, so future tooling (RAG, dashboards) can scan them systematically.


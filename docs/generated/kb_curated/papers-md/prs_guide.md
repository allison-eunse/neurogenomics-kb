---
title: PRS Guide — Choi & O'Reilly 2019
status: draft
updated: 2025-11-18
tags: [paper-notes, genetics, prs]
---

# A Guide to Performing Polygenic Risk Score Analyses (2019)

## 1. Problem & Tasks

- Practical tutorial for constructing, validating, and reporting polygenic risk scores (PRS).
- Relevant for defining covariate controls and baseline genetics features when combining with MRI.

## 2. Datasets

- Examples drawn from UK Biobank and PGC cohorts; outlines required inputs: GWAS summary stats, target genotypes.
- Emphasizes ancestry-matched target sets and QC thresholds (MAF, INFO, Hardy–Weinberg).

## 3. Model / Method Details

### 3.1 Pipeline

- QC target genotypes → LD clumping/thresholding or more advanced methods (LDPred, PRS-CS).
- Calculate PRS per subject, standardize, residualize vs covariates.
- Evaluate predictive performance with logistic regression or linear regression depending on phenotype.

### 3.2 Confound Handling & Evaluation Discipline

- Always include ancestry PCs (≥10), sex, age, batch in regression models.
- Use nested CV when tuning PRS hyperparameters to avoid overfitting.
- Report incremental R² / pseudo-R² and calibration metrics.

## 4. Results & Tables

- Provides illustrative tables showing how PRS performance varies with clumping parameters and ancestry mismatches (drops of 30–50% accuracy when mismatched).

## 5. Limitations & Cautions

- Focused on SNP array + European ancestry; does not directly cover WES/WGS nuance.
- Methods (clumping + thresholding) may be superseded by PRS-CS, but workflow guidance still valid.

---

## 6. Hooks into Neuro-Omics KB

**Relevant KB assets**

- `kb/paper_cards/prs_guide.yaml`
- `kb/datasets/ukb_manifest_stub.yaml` (covariate list, PC requirements).

**Configs / recipes informed**

- Future baseline where PRS is an additional modality alongside gene embeddings and MRI.
- Provides justification for including PCs, age, sex, site covariates in `configs/experiments/*`.

**Concrete guidance for our project**

- Whenever PRS features are added, include at least 10 ancestry PCs + batch covariates in modeling (per guide).
- Use nested CV for PRS hyperparameters; avoid leaking test folds.
- Document QC thresholds and summary-stat sources in dataset cards, referencing this guide.


---
title: Peterson 2019 — GWAS in Diverse Populations
status: draft
updated: 2025-11-18
tags: [paper-notes, genetics, gwas]
---

# Genome-wide Association Studies in Ancestrally Diverse Populations (2019)

## 1. Problem & Tasks

- Review methodological pitfalls and recommendations for conducting GWAS beyond European cohorts.
- Critical for how we handle ancestry PCs and covariates when integrating genetics with MRI.

## 2. Datasets

- Summarizes lessons from PAGE, HCHS/SOL, UKB, biobanks in East Asia and Africa.
- Emphasizes differences in sample sizes and LD structure across populations.

## 3. Model / Method Details

### 3.1 Key Recommendations

- Use linear mixed models or ancestry-informed stratification to control population structure.
- Include local ancestry estimates when available; otherwise rely on global PCs.
- Validate PRS / embeddings separately per ancestry group; avoid pooling unless harmonized.

### 3.2 Confound Handling & Evaluation Discipline

- Check for residual stratification using QQ plots and genomic control.
- Report ancestry composition and site distribution with each GWAS/PRS result.
- Use jackknife/bootstraps for uncertainty when sample sizes are small per subgroup.

## 4. Results & Tables

- Quantifies performance drop (R² decrease up to 5×) when PRS trained in Europeans applied to African ancestry cohorts.
- Provides tables linking LD differences to effect estimation bias.

## 5. Limitations & Cautions

- 2019 state of the field; newer multi-ancestry methods exist but core cautions remain.
- Does not cover foundation models directly; we must adapt recommendations to embeddings.

---

## 6. Hooks into Neuro-Omics KB

**Relevant KB assets**

- `kb/paper_cards/gwas_diverse_populations.yaml`
- `kb/datasets/ukb_manifest_stub.yaml` (records ancestry composition, PCs).

**Configs / recipes informed**

- Covariate list (age/sex/site/PCs) in `configs/experiments/01_cca_gene_smri.yaml` and `02_prediction_baselines.yaml`.
- Future documentation for PRS/embedding fairness analyses.

**Concrete guidance for our project**

- Always log ancestry distribution and include PCs in dataset cards; cite this paper when explaining why.
- If/when we add non-European cohorts, run stratified evaluations instead of assuming global models transfer.
- Keep hooks for local ancestry or site-specific PCs should we extend beyond UKB Europeans.


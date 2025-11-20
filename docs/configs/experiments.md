---
title: Experiment Configs
status: ready
updated: 2025-11-20
---

# ⚙️ Experiment Configs

Ready-to-use YAML templates for reproducible gene-brain-behavior analysis. Each config specifies embedding strategies, covariates, cross-validation schemes, and validation protocols.

---

## Available Configs

### 01_cca_gene_smri.yaml
**Purpose:** CCA + permutation baseline for gene-brain association discovery

**Key Features:**
- Cross-modal null distributions via within-fold permutation
- Canonical correlation coefficients (ρ₁–ρ₃) with p-values
- Site-/scanner-aware stratified CV
- Loadings/feature contributions for interpretation

**When to use:**
- Exploratory structure discovery before prediction tasks
- Testing whether gene-brain associations exist above chance
- Lightweight statistical checks that respect confounds

**References:**
- [Analysis recipe](../integration/analysis_recipes/cca_permutation.md)
- [Integration strategy](../integration/integration_strategy.md)

---

### 02_prediction_baselines.yaml
**Purpose:** Gene vs Brain vs Fusion prediction comparison

**Key Features:**
- Per-modality baselines (Gene-only, Brain-only)
- Late fusion (concatenated embeddings)
- LR + GBDT (LightGBM/CatBoost) for each condition
- DeLong tests to compare AUC across models
- Stratified K-fold with site/ancestry controls

**When to use:**
- Quantifying whether fusion beats single-modality models
- Comparing embedding strategies (e.g., Caduceus vs DNABERT-2)
- Establishing baselines before escalating to contrastive or early fusion

**References:**
- [Prediction baselines recipe](../integration/analysis_recipes/prediction_baselines.md)
- [Late fusion rationale (Li 2022)](../models/integrations/ensemble_integration.md)

---

### 03_logo_gene_attribution.yaml
**Purpose:** Leave-One-Gene-Out (LOGO) attribution protocol

**Key Features:**
- ΔAUC computation: `AUC(all genes) - AUC(all genes except gene_i)`
- Per-gene importance ranking
- Stratified CV to avoid overfitting to site/ancestry
- Optional permutation test for significance

**When to use:**
- Identifying which genes drive prediction performance
- Validating biological hypotheses (e.g., "Is SOD2 critical for MDD prediction?")
- Prioritizing candidates for functional follow-up

**References:**
- [Yoon BIOKDD 2025](../generated/kb_curated/papers-md/yoon_biokdd2025.md) — LOGO + gene embeddings
- [Genomics modality features](../integration/modality_features/genomics.md)

---

### Development Stubs

**cha_dev_smri_pca_dimsearch_template.yaml**  
Template for CHA Hospital developmental cohort: PCA dimensionality search for sMRI embeddings.

**dev_01_brain_only_baseline.yaml**  
Brain-only baseline for developmental trajectory modeling.

**dev_02_gene_brain_behaviour.yaml**  
Gene-brain-behavior integration for longitudinal developmental data.

**ukb_smri_pca_dimsearch_template.yaml**  
UK Biobank sMRI PCA dimension search template.

---

## Usage

### 1. Clone a template

```bash
cp configs/experiments/02_prediction_baselines.yaml my_experiment.yaml
```

### 2. Customize for your cohort

Edit the YAML to specify:
- **Embedding strategies** (references [`kb/integration_cards/embedding_strategies.yaml`](https://github.com/allison-eunse/neuro-omics-kb/blob/main/kb/integration_cards/embedding_strategies.yaml))
- **Harmonization methods** (references [`kb/integration_cards/harmonization_methods.yaml`](https://github.com/allison-eunse/neuro-omics-kb/blob/main/kb/integration_cards/harmonization_methods.yaml))
- **Covariates** (age, sex, site, motion, genetic PCs, etc.)
- **CV scheme** (K-fold, stratified groups, leave-site-out)
- **Metrics** (AUC, accuracy, calibration, DeLong tests)

### 3. Validate before running

```bash
python scripts/manage_kb.py validate experiments --config my_experiment.yaml
```

### 4. Track provenance

Each config embeds:
- Embedding strategy IDs (e.g., `smri_free_surfer_pca512_v1`)
- Harmonization method IDs (e.g., `combat_harmonization`)
- Preprocessing pipeline IDs (e.g., `fmriprep_standard`)

This ensures results are traceable back to exact methods and can be reproduced.

---

## Config Structure

All experiment configs follow this schema:

```yaml
experiment:
  name: "descriptive_name"
  objective: "prediction | association | attribution"
  
data:
  cohort: "ukb | cha_dev | hcp"
  modalities:
    - genetics
    - brain_smri
    - brain_fmri
  embedding_strategies:
    genetics: "caduceus_rc_avg_v1"
    brain_smri: "smri_free_surfer_pca512_v1"
  
covariates:
  - age
  - sex
  - site
  - motion_fd  ← fMRI only
  - genetic_pcs  ← first 10 PCs
  
validation:
  cv_scheme: "stratified_kfold"
  k_folds: 5
  stratify_by: ["site", "ancestry"]
  metrics:
    - auc
    - accuracy
    - calibration
  
baselines:
  - genetics_only
  - brain_only
  - late_fusion
  
statistical_tests:
  - delong_test  ← AUC comparison
  - permutation_test  ← for CCA
```

---

## Best Practices

### ✅ Always residualize confounds
Before fusion, remove age, sex, site, motion (fMRI), and genetic PCs from both modalities.

### ✅ Use site-aware CV
Stratify folds by site/scanner to avoid spurious site-specific signals.

### ✅ Report calibration
AUC alone can be misleading; include calibration curves or Brier scores.

### ✅ Track embedding versions
Use explicit IDs (e.g., `caduceus_rc_avg_v1`) so results are reproducible even if embedding methods evolve.

### ✅ Log negative results
If fusion doesn't beat single-modality baselines, document it—that's valuable for the field.

---

## Integration with KB Assets

Each config references:

- **[Embedding strategies](https://github.com/allison-eunse/neuro-omics-kb/blob/main/kb/integration_cards/embedding_strategies.yaml)** — How to extract gene/brain features
- **[Harmonization methods](https://github.com/allison-eunse/neuro-omics-kb/blob/main/kb/integration_cards/harmonization_methods.yaml)** — Site/batch correction protocols
- **[Preprocessing pipelines](https://github.com/allison-eunse/neuro-omics-kb/blob/main/kb/integration_cards/rsfmri_preprocessing_pipelines.yaml)** — fMRI processing steps
- **[Model cards](https://github.com/allison-eunse/neuro-omics-kb/tree/main/kb/model_cards)** — FM architecture details
- **[Dataset cards](https://github.com/allison-eunse/neuro-omics-kb/tree/main/kb/datasets)** — Sample sizes, QC thresholds

This ensures every experiment is grounded in validated methods and traceable metadata.

---

## Next Steps

1. **Start with CCA + permutation** ([01_cca_gene_smri.yaml](https://github.com/allison-eunse/neuro-omics-kb/blob/main/configs/experiments/01_cca_gene_smri.yaml)) to check if gene-brain structure exists
2. **Run prediction baselines** ([02_prediction_baselines.yaml](https://github.com/allison-eunse/neuro-omics-kb/blob/main/configs/experiments/02_prediction_baselines.yaml)) to quantify fusion gains
3. **Attribute genes** ([03_logo_gene_attribution.yaml](https://github.com/allison-eunse/neuro-omics-kb/blob/main/configs/experiments/03_logo_gene_attribution.yaml)) to identify key drivers

**Need help?** Check the [Integration Strategy](../integration/integration_strategy.md) or [Analysis Recipes](../integration/analysis_recipes/cca_permutation.md)

---

**Browse configs on GitHub:** [configs/experiments/](https://github.com/allison-eunse/neuro-omics-kb/tree/main/configs/experiments)


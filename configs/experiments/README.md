# Experiment Configs

Ready-to-run YAML templates for Nov 26 baseline experiments. Each config specifies:
- Datasets (with references to KB dataset cards)
- Feature preparation (models, preprocessing, covariates)
- Cross-validation strategy (folds, seeds, grouping)
- Analysis methods (CCA, prediction, LOGO)
- Evaluation metrics and statistical tests
- Output paths and visualization specs

## Templates

### 01_cca_gene_smri.yaml
**Objective:** Test cross-modal structure between gene embeddings and sMRI ROIs

**Key specs:**
- Modalities: Caduceus gene embeddings (38 MDD genes) × FreeSurfer sMRI ROIs
- Preprocessing: z-score, residualize covariates, PCA to 512-D per modality
- Analysis: CCA with 1,000 permutations
- Metrics: ρ1–ρ3 with permutation p-values; CCA loadings
- Deadline: Nov 26

**Usage:**
```bash
# 1. Fill in dataset paths after data inventory
# 2. Verify gene set (38 MDD genes) is available
# 3. Run preprocessing + CCA script (to be implemented in separate analysis repo)
# 4. Log results to results/2025-11-26_cca_gene_smri/
```

**Expected outcome:**
- If ρ1 significant (p < 0.05): proceed to prediction baselines
- Loadings guide gene/ROI interpretation

---

### 02_prediction_baselines.yaml
**Objective:** Compare Gene vs sMRI vs Fusion classifiers for MDD prediction

**Key specs:**
- Models: Logistic Regression, LightGBM, CatBoost
- Baselines: Gene only (512-D), sMRI only (512-D), Fusion (1024-D concatenated)
- Same CV folds as CCA (seed=42, critical for fair comparison)
- Metrics: AUROC, AUPRC ± SD; DeLong/bootstrap for Fusion vs single-modality
- Deadline: Nov 26

**Usage:**
```bash
# 1. Reuse exact preprocessing from 01_cca_gene_smri.yaml
# 2. Train LR + GBDT on each baseline
# 3. Save held-out predictions for DeLong/bootstrap tests
# 4. Log results to results/2025-11-26_prediction_baselines/
```

**Expected outcome:**
- If Fusion > max(Gene, sMRI) with p < 0.05: strong evidence for integration value
- Feature importance (LR coefficients, SHAP) guides interpretation

---

### 03_logo_gene_attribution.yaml
**Objective:** Identify most predictive genes using Leave-One-Gene-Out (LOGO)

**Key specs:**
- Protocol: Nested CV to avoid bias; ΔAUC per gene
- Statistical test: Wilcoxon signed-rank across folds + FDR correction
- Expected signal: SOD2 (per Yoon et al. BIOKDD'25)
- Deadline: Nov 26

**Usage:**
```bash
# 1. For each gene: train on 37 genes, compute ΔAUC
# 2. Wilcoxon test across 5 folds per gene
# 3. Apply Benjamini-Hochberg FDR
# 4. Log results to results/2025-11-26_logo_attribution/
```

**Expected outcome:**
- SOD2 should rank top (sanity check)
- Genes with q < 0.05 are most predictive for MDD
- Cross-reference with CCA loadings for consistency

---

## Workflow

1. **Read paper card** for context (e.g., `kb/paper_cards/yoon_biokdd2025.yaml`)
2. **Check code walkthrough** for implementation details (e.g., `docs/code_walkthroughs/caduceus_walkthrough.md`)
3. **Clone config template** and fill dataset paths
4. **Run preprocessing + analysis** (scripts in separate analysis repo)
5. **Log outputs** back to `results/YYYY-MM-DD_experiment_name/`
6. **Update KB** with findings (add to `docs/decisions/` or integration cards)

## CV Discipline (Critical!)

All three experiments MUST use the **same CV folds** for fair comparison:
- Strategy: `StratifiedGroupKFold`
- k_folds: `5`
- group_by: `site` (prevent site leakage)
- stratify_by: `mdd_diagnosis`
- seed: `42`

This ensures CCA correlations, prediction AUCs, and LOGO attributions are computed on identical train/test splits.

## Related Docs

- [Integration baseline plan](../../docs/decisions/2025-11-integration-plan.md) — Why these experiments
- [Analysis recipes](../../docs/integration/analysis_recipes/) — Detailed protocols
- [Paper cards](../../kb/paper_cards/) — Source papers for methods
- [Dataset manifest stub](../../kb/datasets/ukb_manifest_stub.yaml) — Data inventory template

---

**Status:** Ready for Nov 26 baselines  
**Next step:** Fill dataset paths after data inventory meeting (Nov 18-21)


---
title: Integration Strategy
status: draft
updated: 2025-11-18
---

# 🔗 Integration Strategy

> **Late fusion first, then escalate to contrastive and unified architectures**

---

## 🎯 Overall Philosophy

**Late fusion / integration first, then scale if we see gains.**

---

## 🧬🧠 Why This Applies to Genes × Brain

- **Heterogeneous semantics:** Nucleic-acid sequence vs morphology/dynamics → maximize modality specificity before fusion
- **Different confounds:** Ancestry/batch vs site/motion/TR → deconfound independently

---

## 📊 Baselines

- Preprocess per modality
  - Z-score features.
  - Residualize against: age, sex, site/scanner, motion (FD), SES (if available), genetic PCs (PC1–PC10).
- Dimensionality
  - Project to 512 dims per modality (PCA or tiny MLP).
- CCA + permutation
  - CCA on train folds; 1,000 shuffles; report ρ1–ρ3 with p-values.
- Prediction
  - LR (balanced) and LightGBM/CatBoost on Gene, Brain, Fusion; same CV folds; AUROC/AUPRC; DeLong/bootstrap for Fusion vs single-modality.

---

## 📋 Embedding Strategy Registry

**Recipes live under `kb/integration_cards/embedding_strategies.yaml`**

Print them via:
```bash
python scripts/manage_kb.py ops strategy <id>
```

**Available strategies:**
- **sMRI (`smri_free_surfer_pca512_v1`).** FreeSurfer ROI table (~176 features) → fold-wise z-score → residualize age/sex/site/ICV → PCA→512. Future variants: FM encoders, diffusion MRI. Sources: `docs/integration/modality_features/smri.md`, FreeSurfer refs.
- **rs-fMRI baseline (`rsfmri_swift_segments_v1`).** SwiFT exports per 20-frame segment → mean pool tokens → run mean → subject mean → residualize age/sex/site/motion → PCA→512. Variants exist for BrainLM (`rsfmri_brainlm_segments_v1`), Brain-JEPA (`rsfmri_brainjepa_roi_v1`), and BrainMT (`rsfmri_brainmt_segments_v1`); each references the corresponding walkthrough/code.
- **Genetics (`genetics_gene_fm_pca512_v1`).** RC-averaged gene FMs (Caduceus/DNABERT-2/Evo2/HyenaDNA/GENERaTOR) → exon → gene pooling → concatenate curated gene set → residualize age/sex/ancestry PCs/batch → PCA→512. For non–RC-equivariant encoders, follow RCCR/RC-averaging guidance in the genomics modality spec.
- **Fusion (`fusion_concat_gene_brain_1024_v1`).** Concatenate the 512-D genetics vector with the chosen 512-D brain vector; z-score each block independently before concatenation.

**Experiments now reference these IDs** (see `configs/experiments/*.yaml`) to keep per-subject embeddings traceable.

---

## 🔧 Harmonization & Site Effects

**Cataloged in `kb/integration_cards/harmonization_methods.yaml`**

Query via:
```bash
python scripts/manage_kb.py ops harmonization <id>
```

**Available harmonization methods:**
- **Default (`none_baseline`).** Feature-level z-score + covariate residualization; always record site/motion covariates.
- **Statistical (`combat_smri`).** ROI-wise ComBat before PCA for sMRI (Fortin et al., 2018). Run the `02_harmonization_ablation_smri` config to benchmark vs. the baseline.
- **Deep (`murd_t1_t2`).** Apply MURD (Liu & Yap 2024) to T1/T2 volumes before FreeSurfer extraction; compare vs. ComBat and baseline to judge if image-space harmonization improves CCA/prediction.^[See MURD paper summary: `docs/generated/kb_curated/papers-md/murd_2024.md` and card `kb/paper_cards/murd_2024.yaml`.]
- **Representation unlearning (`site_unlearning_module`).** Optional adversarial head that removes site labels from embedding space (Dinsdale et al., 2021); treat as experimental until harmonization ablations justify it.^[See site-unlearning paper summary: `docs/generated/kb_curated/papers-md/dinsdale_site_unlearning_2021.md` and card `kb/paper_cards/dinsdale_site_unlearning_2021.yaml`.]

**Record harmonization IDs** (and preprocessing pipeline IDs such as `rsfmri_preprocessing_pipelines.hcp_like_minimal`) alongside embedding strategy IDs in every run.

---

## 🚀 Escalation Criteria

**If Fusion > max(Gene, Brain) with p < 0.05 (DeLong/bootstrap), consider:**
  - Two-tower contrastive alignment (frozen encoders; small projectors).
  - EI stacking over per-modality models.
  - Harmony-style hub tokens/TAPE if TR/site heterogeneity limits fMRI.

---

## 🔍 Interpretability

- **LOGO ΔAUC** with Wilcoxon + FDR for gene attribution
- **CCA loadings:** Partial correlations of axes with outcomes (covariate-adjusted)

## Tabular FM (TabPFN) baseline

- TabPFN (Nature 2024) is tracked under `kb/model_cards/tabpfn.yaml`. It is a **predictor**, not an fMRI encoder.
- Use TabPFN as a strong small-N tabular baseline on:
  - Raw sMRI ROI tables (`smri_free_surfer_raw_176`).
  - Genetics summary tables (`genetics_pgs_20traits`).
  - Early-fusion tabular features (ROI + PGS).
- Compare TabPFN vs. LR/LightGBM in `configs/experiments/03_prediction_baselines_tabular.yaml` to quantify how much structured embeddings help beyond a tabular FM.

Risks and mitigations

- Leakage: do scaling/residualization within train folds; apply transforms to test.
- Site imbalance: use group/site-aware CV when feasible.
- Overfitting at high dims: prefer 256–512; regularize LR; early stopping for GBDT.

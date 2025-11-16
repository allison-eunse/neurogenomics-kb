# Integration Strategy Template

## Metadata
- **Doc owner:** `<name>`
- **Last updated:** `<YYYY-MM-DD>`
- **Related decisions:** ``docs/decisions/<file>.md``
- **Model cards referenced:** `<comma-separated>`

## Fusion Taxonomy & Guidance
| Level | When to use | Risks / Notes |
| --- | --- | --- |
| **Early fusion** | Homogeneous modalities with aligned semantics | Modality collapse, confound bleed-through |
| **Intermediate fusion** | Shared latent required (e.g., cross-attn, hub tokens) | Heavy engineering, risk of overfitting |
| **Late fusion** | Heterogeneous semantics (DNA vs brain) | Requires good per-modality baselines |

Summarize takeaways from the oncology multimodal review + EI paper; note that default stance is late integration until baselines show reliable gains.

## Baseline Pipeline
1. **Preprocess per modality:** z-score within train folds, residualize age/sex/site/scanner/motion/SES/genetic PCs.
2. **Project to 512-D:** PCA (preferred) or tiny MLP (`Linear 1024→512 → GELU → Dropout 0.1 → Linear 512→512 → LayerNorm`).
3. **Association analysis:** CCA + 1,000 permutations; report top canonical correlations, permutation p-values, loadings.
4. **Prediction:** Logistic Regression (`class_weight='balanced'`) and LightGBM/CatBoost per modality + concatenated fusion; same CV folds; report AUROC/AUPRC ± SD.
5. **Statistical tests:** DeLong or bootstrap for AUROC differences; Wilcoxon + FDR for LOGO ΔAUC.

## Confound Controls
- Demographics: age, sex.
- Imaging: site/scanner, motion FD, TR group, SES if available.
- Genetics: top PCs (≥10) or ancestry group.
- Technical: sequencing batch, acquisition protocol indicators.
- Residualize within fold; log design matrices in `artifacts/generated/confounds/`.

## Evaluation Plan
- **CV design:** Stratified K-fold (k=5 or 10) with group/site-aware splits if leakage risk.
- **Metrics:** AUROC, AUPRC, calibration (Brier/ECE optional), canonical correlations.
- **Significance:** DeLong for AUROC, bootstrap for AUPRC, permutation for CCA, Benjamini–Hochberg for multiple tests.
- **Logging:** Store fold predictions, ROC/PR curves, permutation distributions under `artifacts/generated/metrics/<experiment_id>/`.

## Extension Roadmap
1. **Two-tower contrastive alignment:** Freeze encoders, train 512-D projectors with InfoNCE; assess retrieval R@k and downstream AUROC.
2. **Ensemble Integration (stacking / ensemble selection):** Train heterogeneous base learners per modality, stack with logistic meta-learner; record EI interpretation ranks.
3. **Joint latent models:** Brain Harmony hub tokens / TAPE or cross-attention fusion, only after late-fusion + EI baselines saturate.
4. **Deployment hygiene:** Missing-modality handling, calibration transfer, privacy considerations.

## References
- Ensemble Integration (Li et al., Bioinformatics Advances 2022)
- Multimodal oncology review (Waqas et al., 2024)
- BrainLM / Brain-JEPA / Brain Harmony primary papers
- Internal decisions (`docs/decisions/2025-11-integration-direction.md`)


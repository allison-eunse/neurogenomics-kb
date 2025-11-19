# Integration benchmarks

## Datasets
- UK Biobank
- Targets:
  - **Primary:** MDD diagnosis (binary; Howard et al. GWAS), PHQ-9 depression severity (continuous)
  - **Secondary cognitive:** Fluid intelligence, reaction time, cognitive composite
  - **Stratifications:** Early-onset MDD (age < 21), late-onset MDD (age ≥ 21)

## Splits
- GroupKFold by site
- Deterministic seeds with versioned `splits.json`

## Metrics & reporting
- AUC, PR-AUC, Brier score
- Calibration curves + subgroup reporting with confidence intervals







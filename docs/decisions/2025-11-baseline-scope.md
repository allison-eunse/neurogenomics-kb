# Baseline scope (Nov 2025)

## Rationale
- Focus on covariates-only, brain-only, and late-fusion baselines to bound uplift from multimodal methods.
- Use elastic net + CatBoost for speed, interpretability, and resilience with limited samples per site.
- Deliver Nov 26 table to unblock alignment experiments.

## Open questions
- Do we freeze genetics encoders or permit lightweight fine-tuning for baselines?
- How do we handle class imbalance for MDD vs cognition tasks within GroupKFold?
- Should we reserve a calibration set distinct from validation folds?




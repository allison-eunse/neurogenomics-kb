# Integration direction (Nov 2025)

## Rationale
- Prioritize CLIP-style contrastive alignment + gradient reversal for site invariance.
- Maintain Deep CCA variant as interpretable backup for regulators.
- Stage experiments: pretrain embeddings → align → downstream finetune.

## Open questions
- Which modality pairings get highest priority (whole-genome vs exome, structural vs functional)?
- Do we enforce shared latent dimension across all models or allow per-task projections?
- How strict should site-robustness penalties be before harming accuracy?




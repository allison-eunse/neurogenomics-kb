# Integration design patterns

## Early fusion
- Merge genetics + brain embeddings before task head
- Risk: modality imbalance and overfitting

## Shared latent space
- Project modalities into common latent prior to prediction
- Risk: difficult interpretability

## Cross-modal transformer
- Alternating attention layers across modalities
- Risk: high compute/VRAM

## Staged pretraining
- Pretrain per modality, freeze, then fine-tune joint layers
- Risk: stale frozen encoders




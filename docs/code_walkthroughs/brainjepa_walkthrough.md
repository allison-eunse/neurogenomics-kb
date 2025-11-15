# Brain-JEPA Code Walkthrough

## Overview
Brain-JEPA extends Image/Joint-Embedding Predictive Architecture ideas to 4D fMRI tensors: a Vision Transformer encoder ingests masked spatiotemporal blocks, a predictor Transformer reconstructs masked targets with gradient-informed positional encodings, and masking policies operate jointly across space and time.^[```1:200:external_repos/brainjepa/src/models/vision_transformer.py```][```18:282:external_repos/brainjepa/src/masks/spatialtemporal_multiblock.py```]

## At-a-Glance
| Architecture | Params | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| 4D Swin/ViT encoder + predictor head with gradient positional embeddings^[```22:400:external_repos/brainjepa/src/models/vision_transformer.py```] | Configurable (base uses ViT-Base + predictor depth 12)^[```400:565:external_repos/brainjepa/src/models/vision_transformer.py```] | 450 ROIs × 160 time frames (default)^[```19:210:external_repos/brainjepa/src/masks/spatialtemporal_multiblock.py```] | Preprocessed fMRI tensors from `fMRIDataset` (UKB/S1200)^[```14:205:external_repos/brainjepa/src/datasets/ukbiobank_scale.py```] | Spatiotemporal JEPA pretraining, downstream fine-tuning & linear probing scripts^[```67:360:external_repos/brainjepa/src/train.py```][```15:94:external_repos/brainjepa/downstream_eval.py```] | [github.com/janklees/brainjepa](https://github.com/janklees/brainjepa) |

### Environment & Hardware Notes
- **Conda + pip install.** Long-context masking requires the project’s Python 3.8 environment: `conda create -n brain-jepa python=3.8` followed by `pip install -r requirement.txt`.^[```80:82:external_repos/brainjepa/README.md```]
- **Hardware guidance.** Official docs note that pretraining ran on four A100 (40 GB) GPUs and provide the multi-GPU launch command `python main.py --fname configs/ukb_vitb_ep300.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3`.^[```84:92:external_repos/brainjepa/README.md```]
- **Gradient checkpoint flag.** The encoder exposes a `gradient_checkpointing` argument and wraps each block with `torch.utils.checkpoint.checkpoint(...)` whenever the flag is set, so you can trade compute for memory on large ROI × time grids.^[```422:504:external_repos/brainjepa/src/models/vision_transformer.py```]

## Key Components

### Dataset & Preprocessing (`src/datasets/ukbiobank_scale.py`)
The dataset class loads ROI-wise time series, applies robust scaling, optional downsampling, and returns tensors shaped `[channels, depth, height, width, time]` wrapped in dicts (`{'fmri': tensor}`) for the mask collator.

```14:186:external_repos/brainjepa/src/datasets/ukbiobank_scale.py
class fMRIDataset(Dataset):
    def __getitem__(self, idx):
        ts_cortical = self._load_ts(id, self.cortical_file)
        ts_subcortical = self._load_ts(id, self.subcortical_file)
        ts_array = np.concatenate((ts_subcortical, ts_cortical), axis=0).astype(np.float32)
        if self.downsample:
            ts_array = self._temporal_sampling(...)
        ts = torch.unsqueeze(torch.from_numpy(ts_array), 0).to(torch.float32)
        return {'fmri': ts}
```

### Mask Collator (`src/masks/spatialtemporal_multiblock.py`)
`MaskCollator_fmri` samples encoder/predictor windows over ROIs × time, enforcing non-overlapping context/target regions and returning boolean masks for each batch.

```18:282:external_repos/brainjepa/src/masks/spatialtemporal_multiblock.py
class MaskCollator_fmri(object):
    def __call__(self, batch):
        mask_e, _ = self._sample_block_mask_e(e_size)
        masks_p.append(self._sample_block_mask_p_roi(p_size_roi)[0])
        mask, mask_C = self._sample_block_mask_p_ts(...)
        mask_e = self.constrain_e_mask(mask_e, acceptable_regions=masks_C)
        collated_masks_pred.append([mask_p_final])
        collated_masks_enc.append([mask_e])
        return collated_batch, collated_masks_enc, collated_masks_pred
```

### Positional Embeddings & Encoder (`src/models/vision_transformer.py`)
Gradient-informed positional encoding (`GradTs_2dPE`) injects atlas gradients, while the encoder (`VisionTransformer`) patchifies `[B, C, D, H, W, T]` tensors, adds position encodings, and runs stacked Swin-like blocks.

```22:100:external_repos/brainjepa/src/models/vision_transformer.py
class GradTs_2dPE(nn.Module):
    def __init__(...):
        self.emb_h = nn.Parameter(...)
        self.emb_w = ... if add_w == 'origin' else predictor_pos_embed_proj(gradient)
```

```430:514:external_repos/brainjepa/src/models/vision_transformer.py
x = self.patch_embed(x)
pos_embed = self.pos_embed_proj(self.gradient_pos_embed)
x = x + pos_embed
if masks is not None:
    x = apply_masks(x, masks)
for blk in self.blocks:
    x = blk(x)
```

### Predictor Head (`src/models/vision_transformer.py`)
The predictor maps context tokens to a lower-dimensional space, concatenates learnable mask tokens (with their own positional embeddings), and runs Transformer blocks to regress target embeddings.

```280:396:external_repos/brainjepa/src/models/vision_transformer.py
class VisionTransformerPredictor(nn.Module):
    x = self.predictor_embed(x)
    predictor_pos_embed = self.predictor_2dpe_proj(self.gradient_pos_embed)
    pos_embs = apply_masks(predictor_pos_embed.repeat(B, 1, 1), masks)
    pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
    x = torch.cat([x, pred_tokens + pos_embs], dim=1)
    for blk in self.predictor_blocks:
        x = blk(x)
    x = self.predictor_proj(x[:, N_ctxt:])
```

### Training Loop (`src/train.py`)
The training script builds data loaders, mask collators, encoder/predictor pairs, and optimizers; the loss is Smooth L1 between predictor outputs and target encoder features.

```215:360:external_repos/brainjepa/src/train.py
def train_step():
    def forward_target():
        with torch.no_grad():
            h = target_encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))
            h = apply_masks(h, masks_pred)
            h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
    def forward_context():
        z = encoder(imgs, masks_enc, return_attention=False)
        z = predictor(z, masks_enc, masks_pred, return_attention=False)
    def loss_fn(z, h):
        return F.smooth_l1_loss(z, h)
```

### Downstream Evaluation (`downstream_tasks/models_vit.py`)
Linear-probe and fine-tuning scripts load pretrained encoders, optionally apply gradient checkpointing, and return global-pooled embeddings for classification/regression heads.

```15:74:external_repos/brainjepa/downstream_tasks/models_vit.py
self.encoder, _ = init_model(...)
if self.global_pool:
    outcome = self.fc_norm(self.encoder(x)[:, :, :].mean(dim=1))
else:
    outcome = self.encoder(x)[:, 0]
x = self.head(outcome)
```

## Integration Hooks (Brain ↔ Genetics)

- **Embedding shape.** Encoder outputs `[B, N_tokens, embed_dim]` (after flattening 4D patches). Downstream heads either take the CLS token or mean pool spatial tokens; JEPA predictors output only masked-token predictions shaped `[num_masks, embed_dim]`.^[```280:396:external_repos/brainjepa/src/models/vision_transformer.py```]
- **Pooling choices.** For multimodal fusion, use the downstream `VisionTransformer` global pool (`mean(dim=1)`) or compute mean pooling across context tokens to mirror predictor inputs.
- **Projection to shared latent.** Map `[B, embed_dim]` vectors (384/768 for small/base) into 512-D shared space via a lightweight projector:

```python
import torch.nn as nn

class BrainJEPAProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=512, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.layers(x)
```
- **Mask-aware embeddings.** When extracting representations for multimodal tasks, disable masking (feed identity masks) or average multiple masked views to reduce noise; the same mask collator can generate augmented views for contrastive objectives.
- **Gradient positional alignment.** Because `GradTs_2dPE` injects atlas gradients, keep those embeddings when aligning with genetics—do not strip them off—so the spatial axes remain consistent across modalities.^[```22:100:external_repos/brainjepa/src/models/vision_transformer.py```]

Following these hooks yields `[B, 512]` Brain-JEPA embeddings compatible with projected DNA embeddings (Evo 2, GENERator, Caduceus) for multimodal representation learning.

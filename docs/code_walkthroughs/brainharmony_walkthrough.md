# BrainHarmony Code Walkthrough

> **KB references:** [Model card](../models/brain/brainharmony.md) · [fMRI feature spec](../integration/modality_features/fmri.md) · [sMRI feature spec](../integration/modality_features/smri.md) · [Integration strategy](../integration/integration_strategy.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
BrainHarmony (a.k.a. BrainHarmonix) is a three-stage pipeline that first extracts modality-specific embeddings from fMRI ROI time-series and structural T1 volumes, then performs JEPA-style token-space pretraining, and finally fine-tunes classification heads on downstream cohorts (e.g., ABIDE). Stage 0 runs fused encoders with anatomical+functional positional priors, Stage 1 trains a latent-token predictor with smooth L1 loss between student and EMA targets, and Stage 2 attaches lightweight heads for clinical prediction.^[```1:94:external_repos/brainharmony/README.md```][```32:138:external_repos/brainharmony/configs/harmonizer/stage0_embed/conf_embed_pretrain.py```]

## At-a-Glance
| Architecture | Params | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| Dual FlexVisionTransformer encoders (fMRI + 3D T1) feeding a JEPA predictor with latent tokens, FlashAttention-2 blocks, and mask-conditioned regressors.^[```482:610:external_repos/brainharmony/libs/flex_transformer.py```][```22:260:external_repos/brainharmony/modules/harmonizer/stage1_pretrain/models.py```][```23:112:external_repos/brainharmony/libs/ssl_models/jepa_flex.py```] | Default “base” uses 768-d embeddings, 12 encoder blocks, 6-layer predictor, mask ratio 0.75, and 128 latent tokens (configurable via scripts).^[```48:138:external_repos/brainharmony/configs/harmonizer/stage0_embed/conf_embed_pretrain.py```][```38:49:external_repos/brainharmony/scripts/harmonizer/stage1_pretrain/run_pretrain.sh```] | 400 cortical ROIs × 490 TRs are chunked into 18 patches (48-step windows) plus optional 50 subcortical channels; structural MRI cubes are normalized to 160 × 192 × 160 voxels.^[```317:465:external_repos/brainharmony/datasets/datasets.py```][```499:561:external_repos/brainharmony/datasets/datasets.py```] | Stage 0 ingests `(fMRI, T1)` pairs via `UKB_FusionDataset`, Stage 1/2 read `.npz` embeddings with attention masks/labels using `GenerateEmbedDataset(_downstream)`.^[```566:581:external_repos/brainharmony/datasets/datasets.py```][```803:857:external_repos/brainharmony/datasets/datasets.py```] | Provided scripts wrap embedding extraction (Accelerate), JEPA pretraining, and downstream finetuning for reproducibility.^[```1:23:external_repos/brainharmony/scripts/harmonizer/stage0_embed/run_embed_pretrain.sh```][```1:49:external_repos/brainharmony/scripts/harmonizer/stage1_pretrain/run_pretrain.sh```][```1:59:external_repos/brainharmony/scripts/harmonizer/stage2_finetune/run_finetune.sh```] | [external_repos/brainharmony](../../external_repos/brainharmony) |

### Environment & Hardware Notes
- **Conda + pip workflow.** Create `brainharmonix` (Python 3.10), install CUDA 12.4 wheels for PyTorch 2.6, then `pip install -r requirements.txt` and `pip install -e .`.^[```40:56:external_repos/brainharmony/README.md```]
- **Checkpoint placement.** Download pretrained encoders (harmonix-f/s) plus harmonizer checkpoints and drop them under `checkpoints/{harmonix-f,harmonix-s,harmonizer}` before running Stage 0/1/2.^[```58:71:external_repos/brainharmony/README.md```]
- **FlashAttention 2 expectation.** `FlexVisionTransformer` selects FlashAttention 2 when installed (see `is_flash_attn_2_available`) so ensure compatible GPU builds or fall back to “eager” attention.^[```8:52:external_repos/brainharmony/libs/attn_utils/fa2_utils.py```][```138:214:external_repos/brainharmony/libs/flex_transformer.py```]

## Key Components

### Stage 0: Embedding Extraction (`modules/harmonizer/stage0_embed`)
Accelerate launches (`run_embed_pretrain.sh`) call `embedding_pretrain.py`, which loads configurable datasets (default `UKB_FusionDataset`) and wraps pretrained fMRI/T1 encoders specified in `conf_embed_pretrain.py`. The fmri encoder receives gradient+geometric-harmonic positional embeddings, while the MAE-style T1 encoder reuses volumetric patches. Each batch returns fmri tokens, T1 tokens, attention masks, and subject IDs; Stage 0 runs both encoders, concatenates their representations, and persists `.npz` files along with the attention mask for later stages.^[```1:23:external_repos/brainharmony/scripts/harmonizer/stage0_embed/run_embed_pretrain.sh```][```48:138:external_repos/brainharmony/configs/harmonizer/stage0_embed/conf_embed_pretrain.py```][```81:185:external_repos/brainharmony/modules/harmonizer/stage0_embed/embedding_pretrain.py```]

```566:581:external_repos/brainharmony/datasets/datasets.py
class UKB_FusionDataset(UKBDataset, UKB_T1_Dataset):
    def __init__(self, **kwargs):
        UKBDataset.__init__(self, **kwargs)
        UKB_T1_Dataset.__init__(self, **kwargs)

    def __getitem__(self, index):
        ts, _ = self.load_fmri(index)
        attn_mask = self.signal_attn_mask()
        t1 = self.load_t1(index)
        return ts, t1, self.patch_size, attn_mask, self.ids[index]
```

### Stage 1: Token-Space JEPA Pretraining (`modules/harmonizer/stage1_pretrain`)
The generated `.npz` files are streamed by `GenerateEmbedDataset`, which yields concatenated embeddings and their attention masks; distributed loaders feed `OneTokRegViT`, a latent-token ViT that appends learnable latent vectors and mask tokens before passing through decoder blocks. `train_one_epoch` applies Smooth L1 loss between the predictor output and EMA targets from the frozen teacher encoder, mirroring the JEPA objective.^[```803:825:external_repos/brainharmony/datasets/datasets.py```][```22:260:external_repos/brainharmony/modules/harmonizer/stage1_pretrain/models.py```][```13:84:external_repos/brainharmony/modules/harmonizer/stage1_pretrain/engine_pretrain.py```][```38:49:external_repos/brainharmony/scripts/harmonizer/stage1_pretrain/run_pretrain.sh```]

```803:825:external_repos/brainharmony/datasets/datasets.py
class GenerateEmbedDataset(Dataset):
    def __init__(self, root_dir, portion=1.0, seed=42):
        pattern = os.path.join(root_dir, "*.npz")
        all_files = sorted(glob.glob(pattern))
        self.files = all_files
        if len(all_files) == 0:
            raise RuntimeError(f"No .npz files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        arr = np.load(filepath)
        tensor = torch.from_numpy(arr["data"])
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor.squeeze(), arr["attn_mask"].squeeze()
```

```180:223:external_repos/brainharmony/modules/harmonizer/stage1_pretrain/models.py
    def forward_encoder(self, x, attn_mask):
        target = x
        if self.add_pre_mapping:
            x = self.pre_map(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        latent_tokens = self.latent_tokens.expand(x.shape[0], -1, -1)
        latent_tokens = latent_tokens + self.enc_latent_token_positional_embedding
        x = torch.cat([x, latent_tokens], dim=1)
        pad = torch.ones(attn_mask.shape[0], 1200 + latent_tokens.shape[1], dtype=attn_mask.dtype, device=attn_mask.device)
        pad_0 = torch.ones(attn_mask.shape[0], 1, dtype=attn_mask.dtype, device=attn_mask.device)
        attn_mask = torch.cat([pad_0, attn_mask, pad], dim=1)
        for blk in self.blocks:
            x = blk(x, attention_mask=attn_mask)
        x = self.norm(x)
        latent_tokens = torch.cat([x[:, :1, :], x[:, -self.num_latent_tokens :]], dim=1)
        return latent_tokens, target
```

### Stage 2: Downstream Harmonizer Heads (`modules/harmonizer/stage2_finetune`)
For tasks like ABIDE diagnosis, `GenerateEmbedDataset_downstream` reads the saved embeddings plus labels, and `VisionTransformer` attaches either a CLS-token head or global pooler atop the latent-token expanded sequence. Training mixes standard augmentation knobs (mixup/cutmix) with layer-wise LR decay, and evaluation logs accuracy + F1.^[```828:857:external_repos/brainharmony/datasets/datasets.py```][```1:350:external_repos/brainharmony/modules/harmonizer/stage2_finetune/main_finetune.py```][```16:166:external_repos/brainharmony/modules/harmonizer/stage2_finetune/engine_finetune.py```][```1:59:external_repos/brainharmony/scripts/harmonizer/stage2_finetune/run_finetune.sh```]

```117:170:external_repos/brainharmony/modules/harmonizer/stage2_finetune/models.py
    def forward_features(self, x, attn_mask):
        if self.add_pre_mapping:
            x = self.pre_map(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        latent_tokens = self.latent_tokens.expand(x.shape[0], -1, -1)
        latent_tokens = latent_tokens + self.enc_latent_token_positional_embedding
        x = torch.cat([x, latent_tokens], dim=1)
        pad = torch.ones(attn_mask.shape[0], 1200 + latent_tokens.shape[1], dtype=attn_mask.dtype, device=attn_mask.device)
        pad_0 = torch.ones(attn_mask.shape[0], 1, dtype=attn_mask.dtype, device=attn_mask.device)
        attn_mask = torch.cat([pad_0, attn_mask, pad], dim=1)
        for blk in self.blocks:
            x = blk(x, attention_mask=attn_mask)
        if self.global_pool:
            x = torch.cat([x[:, :1, :], x[:, -self.num_latent_tokens :]], dim=1)
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = torch.cat([x[:, :1, :], x[:, -self.num_latent_tokens :]], dim=1)
            outcome = x[:, 0]
        return outcome
```

### FlexVisionTransformer & Masked Predictor (`libs/flex_transformer.py`)
`FlexVisionTransformer` supports flexible patch sizes via `FlexiPatchEmbed`, optional gradient checkpointing, and either FlashAttention 2 or eager attention blocks. Predictor heads (`VisionTransformerPredictor`) project encoder outputs into predictor space, tile positional embeddings for masked regions, append learnable mask tokens, and regress back to encoder dimensionality; they reuse `apply_masks` to select context/target indices.^[```482:610:external_repos/brainharmony/libs/flex_transformer.py```][```322:463:external_repos/brainharmony/libs/flex_transformer.py```]

```403:463:external_repos/brainharmony/libs/flex_transformer.py
    def forward(self, x, masks_x, masks, attention_masks=None, return_attention=False):
        assert (masks is not None) and (masks_x is not None)
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]
        B = len(x) // len(masks_x)
        x = self.predictor_embed(x)
        predictor_pos_embed = self.predictor_pos_embed()[1]
        if self.cls_token is not None:
            x_pos_embed = predictor_pos_embed.repeat(B, 1, 1)
            x_pos_embed = apply_masks(x_pos_embed, masks_x, cls_token=True)
            x += x_pos_embed
            _, N_ctxt, D = x.shape
            pos_embs = predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs[:, 1:, :], masks)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
            pred_tokens += pos_embs
        else:
            x_pos_embed = predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(x_pos_embed, masks_x)
            _, N_ctxt, D = x.shape
            pos_embs = predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
            pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        for blk in self.predictor_blocks:
            x = blk(x, attention_masks)
        x = self.predictor_norm(x)
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)
        return x
```

### Positional Embeddings & Mask Utilities
BrainHarmony blends anatomical gradients with geometric harmonics to produce shared positional priors across encoder/predictor stacks, and the same module can emit decoder embeddings for Stage 1. Mask helpers expose gather-style APIs for re-indexing context/target tokens.^[```137:209:external_repos/brainharmony/libs/position_embedding.py```][```11:31:external_repos/brainharmony/libs/masks/utils.py```]

```167:209:external_repos/brainharmony/libs/position_embedding.py
        geo_harm_pos_embed = self.geo_harm_proj(self.geo_harm)
        gradient_pos_embed = self.grad_proj(self.gradient)
        pos_embed = (gradient_pos_embed + geo_harm_pos_embed) * 0.5
        emb_w = pos_embed.squeeze().repeat_interleave(self.repeat_time, dim=0)
        emb_w = (emb_w - emb_w.min()) / (emb_w.max() - emb_w.min()) * 2 - 1
        emb_encoder = torch.cat([self.emb_h_encoder, emb_w], dim=1).unsqueeze(0)
        if self.cls_token:
            pos_embed_encoder = torch.concat(
                [torch.zeros([1, 1, emb_encoder.shape[2]], requires_grad=False).to(self.device), emb_encoder],
                dim=1,
            )
        else:
            pos_embed_encoder = emb_encoder
        if self.use_pos_embed_decoder:
            emb_w_decoder = self.decoder_pos_embed_proj(emb_w.detach())
            emb_decoder = torch.cat([self.emb_h_decoder, emb_w_decoder], dim=1).unsqueeze(0)
            if self.cls_token:
                pos_embed_decoder = torch.concat(
                    [torch.zeros([1, 1, emb_decoder.shape[2]], requires_grad=False).to(self.device), emb_decoder],
                    dim=1,
                )
            else:
                pos_embed_decoder = emb_decoder
            return pos_embed_encoder, pos_embed_decoder
        return pos_embed_encoder, None
```

```11:31:external_repos/brainharmony/libs/masks/utils.py
def apply_masks(x, masks, cls_token=False):
    all_x = []
    if cls_token:
        cls_t = x[:, :1, :]
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.cat((cls_t, torch.gather(x[:, 1:, :], dim=1, index=mask_keep)), dim=1)]
    else:
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)
```

## Integration Hooks (Brain ↔ Genetics)
- **Token shapes.** `FlexVisionTransformer.forward` outputs `[B, N_tokens, embed_dim]` (CLS + patches [+ latent tokens]); Stage 2 heads either take the CLS vector or mean-pool latent tokens, so downstream genetics encoders should expect 768-d (base) or 1024-d (large) vectors per sample.^[```563:610:external_repos/brainharmony/libs/flex_transformer.py```][```117:170:external_repos/brainharmony/modules/harmonizer/stage2_finetune/models.py```]
- **Attention masks.** Both `GenerateEmbedDataset` variants surface per-sample masks; reusing them when aligning with long genomic sequences preserves which ROI/time windows were padded vs. observed.^[```803:857:external_repos/brainharmony/datasets/datasets.py```]
- **Stage bridging.** Stage 0 writes `.npz` files with `data` and `attn_mask` arrays; you can append additional modalities (e.g., gene-expression embeddings) into the same `data` vector before Stage 1 as long as the downstream models’ positional encoders are updated accordingly.^[```138:184:external_repos/brainharmony/modules/harmonizer/stage0_embed/embedding_pretrain.py```]
- **Projecting to shared latent spaces.** A lightweight projector keeps BrainHarmony tokens compatible with genetics embeddings:

```python
import torch.nn as nn

class BrainHarmonyProjector(nn.Module):
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

Map mean-pooled latent tokens through this projector, concatenate with genetics embeddings, and fine-tune a fusion head while reusing BrainHarmony’s attention masks for masking-aware losses.


# SwiFT Code Walkthrough

> **KB references:** [Model card](../models/brain/swift.md) · [fMRI feature spec](../integration/modality_features/fmri.md) · [Integration strategy](../integration/integration_strategy.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
SwiFT (Swin 4D fMRI Transformer) tokenizes 4D fMRI volumes with 3D convolutions, processes them with windowed 4D self-attention (spatial + temporal windows), and trains contrastive or supervised heads via PyTorch Lightning.^[```1:400:external_repos/swift/project/module/models/swin4d_transformer_ver7.py```][```1:188:external_repos/swift/project/main.py```]

## At-a-Glance
| Architecture | Params | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| Swin-inspired 4D transformer w/ window attention & patch merging^[```21:400:external_repos/swift/project/module/models/swin4d_transformer_ver7.py```] | Configurable (e.g., embed_dim=96, depths from config)^[```402:565:external_repos/swift/project/module/models/swin4d_transformer_ver7.py```] | 96×96×96 voxels × 20 frames (default)^[```250:300:external_repos/swift/project/module/utils/data_module.py```] | Preprocessed volumes from `fMRIDataModule` (UKB/HCP/etc.)^[```13:260:external_repos/swift/project/module/utils/data_module.py```] | Lightning training with contrastive or supervised heads, downstream evaluation scripts^[```21:187:external_repos/swift/project/main.py```][```32:395:external_repos/swift/project/module/pl_classifier.py```] | [github.com/Transconnectome/SwiFT](https://github.com/Transconnectome/SwiFT) |

### Environment & Hardware Notes
- **Conda environment.** The README tells you to run `conda env create -f envs/py39.yaml` followed by `conda activate py39` to pull in the exact PyTorch/Lightning versions used for the released checkpoints.^[```45:55:external_repos/swift/README.md```]
- **Gradient checkpoint knobs.** Every Swin4D stage accepts `use_checkpoint` and executes `torch.utils.checkpoint.checkpoint(...)` when set, so add `use_checkpoint=True` in your model config to extend contexts without exceeding GPU memory.^[```224:312:external_repos/swift/project/module/models/swin4d_transformer_ver7.py```][```507:744:external_repos/swift/project/module/models/swin4d_transformer_ver7.py```]

## Key Components

### Data Module (`project/module/utils/data_module.py`)
`fMRIDataModule` loads datasets (UKB, HCP, etc.), splits subjects, and returns PyTorch `DataLoader`s. Augmentations (affine/noise) are applied in the Lightning module.

```13:230:external_repos/swift/project/module/utils/data_module.py
class fMRIDataModule(pl.LightningDataModule):
    def get_dataset(self):
        if self.hparams.dataset_name == "S1200": return S1200
        ...
    def setup(self, stage=None):
        Dataset = self.get_dataset()
        params = {"root": self.hparams.image_path, "sequence_length": self.hparams.sequence_length, ...}
        self.train_dataset = Dataset(**params, subject_dict=train_dict, ...)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, ...)
```

### Patch Embedding & Window Attention (`swin4d_transformer_ver7.py`)
`PatchEmbed` downsamples volumes with strided 3D convs, `WindowAttention4D` computes attention inside local 4D windows, and `SwinTransformerBlock4D` applies shifted windows for better coverage. `PatchMergingV2` reduces spatial resolution while keeping temporal size.

```202:399:external_repos/swift/project/module/models/swin4d_transformer_ver7.py
class PatchEmbed(nn.Module):
    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))
...
class WindowAttention4D(nn.Module):
    def forward(self, x, mask):
        qkv = self.qkv(x).reshape(...)
        attn = self.softmax((q @ k.transpose(-2, -1)) * self.scale)
        x = (attn @ v)
```

### Swin4D Backbone (`swin4d_transformer_ver7.py`)
`BasicLayer` stacks windowed blocks, handles padding, applies attention masks, and optionally downsamples. The main `SwinTransformer4D` builds multiple stages with positional embeddings, patch merging, and normalization.

```400:796:external_repos/swift/project/module/models/swin4d_transformer_ver7.py
class BasicLayer(nn.Module):
    for blk in self.blocks:
        x = blk(x, attn_mask)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
    if self.downsample is not None:
        x = self.downsample(x)
...
class SwinTransformer4D(nn.Module):
    self.patch_embed = PatchEmbed(...)
    self.layers = nn.ModuleList([...])
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = self.pos_embeds[i](x)
            x = self.layers[i](x.contiguous())
        return x
```

### Lightning Module (`project/module/pl_classifier.py`)
`LitClassifier` wraps the encoder, applies augmentations if requested, and attaches task-specific heads (classification/regression/contrastive). `_calculate_loss` routes to BCE, MSE, or contrastive losses.

```32:205:external_repos/swift/project/module/pl_classifier.py
self.model = load_model(self.hparams.model, self.hparams)
if self.hparams.downstream_task == 'sex':
    self.output_head = load_model("clf_mlp", self.hparams)
elif self.hparams.downstream_task == 'age':
    self.output_head = load_model("reg_mlp", self.hparams)
...
def _calculate_loss(self, batch, mode):
    if self.hparams.pretraining:
        # contrastive losses (NT-Xent)
    else:
        subj, logits, target = self._compute_logits(batch)
        if classification:
            loss = F.binary_cross_entropy_with_logits(logits, target)
        else:
            loss = F.mse_loss(logits.squeeze(), target.squeeze())
```

### Training Entry Point (`project/main.py`)
CLI parses dataset/model/task args, instantiates the Lightning module + data module, and launches PyTorch Lightning `Trainer` with callbacks (checkpointing, LR monitor).

```18:187:external_repos/swift/project/main.py
parser = ArgumentParser(...)
parser = Classifier.add_model_specific_args(parser)
parser = Dataset.add_data_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
data_module = Dataset(**vars(args))
model = Classifier(data_module=data_module, **vars(args))
trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)
trainer.fit(model, datamodule=data_module)
```

## Integration Hooks (Brain ↔ Genetics)

- **Embedding shape.** Encoder outputs `[B, N_tokens, embed_dim]`. Downstream heads either global-average tokens (`mean(dim=[2,3,4])`) or use CLS-like features (depending on head). Use `_compute_logits` to capture the tensor before the head for multimodal projection.^[```108:205:external_repos/swift/project/module/pl_classifier.py```]
- **Pooling choices.** Mean pooling across spatial dimensions (`features.mean(dim=[2,3,4])`) produces `[B, embed_dim]`; temporal pooling can be added if you keep time as a separate axis prior to patch merging.
- **Projection to shared latent.** Apply a lightweight projector to map `[B, embed_dim]` into a 512-D shared space:

```python
import torch.nn as nn

class SwiFTProjector(nn.Module):
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
- **Augmentation awareness.** When extracting embeddings for alignment, disable augmentations (`augment_during_training=False`) to avoid random affine/noise perturbations that would misalign with genetic features.^[```108:205:external_repos/swift/project/module/pl_classifier.py```]
- **Window constraints.** Ensure inference volumes match training window sizes (`img_size`, `window_size`)—`get_window_size` shrinks windows when needed, but you lose attention overlap if sizes are too small.^[```110:200:external_repos/swift/project/module/models/swin4d_transformer_ver7.py```]

After projection, SwiFT embeddings (global pooled or CLS) can be concatenated or contrastively aligned with Evo 2/GENERator/Caduceus projections for multimodal neurogenomics.

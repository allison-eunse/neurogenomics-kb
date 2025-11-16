# BrainMT Code Walkthrough

> **KB references:** [Model card](../models/brain/brainmt.md) · [fMRI feature spec](../integration/modality_features/fmri.md) · [Integration strategy](../integration/integration_strategy.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
BrainMT pairs bidirectional Mamba mixers (temporal-first scanning) with MHSA transformer blocks to model long-range fMRI dynamics, delivering state-of-the-art regression/classification on UKB and HCP phenotypes.^[```3:170:external_repos/brainmt/README.md```][```294:462:external_repos/brainmt/src/brainmt/models/brain_mt.py```]

## At-a-Glance
| Architecture | Params | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| 3D Conv patch embed → bidirectional Mamba blocks → Transformer attention blocks^[```202:462:external_repos/brainmt/src/brainmt/models/brain_mt.py```] | Configurable (default hidden 512, depth `[12,8]`)^[```293:375:external_repos/brainmt/src/brainmt/models/brain_mt.py```] | 91×109×91 voxels × 200 frames (default)^[```294:339:external_repos/brainmt/src/brainmt/models/brain_mt.py```] | Preprocessed `.pt` tensors from `data/datasets.py`^[```15:80:external_repos/brainmt/src/brainmt/data/datasets.py```] | DDP training with regression/classification heads, inference utilities^[```1:330:external_repos/brainmt/src/brainmt/train.py```][```1:390:external_repos/brainmt/src/brainmt/inference.py```] | [github.com/arunkumar-kannan/brainmt-fmri](https://github.com/arunkumar-kannan/brainmt-fmri) |

### Environment & Hardware Notes
- **Exact environment commands.** The README targets Python 3.9.18 + PyTorch 2.6/CUDA 12.4, created via `python -m venv brainmt_env`, `source brainmt_env/bin/activate`, and `pip install -r requirements.txt`.^[```44:60:external_repos/brainmt/README.md```]
- **Gradient checkpoint flag.** Every Mamba block accepts `use_checkpoint` and conditionally calls `checkpoint.checkpoint(...)`, so you can instantiate `BrainMT(..., use_checkpoint=True)` to reduce memory usage on long temporal contexts.^[```95:125:external_repos/brainmt/src/brainmt/models/brain_mt.py```][```293:334:external_repos/brainmt/src/brainmt/models/brain_mt.py```]

## Key Components

### Dataset Loader (`src/brainmt/data/datasets.py`)
The dataset stores fMRI volumes as fp16 tensors (`func_data_MNI_fp16.pt`), slices contiguous time segments, permutes them into `[frames, channel, depth, height, width]`, and returns `(tensor, target)` pairs.

```15:80:external_repos/brainmt/src/brainmt/data/datasets.py
class fMRIDataset(Dataset):
    def __getitem__(self, idx):
        data = torch.load(img_file)
        start_index = torch.randint(0, total_frames - num_frames + 1, (1,)).item()
        data_sliced = data[:, :, :, start_index:end_index]
        data_global = data_sliced.unsqueeze(0).permute(4, 0, 2, 1, 3)
        target = self.target_dict[subject_dir]
        return data_global, torch.tensor(target, dtype=torch.float16)
```

### Patch Embed & Conv Blocks (`src/brainmt/models/brain_mt.py`)
`PatchEmbed` downsamples the 4D tensor with strided 3D convolutions before two ConvBlocks + Downsample layers reduce spatial resolution while keeping temporal length.

```202:263:external_repos/brainmt/src/brainmt/models/brain_mt.py
class PatchEmbed(nn.Module):
    self.conv_down = nn.Sequential(
        nn.Conv3d(in_chans, in_dim, 3, 2, 1, bias=False),
        nn.ReLU(),
        nn.Conv3d(in_dim, dim, 3, 2, 1, bias=False),
        nn.ReLU()
    )
```

### Hybrid Mamba + Transformer Backbone (`src/brainmt/models/brain_mt.py`)
Temporal-first processing reshapes tokens, feeds them through `create_block` (bidirectional Mamba) and then through transformer attention + MLP to capture residual spatial dependencies.

```331:462:external_repos/brainmt/src/brainmt/models/brain_mt.py
self.layers = nn.ModuleList([
    create_block(embed_dim, ssm_cfg=ssm_cfg, ..., drop_path=inter_dpr[i], ...)
    for i in range(depth[0])
])
self.blocks = nn.ModuleList([
    Attention(embed_dim, num_heads=num_heads, ...)
    for i in range(depth[1])
])
...
def forward_features(self, x, ...):
    x = self.patch_embed(x)
    x = self.conv_block0(x); x = self.downsample0(x)
    x = self.conv_block1(x); x = self.downsample1(x)
    x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
    x = x + self.temporal_pos_embedding
    for layer in self.layers:
        hidden_states, residual = layer(hidden_states, residual, ...)
    for block in self.blocks:
        hidden_states = hidden_states + drop_path_attn(block(self.norm(hidden_states)))
```

### Forward & Head (`src/brainmt/models/brain_mt.py`)
CLS token is prepended before Mamba blocks; `forward` returns final MLP head output for regression/classification.

```400:461:external_repos/brainmt/src/brainmt/models/brain_mt.py
cls_token = self.cls_token.expand(x.shape[0], -1, -1)
x = torch.cat((cls_token, x), dim=1)
...
return hidden_states[:, 0, :]
```

### Training Loop (`src/brainmt/train.py`)
Hydra config builds datasets, wraps the model in DDP, selects loss (MSE or BCEWithLogits), constructs layer-wise LR decay groups, and trains with `GradScaler` + cosine warm restarts.

```132:234:external_repos/brainmt/src/brainmt/train.py
model = BrainMT(**model_config).to(device)
model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], ...)
if cfg.task.loss_fn == "mse":
    criteria = nn.MSELoss()
...
train_loss, train_outputs, train_targets = train_one_epoch(model, criteria, train_loader, optimizer, scaler, device, epoch, cfg)
val_loss, val_outputs, val_targets = evaluate(model, criteria, val_loader, device)
```

### Inference (`src/brainmt/inference.py`)
The inference script mirrors dataset splits, loads checkpoints, and computes detailed metrics (accuracy/AUROC for classification, MSE/MAE/R²/Pearson for regression), plus optional plots.

```26:210:external_repos/brainmt/src/brainmt/inference.py
model = BrainMT(**model_config).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
if cfg.task.name == 'classification':
    metrics = calculate_classification_metrics(test_outputs, test_targets)
else:
    metrics = calculate_regression_metrics(test_outputs, test_targets)
```

## Integration Hooks (Brain ↔ Genetics)

- **Embedding shape.** `forward_features` returns `[B, hidden]` CLS vectors (hidden default 512). To access intermediate token embeddings, tap `hidden_states[:, 1:, :]` before the final average/MLP.^[```400:462:external_repos/brainmt/src/brainmt/models/brain_mt.py```]
- **Pooling choices.** CLS token encodes temporal-first, globally attentive context. For voxel-conditioned embeddings, reshape post-Mamba tensor back to `[B, voxels, hidden]` prior to the transformer block and average along the voxel axis.
- **Projection to shared latent.** Map `[B, 512]` BrainMT vectors into a 512-D multimodal space with a lightweight projector:

```python
import torch.nn as nn

class BrainMTProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, dropout=0.1):
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
- **Normalization.** Because BrainMT ends with LayerNorm (`self.norm_f`), additional LayerNorm in the projector keeps the scale comparable to genetic embeddings.
- **Temporal handling.** The temporal-first scan (`rearrange(..., b n) t m -> b (n t) m`) is crucial for long-range modeling—preserve this ordering if you export intermediate features for contrastive alignment with DNA sequences.^[```421:444:external_repos/brainmt/src/brainmt/models/brain_mt.py```]

Projected BrainMT features can then be concatenated or contrastively aligned with Evo 2/GENERator/Caduceus embeddings to study genetics↔fMRI correspondences.

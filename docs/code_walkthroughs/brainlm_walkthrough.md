# BrainLM Code Walkthrough

## Overview
BrainLM is a ViT-MAE–style masked autoencoder: it slices each voxel’s time course into short windows, randomly masks most of them, and reconstructs the missing segments with Nystromformer encoder layers and a lightweight decoder trained on UK Biobank fMRI.^[```1:48:external_repos/brainlm/README.md```][```63:205:external_repos/brainlm/brainlm_mae/modeling_brainlm.py```]

## At-a-Glance
| Architecture | Params | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| ViT-MAE encoder (Nystromformer) + MAE decoder^[```227:515:external_repos/brainlm/brainlm_mae/modeling_brainlm.py```] | 111 M / 650 M checkpoints^[```39:48:external_repos/brainlm/README.md```] | 424 parcels × 490 timepoints patched into windows^[```63:200:external_repos/brainlm/brainlm_mae/modeling_brainlm.py```] | Arrow datasets of `[B, voxels, time]` tensors + XYZ coordinates^[```43:205:external_repos/brainlm/train.py```] | Masked reconstruction, downstream probes/fine-tuning via `BrainLMTrainer`^[```351:470:external_repos/brainlm/train.py```] | [github.com/vandijklab/BrainLM](https://github.com/vandijklab/BrainLM) |

### Environment & Hardware Notes
- **Setup script.** The README directs users to run `sh setup.sh` to create the conda environment (with FlashAttention for the 111 M/650 M checkpoints) and verify PyTorch/HuggingFace installs via the provided sanity commands.^[```16:26:external_repos/brainlm/README.md```][```50:52:external_repos/brainlm/README.md```]
- **Gradient checkpointing toggle.** Both the encoder and decoder wrap their Nystromformer layers with `if self.gradient_checkpointing and self.training: torch.utils.checkpoint.checkpoint(...)`, so you can enable `model.gradient_checkpointing_enable()` before large runs to keep memory in check.^[```245:269:external_repos/brainlm/brainlm_mae/modeling_brainlm.py```][```453:500:external_repos/brainlm/brainlm_mae/modeling_brainlm.py```]

## Key Components

### Data & Collation (`brainlm/train.py`)
Hydra dataclasses declare dataset paths, voxel/time dimensions, and labels; `collate_fn` stacks tensors into the format expected by the MAE model.

```43:210:external_repos/brainlm/train.py
@dataclass
class DataTrainingArguments:
    train_dataset_path: str
    val_dataset_path: str
    coords_dataset_path: str
    num_timepoints_per_voxel: int = 490
    timepoint_patching_size: int = 49
...
def collate_fn(examples):
    signal_vectors = torch.stack([example["signal_vectors"] for example in examples], dim=0)
    xyz_vectors = torch.stack([example["xyz_vectors"] for example in examples])
    labels = torch.stack([example["label"] for example in examples])
    return {"signal_vectors": signal_vectors, "xyz_vectors": xyz_vectors, "input_ids": signal_vectors, "labels": labels}
```

### Embeddings & Masking (`brainlm_mae/modeling_brainlm.py`)
`BrainLMEmbeddings` reshapes time signals into patches, projects signals and spatial coordinates, injects positional encoding, and randomly masks patches before appending a CLS token.

```63:160:external_repos/brainlm/brainlm_mae/modeling_brainlm.py
reshaped_signal_vectors = torch.reshape(signal_vectors, (batch, num_voxels, -1, self.timepoint_patching_size))
signal_projection = self.signal_embedding_projection(reshaped_signal_vectors)
xyz_projection = self.xyz_embedding_projection(xyz_vectors).unsqueeze(2).repeat(1, 1, num_patch_tokens, 1)
x = self.pos_embedding(signal_projection + xyz_projection)
embeddings, mask, ids_restore = self.random_masking(x, noise=noise)
cls_tokens = self.cls_token.expand(embeddings.shape[0], -1, -1)
embeddings = torch.cat((cls_tokens, embeddings), dim=1)
```

### Encoder & Decoder (`brainlm_mae/modeling_brainlm.py`)
The encoder stacks Nystromformer layers, while the decoder reintroduces mask tokens, adds spatial/time encodings again, and predicts the missing time windows.

```227:340:external_repos/brainlm/brainlm_mae/modeling_brainlm.py
class BrainLMModel(ViTMAEModel):
    self.embeddings = BrainLMEmbeddings(config)
    self.encoder = BrainLMEncoder(config)
    encoder_outputs = self.encoder(embedding_output, ...)
    sequence_output = self.layernorm(encoder_outputs[0])
```

```355:515:external_repos/brainlm/brainlm_mae/modeling_brainlm.py
mask_tokens = self.mask_token.repeat(batch_size, num_mask_tokens, 1)
x_ = torch.reshape(x_, (batch_size, self.num_brain_voxels, num_patch_tokens, hidden_dim))
x_ = x_ + self.decoder_xyz_projection(xyz_vectors)
x_ = self.pos_embedding(x_)
logits = self.decoder_pred2(self.decoder_pred_nonlinearity(self.decoder_pred1(hidden_states)))
logits = torch.reshape(logits, (batch_size, self.num_brain_voxels, ..., self.timepoint_patching_size))
```

### Loss (`brainlm_mae/modeling_brainlm.py`)
Masked reconstruction loss is computed only on the masked tokens (MSE or MAE).

```562:584:external_repos/brainlm/brainlm_mae/modeling_brainlm.py
mask = mask.unsqueeze(-1).repeat(1, 1, 1, pred_values.shape[-1])
if self.config.loss_fn == "mse":
    loss = (((pred_values - signal_values) ** 2) * mask).sum() / mask.sum()
elif self.config.loss_fn == "mae":
    loss = abs((pred_values - signal_values) * mask).sum() / mask.sum()
```

### Training Driver (`brainlm/train.py`)
`BrainLMTrainer` glues everything together—optimizer, scheduler, metrics, evaluation.

```351:470:external_repos/brainlm/train.py
trainer = BrainLMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=None,
    data_collator=collate_fn,
    compute_metrics=metrics.compute_metrics,
)
trainer.train()
```

## Integration Hooks (Brain ↔ Genetics)

- **Embedding shape.** Encoder outputs `[B, (num_voxels * kept_tokens) + 1, hidden]`. Index 0 is CLS; the rest represent unmasked voxel windows sorted deterministically.^[```329:350:external_repos/brainlm/brainlm_mae/modeling_brainlm.py```]
- **Pooling choices.** CLS pooling mirrors the MAE training objective; mean pooling across tokens smooths noise; reshaping tokens back to `[B, voxels, windows, hidden]` lets you average over time first, then voxels.
- **Projection to shared latent.** Map pooled `[B, hidden]` vectors (hidden≈768 on the 111 M model) into a 512-D shared space:

```python
import torch.nn as nn

class BrainLMProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)
```

- **Masking control.** When extracting embeddings, set `mask_ratio=0.0` so every patch contributes; enable masking only for pretraining/augmentation.
- **Alignment with genetics.** After projection, normalize (LayerNorm or z-score) before concatenating with genetic embeddings (Evo 2, GENERator, Caduceus) or using contrastive loss.

This workflow delivers `[B, 512]` fMRI embeddings that align with projected DNA representations for multimodal analyses.

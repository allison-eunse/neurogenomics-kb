# Caduceus Code Walkthrough

> **KB references:** [Model card](../models/genetics/caduceus.md) · [Genomics feature spec](../integration/modality_features/genomics.md) · [Integration strategy](../integration/integration_strategy.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
Caduceus couples Mamba sequence mixers with reverse-complement parameter sharing so every layer sees both DNA strands simultaneously, yielding 131 kbp masked-language-model checkpoints that remain equivariant to strand flips across the published HuggingFace collection.^[```6:141:external_repos/caduceus/README.md```]

## At-a-Glance
| Architecture | Params | Context | Tokenization / Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| RC-equivariant BiMamba blocks inside `CaduceusMixerModel`^[```33:389:external_repos/caduceus/caduceus/modeling_caduceus.py```] | ~150 M (e.g., 256 × 16 layers in HF releases)^[```15:22:external_repos/caduceus/README.md```] | 131 kbp pretraining windows^[```15:22:external_repos/caduceus/README.md```] | Character tokenizer w/ explicit complement map and BOS/PAD logic^[```15:158:external_repos/caduceus/src/dataloaders/datasets/hg38_char_tokenizer.py```] | Lightning training for pretraining/downstream tasks, RC embeddings for VEP^[```1:400:external_repos/caduceus/train.py```][```30:399:external_repos/caduceus/vep_embeddings.py```] | [github.com/kuleshov-group/caduceus](https://github.com/kuleshov-group/caduceus) |

### Environment & Hardware Notes
- **Conda bootstrap.** Long-context experiments rely on the repo’s environment file; create it with `conda env create -f caduceus_env.yml` and activate via `conda activate caduceus_env` before running the Hydra configs.^[```53:63:external_repos/caduceus/README.md```]
- **Gradient checkpointing status.** The backbone still has a TODO for checkpointing and the HF wrapper sets `supports_gradient_checkpointing = False`, so memory savings must come from RCPS channel splitting or ZeRO rather than built-in checkpoint flags.^[```228:231:external_repos/caduceus/caduceus/modeling_caduceus.py```][```297:302:external_repos/caduceus/caduceus/modeling_caduceus.py```]

## Key Components

### Tokenizer & Preprocessing (`hg38_char_tokenizer.py`, `rc.py`)
Tokenization is strictly character-based, enumerating all specials and precomputing a complement map so RCPS layers can look up complements without re-tokenizing. String-level utilities supply reverse complements for augmentation or evaluation.

```15:74:external_repos/caduceus/src/dataloaders/datasets/hg38_char_tokenizer.py
class CharacterTokenizer(PreTrainedTokenizer):
    self._vocab_str_to_int = {
        "[CLS]": 0,
        ...
        "[UNK]": 6,
        **{ch: i + 7 for i, ch in enumerate(characters)},
    }
    ...
    complement_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    self.complement_map[self._vocab_str_to_int[k]] = complement_id
```

```7:27:external_repos/caduceus/src/dataloaders/utils/rc.py
STRING_COMPLEMENT_MAP = {"A": "T", ...}
def string_reverse_complement(seq):
    rev_comp = ""
    for base in seq[::-1]:
        rev_comp += STRING_COMPLEMENT_MAP.get(base, base)
    return rev_comp
```

### Positional & RC Handling (`modeling_caduceus.py`)
Bidirectional Mamba wrappers run forward and reverse streams (optionally weight tied) and merge them, while RCPS-aware embeddings split channel dimensions so hidden states encode forward and RC halves that can be combined later.

```87:141:external_repos/caduceus/caduceus/modeling_caduceus.py
class BiMambaWrapper(nn.Module):
    self.mamba_fwd = Mamba(...)
    if bidirectional:
        self.mamba_rev = Mamba(...)
        if bidirectional_weight_tie:
            self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
    def forward(...):
        out = self.mamba_fwd(hidden_states, ...)
        if self.bidirectional:
            out_rev = self.mamba_rev(hidden_states.flip(dims=(1,))).flip(dims=(1,))
            out = out + out_rev if self.bidirectional_strategy == "add" else out * out_rev
```

```152:214:external_repos/caduceus/caduceus/modeling_caduceus.py
class CaduceusEmbeddings(nn.Module):
    if config.rcps:
        self.word_embeddings = RCPSEmbedding(...)
...
class CaduceusMixerModel(nn.Module):
    self.layers = nn.ModuleList([
        create_block(..., rcps=config.rcps, ...) for i in range(config.n_layer)
    ])
    self.norm_f = norm_f if (config.fused_add_norm or not config.rcps) else RCPSAddNormWrapper(norm_f)
```

### Backbone & Embedding Wrapper (`dna_embedding.py`)
`DNAEmbeddingModelCaduceus` strips the LM head and exposes hidden states shaped `[B, L, d]` for standard mode or `[B, L, d, 2]` when RCPS/conjoined inference is enabled.

```156:195:external_repos/caduceus/src/models/sequence/dna_embedding.py
class DNAEmbeddingModelCaduceus(DNAEmbeddingModel):
    def forward(...):
        if self.config.rcps:
            hidden_states = self.caduceus(input_ids, return_dict=False)
            num_chan = hidden_states.shape[-1]
            return torch.stack(
                [hidden_states[..., :num_chan // 2], torch.flip(hidden_states[..., num_chan // 2:], dims=[1, 2])],
                dim=-1
            ), None
        if self.conjoin_train or (self.conjoin_test and not self.training):
            hidden_states = self.caduceus(input_ids[..., 0], return_dict=False)
            hidden_states_rc = self.caduceus(input_ids[..., 1], return_dict=False)
            return torch.stack([hidden_states, hidden_states_rc], dim=-1), None
        return self.caduceus(input_ids, return_dict=False), None
```

### Training Loop (`train.py`)
The Lightning `SequenceLightningModule` builds datasets/encoders from Hydra configs, forwards batches through encoder/decoder stacks, logs losses/metrics, and supports distributed strategies plus gradient accumulation.

```126:377:external_repos/caduceus/train.py
class SequenceLightningModule(pl.LightningModule):
    def setup(...):
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](**self.hparams.dataset)
        ...
        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        self.loss = self.task.loss
    def _shared_step(...):
        x, y, w = self.forward(batch)
        loss = self.loss(x, y, **w)
        metrics = self.metrics(x, y, **w)
        self.log_dict({f"{prefix}/{k}": v for k, v in metrics.items()}, ...)
```

### Inference & VEP Embeddings (`vep_embeddings.py`)
The VEP helper loads any HF model (Caduceus by default), tokenizes ref/alt + RC windows, averages variant-centered windows, and writes `.pt` tensors per split—handy for multimodal or downstream regression.

```30:392:external_repos/caduceus/vep_embeddings.py
class DNAEmbeddingModel(nn.Module):
    def forward(self, input_ids):
        return self.backbone(input_ids).last_hidden_state
...
tokens_window_ref = torch.gather(
    item_ref, 1,
    expanded_indices.unsqueeze(-1).expand(-1, -1, item_ref.size(2))
).mean(dim=1)
storage_dict["concat_avg_ws"] = torch.cat([tokens_window_ref, tokens_window_alt], dim=-1)
```

### Sequence Constraints (`configuration_caduceus.py`)
All RC/bidirectional behavior is driven by config: enabling RCPS, picking merge strategy (`add` vs `ew_multiply`), and passing complement maps extracted from the tokenizer.

```10:55:external_repos/caduceus/caduceus/configuration_caduceus.py
class CaduceusConfig(PretrainedConfig):
    def __init__(..., bidirectional: bool = True, bidirectional_strategy: Union[str, None] = "add",
                 bidirectional_weight_tie: bool = True, rcps: bool = False, complement_map: Optional[dict] = None, ...):
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.rcps = rcps
        self.complement_map = complement_map
```

## Integration Hooks (Genetics ↔ Brain)

- **Embedding shapes.** Outputs are `[B, L, d]` by default, `[B, L, d, 2]` when strands are stacked, or `[B, K, 2d]` after VEP window pooling—mean over tokens to get `[B, d]` per strand before any fusion.^[```156:195:external_repos/caduceus/src/models/sequence/dna_embedding.py```][```275:385:external_repos/caduceus/vep_embeddings.py```]
- **Pooling choices.** Mean pooling across tokens mirrors the masked-LM objective; to keep strand info, average forward and RC halves separately, then concatenate them to `[B, 2d]`.
- **Projection to shared latent.** Map pooled vectors into a 512-D multimodal space with a lightweight projector:

```python
import torch.nn as nn

class CaduceusProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, dropout=0.1):
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

- **Normalization & RC alignment.** When RCPS doubles channels, follow the model’s own `RCPSAddNormWrapper` convention: flip the RC half, sum/average with the forward half, and only then project—this keeps embeddings invariant to strand order.^[```210:274:external_repos/caduceus/caduceus/modeling_caduceus.py```]
- **Batch & memory tips.** The Lightning module instantiates DDP-aware samplers and can rehydrate checkpoints via `utils.instantiate`, so mirror that pattern (especially `_shared_step`) when adding multimodal heads to avoid redundant forward passes.^[```291:377:external_repos/caduceus/train.py```]

Using this flow yields `[B, 512]` Caduceus embeddings ready to align with BrainLM CLS tokens, average BrainMT CLS outputs, or SwiFT pooled features for cross-modal genetics↔fMRI experiments.

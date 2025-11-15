# Evo 2 Code Walkthrough

## Overview
Evo 2 packages StripedHyena 2 checkpoints behind a lightweight Python API so you can run 1 Mbp autoregressive DNA modeling, scoring, and generation without touching the underlying Vortex stack. The repo exposes the 1B/7B/40B parameter checkpoints described in the bioRxiv preprint and HuggingFace collection, all of which share the same tokenizer (single‐nucleotide CharLevelTokenizer with a 512-symbol vocab) and reverse-complement aware inference utilities.^[```5:118:external_repos/evo2/README.md```]

## At-a-Glance
| Architecture | Params | Context | Tokenization / Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| StripedHyena 2 mixer loaded via Vortex (`StripedHyena` backbone)^[```171:258:external_repos/evo2/evo2/models.py```] | 1B / 7B / 40B checkpoints (`MODEL_NAMES`)^[```1:33:external_repos/evo2/evo2/utils.py```] | `max_seqlen: 1048576` (≈1 Mbp)^[```3:60:external_repos/evo2/evo2/configs/evo2-7b-1m.yml```] | Char-level tokenizer with 512 symbols; padding handled in `prepare_batch`^[```19:51:external_repos/evo2/evo2/models.py```][```10:34:external_repos/evo2/evo2/scoring.py```] | Autoregressive scoring, reverse-complement averaging, cached generation^[```109:169:external_repos/evo2/evo2/models.py```][```127:170:external_repos/evo2/evo2/scoring.py```] | [github.com/ArcInstitute/evo2](https://github.com/ArcInstitute/evo2) |

### Environment & Hardware Notes
- **Exact pip/conda setup for FP8 + long context.** The README prescribes installing Transformer Engine and FlashAttention before running million-token inference:  
  `conda install -c nvidia cuda-nvcc cuda-cudart-dev`  
  `conda install -c conda-forge transformer-engine-torch=2.3.0`  
  `pip install flash-attn==2.8.0.post2 --no-build-isolation`^[```54:58:external_repos/evo2/README.md```]
- **Hardware guidance.** Official instructions call for Linux/WSL2 with CUDA 12.1+, FP8-capable GPUs (compute capability ≥8.9) and note that the 40 B checkpoints require multi-GPU partitioning via Vortex; Python 3.12 is required for the packaged binaries.^[```34:52:external_repos/evo2/README.md```]

## Key Components

### Tokenizer & Preprocessing (`evo2/evo2/models.py`, `evo2/evo2/scoring.py`)
The API always instantiates a Vortex `CharLevelTokenizer(512)` and pairs it with pad/BOS conventions inside `prepare_batch`, which right-pads sequences to the longest length in the batch while optionally prepending an EOD token for language modeling reductions.

```19:34:external_repos/evo2/evo2/models.py
class Evo2:
    def __init__(self, model_name: str = MODEL_NAMES[1], local_path: str = None):
        ...
        self.tokenizer = CharLevelTokenizer(512)
```

```10:34:external_repos/evo2/evo2/scoring.py
def prepare_batch(
        seqs: List[str],
        tokenizer: object,
        prepend_bos: bool = False,
        device: str = 'cuda:0'
) -> Tuple[torch.Tensor, List[int]]:
    ...
    padding = [tokenizer.pad_id] * (max_seq_length - len(seq))
    input_ids.append(
        torch.tensor(
            ([tokenizer.eod_id] * int(prepend_bos)) + tokenizer.tokenize(seq) + padding,
            dtype=torch.long,
        ).to(device).unsqueeze(0)
    )
```

### Positional & Reverse-Complement Handling (`evo2/evo2/scoring.py`)
Reverse-complement scoring duplicates the batch, flips it via Biopython, and averages the two likelihood traces so downstream metrics remain strand invariant.

```127:170:external_repos/evo2/evo2/scoring.py
def score_sequences_rc(...):
    batch_seqs_rc = [ str(Seq(seq).reverse_complement()) for seq in batch_seqs ]
    ...
    batch_scores = _score_sequences(...)
    batch_scores_rc = _score_sequences(...)
    batch_scores = (np.array(batch_scores) + np.array(batch_scores_rc)) * 0.5
```

### Backbone Loader (`evo2/evo2/models.py`, `evo2/evo2/configs/evo2-7b-1m.yml`)
`load_evo2_model` resolves the YAML config, instantiates `StripedHyena` with the Hyena/Mamba layer schedule, and merges HF shards (removing them afterward) before loading the `.pt` checkpoint.

```171:258:external_repos/evo2/evo2/models.py
def load_evo2_model(...):
    config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))
    model = StripedHyena(config)
    load_checkpoint(model, weights_path)
    return model
```

```3:65:external_repos/evo2/evo2/configs/evo2-7b-1m.yml
hidden_size: 4096
num_layers: 32
max_seqlen: 1048576
tokenizer_type: CharLevelTokenizer
use_flash_attn: True
```

### Objective & Diagnostics (`evo2/evo2/test/test_evo2.py`)
The supplied test harness tokenizes CSV prompts, runs a forward pass, and reports cross-entropy plus next-token accuracy so you can validate checkpoints against the published numbers.

```34:124:external_repos/evo2/evo2/test/test_evo2.py
with torch.inference_mode():
    logits, _ = model.model.forward(input_ids.unsqueeze(0))
    loss = F.cross_entropy(pred_logits, target_ids.long())
    accuracy = (target_ids == pred_tokens).float().mean().item()
...
expected_metrics = {
    'evo2_40b': {'loss': 0.2159424, 'acc': 91.673},
    ...
}
```

### Inference Helpers (`evo2/evo2/models.py`)
`score_sequences` and `generate` wrap the Hyena forward pass with batching, tokenizer reuse, and memory-aware knobs such as cached generation and `force_prompt_threshold` so you can guard against OOMs on long prompts.

```109:169:external_repos/evo2/evo2/models.py
def score_sequences(...):
    scoring_func = partial(score_sequences_rc if average_reverse_complement else score_sequences, ...)
    with torch.no_grad():
        scores = scoring_func(seqs)
...
def generate(...):
    output = vortex_generate(
        prompt_seqs=prompt_seqs,
        model=self.model,
        tokenizer=self.tokenizer,
        n_tokens=n_tokens,
        temperature=temperature,
        top_k=top_k,
        cached_generation=cached_generation,
        force_prompt_threshold=force_prompt_threshold,
    )
```

### Embedding Extraction Hooks (`evo2/evo2/models.py`)
Forward hooks can be registered on any named submodule (e.g., `blocks.28.mlp`) so you can capture intermediate representations without modifying Vortex internals.

```52:105:external_repos/evo2/evo2/models.py
def forward(..., return_embeddings: bool = False, layer_names=None):
    if return_embeddings:
        layer = self.model.get_submodule(name)
        handles.append(layer.register_forward_hook(hook_fn(name)))
    logits = self.model.forward(input_ids)
    if return_embeddings:
        return logits, embeddings
```

### Sequence Constraints (`evo2/evo2/configs/evo2-7b-1m.yml`, `evo2/evo2/scoring.py`)
Hyena checkpoints assume 1 M tokens and the char tokenizer emits IDs per nucleotide, so batching must pad or truncate to keep tensors rectangular; the scorer enforces this padding and strips BOS tokens before computing positional log-likelihoods.

```52:88:external_repos/evo2/evo2/scoring.py
softmax_logprobs = torch.log_softmax(logits, dim=-1)
softmax_logprobs = softmax_logprobs[:, :-1]
input_ids = input_ids[:, 1:]
logprobs = torch.gather(softmax_logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
```

## Integration Hooks (Genetics ↔ Brain)

- **Embedding shape.** The Hyena forward returns logits shaped `[B, L, vocab]` and any hooked layer keeps `[B, L, hidden]`, matching the token dimension used in `_score_sequences`. Mean pooling along the token axis reproduces the default `reduce_method='mean'` used for log-likelihoods.^[```70:89:external_repos/evo2/evo2/scoring.py```]
- **Pooling choices.** Use mean pooling for global sequence summaries, max pooling to emphasize motifs, or last-token pooling when you want autoregressive state; all three are equivalent to selecting how you reduce `logprobs[idx][:seq_len]` inside the scorer.^[```79:88:external_repos/evo2/evo2/scoring.py```]
- **Reverse-complement normalization.** For strand-agnostic features, reuse `score_sequences_rc` logic: tokenize both strands, run the same hook set, and average activations before pooling so the resulting `[B, H]` encodes orientation-invariant context.^[```127:170:external_repos/evo2/evo2/scoring.py```]
- **Projection to shared space.** Once you have pooled `[B, H]` embeddings (H≈4096 for the 7B model), feed them through a lightweight projector to align with fMRI representations. A drop-in block that mirrors common multimodal heads is:

```python
import torch.nn as nn

class MultimodalProjector(nn.Module):
    def __init__(self, input_dim=4096, output_dim=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.proj(x)
```

- **Batch & memory tips.** Long prompts can trigger Hyena’s FFT prefill path; set `force_prompt_threshold` or `cached_generation=False` when working on 24 GB GPUs, and ensure you shard batches via the `batch_size` argument in `score_sequences` to keep `prepare_batch` from materializing multi-megabase tensors at once.^[```109:169:external_repos/evo2/evo2/models.py```][```10:34:external_repos/evo2/evo2/scoring.py```]

With these hooks, Evo 2 embeddings (`[B, H]` after pooling) can be projected into the same 512-D latent space used by your fMRI encoder, normalized (LayerNorm) for stability, and concatenated or contrastively aligned with brain-derived features.
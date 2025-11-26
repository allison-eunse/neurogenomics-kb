# StripedHyena Code Walkthrough

> **KB references:** [Integration strategy](../integration/integration_strategy.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
StripedHyena is a hybrid sequence model that alternates between Hyena convolution blocks and grouped rotary attention blocks, achieving near-linear scaling with context length while maintaining competitive performance with Transformers. The architecture uses state-space models (SSMs) implemented via finite impulse response (FIR) and infinite impulse response (IIR) filters for efficient long-context processing, with attention layers providing targeted pattern recall capabilities.^[```9:20:external_repos/hyena/README.md```][```336:403:external_repos/hyena/stripedhyena/model.py```]

## At-a-Glance
| Architecture | Params | Context | Tokenization / Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| Alternating Hyena blocks (gated convolutions) and Attention blocks (grouped rotary MHA)^[```324:333:external_repos/hyena/stripedhyena/model.py```] | 7B (StripedHyena-Hessian-7B, StripedHyena-Nous-7B)^[```32:41:external_repos/hyena/README.md```] | Up to 32k tokens (configurable via `max_seqlen`)^[```35:36:external_repos/hyena/configs/7b-sh-32k-v1.yml```] | HuggingFace tokenizer or character-level tokenizer; input shape `[B, L]`^[```340:371:external_repos/hyena/stripedhyena/model.py```] | Efficient autoregressive generation (>500k tokens with 80GB GPU), faster decoding than Transformers, recurrent inference mode^[```16:20:external_repos/hyena/README.md```][```14:158:external_repos/hyena/stripedhyena/generation.py```] | [github.com/togethercomputer/stripedhyena](https://github.com/togethercomputer/stripedhyena) |

### Environment & Hardware Notes
- **Docker setup.** The recommended installation path uses Docker: `docker build --tag sh:test .` followed by `docker run -it --gpus all ...`. The Dockerfile installs FlashAttention, rotary/normalization kernels, and other dependencies.^[```58:68:external_repos/hyena/README.md```]
- **FlashAttention requirement.** FlashAttention v2 is required for the attention blocks; the code checks for `flash_attn.modules.mha.MHA` and falls back gracefully if unavailable. RMSNorm can optionally use flash kernels when `use_flash_rmsnorm=True`.^[```15:26:external_repos/hyena/stripedhyena/layers.py```][```39:50:external_repos/hyena/stripedhyena/model.py```]
- **Mixed precision handling.** Poles and residues must remain in `float32` for numerical stability; all other parameters can be converted to `bfloat16` via `to_bfloat16_except_poles_residues()`.^[```438:445:external_repos/hyena/stripedhyena/model.py```]
- **Prefill modes.** Long sequences support multiple prefill strategies: `fft` (default, fast for even-length sequences), `recurrence` (slower but lower memory), and `modal-fft` (hybrid). Very long prompts may require `prefill_style: recurrence` to avoid OOM.^[```16:23:external_repos/hyena/stripedhyena/engine.py```][```79:82:external_repos/hyena/README.md```]

## Key Components

### Model Architecture (`stripedhyena/model.py`)

`StripedHyena` alternates between `ParallelGatedConvBlock` (Hyena) and `AttentionBlock` layers based on config-specified indices. The model supports both stateless (training) and stateful (inference) forward passes, with inference parameters managing recurrent state for efficient generation.

**Model structure with alternating blocks:**

```336:403:external_repos/hyena/stripedhyena/model.py
class StripedHyena(nn.Module):
    def __init__(self, config):
        self.embedding_layer = VocabParallelEmbedding(config)
        self.norm = RMSNorm(config) if config.get("final_norm", True) else None
        self.unembed = self.embedding_layer if config.tie_embeddings else VocabParallelEmbedding(config)
        self.blocks = nn.ModuleList([
            get_block(config, layer_idx, flash_fft=self.flash_fft) 
            for layer_idx in range(config.num_layers)
        ])
    def forward(self, x, inference_params_dict=None, padding_mask=None):
        x = self.embedding_layer.embed(x)
        if inference_params_dict is not None:
            x, inference_params_dict_out = self.stateful_forward(x, inference_params_dict=inference_params_dict)
        else:
            x, inference_params_dict_out = self.stateless_forward(x, padding_mask=padding_mask)
        x = self.norm(x)
        x = self.unembed.unembed(x)
        return x, inference_params_dict_out
```

**Block selection logic:**

```324:333:external_repos/hyena/stripedhyena/model.py
def get_block(config, layer_idx, flash_fft=None):
    if layer_idx in config.attn_layer_idxs:
        return AttentionBlock(config, layer_idx)
    elif layer_idx in config.hyena_layer_idxs:
        block = ParallelGatedConvBlock(config, layer_idx)
        if config.get("use_flashfft", "False"):
            block.filter.fftconv_fn = flash_fft
        return block
```

### Hyena Filter (`stripedhyena/model.py`)

The `ParallelHyenaFilter` implements the core Hyena operation: a short FIR filter followed by a long IIR filter parameterized by learnable poles and residues. The filter supports both parallel (training/prefill) and sequential (autoregressive generation) modes.

**Hyena filter structure:**

```85:215:external_repos/hyena/stripedhyena/model.py
class ParallelHyenaFilter(nn.Module):
    def __init__(self, config, layer_idx):
        self.short_filter_length = config.short_filter_length
        self.short_filter_weight = nn.Parameter(torch.randn(3 * config.hidden_size, 1, config.short_filter_length))
        self.poles = nn.Parameter(poles)
        self.residues = nn.Parameter(torch.randn(self.num_systems, self.state_size, 1, 2))
        self.engine = HyenaInferenceEngine(layer_idx=layer_idx)
    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if inference_params is not None and self.layer_idx in inference_params.fir_state_dict.keys():
            return self.sequential_forward(u, inference_params)
        else:
            return self.parallel_forward(u, inference_params, padding_mask)
```

**Parallel forward (training/prefill):**

```163:215:external_repos/hyena/stripedhyena/model.py
def parallel_forward(self, u, inference_params=None, padding_mask=None):
    L = u.shape[1]
    z_pre, fir_state = self.engine.parallel_fir(
        self.fir_fn, u, self.short_filter_weight, self.short_filter_bias, L,
        fir_length=self.short_filter_length, inference_params=inference_params, padding_mask=padding_mask
    )
    if self.h is None:
        h, filter_dtype, poles, residues = self.compute_filter(L, u.device)
    y = self.engine.parallel_iir(
        z_pre, h, self.D, L, t=self.t, poles=self.poles, residues=self.residues,
        dims=dims, inference_params=inference_params, layer_idx=self.layer_idx,
        prefill_style=self.config.get("prefill_style", "fft"), use_flashfft=self.use_flashfft, ...
    )
    return y, inference_params
```

**Sequential forward (autoregressive generation):**

```217:251:external_repos/hyena/stripedhyena/model.py
def sequential_forward(self, u, inference_params):
    if len(u.shape) > 2:
        u = u[:, -1]
    fir_state, iir_state = inference_params.fir_state_dict[self.layer_idx], inference_params.state_dict[self.layer_idx]
    z_pre, fir_state = self.engine.step_fir(u, fir_state, weight=self.short_filter_weight, bias=self.short_filter_bias)
    x2, x1, v = column_split(z_pre, self.num_attention_heads, self.hidden_size_per_attention_head)
    y, iir_state = self.engine.step_iir(x2, x1, v, self.D, self.residues, self.poles, iir_state, iir_groups=self.hyena_filter_groups)
    inference_params.fir_state_dict[self.layer_idx] = fir_state
    inference_params.state_dict[self.layer_idx] = iir_state
    return y[:, None], inference_params
```

### Hyena Inference Engine (`stripedhyena/engine.py`)

The `HyenaInferenceEngine` handles FIR and IIR computations for both parallel and sequential modes. It supports multiple prefill strategies (FFT, recurrence, modal-FFT) and manages state caching for efficient generation.

**Parallel FIR computation:**

```65:109:external_repos/hyena/stripedhyena/engine.py
def parallel_fir(self, fir_fn, u, weight, bias, L, fir_length=3, inference_params=None, prefill_mode=None, padding_mask=None):
    if fir_fn != torch.nn.functional.conv1d:
        z_pre = fir_fn(u)[:, :L]
        z_pre = z_pre.permute(0, 2, 1)
    else:
        u = u.permute(0, 2, 1)
        z_pre = fir_fn(u, weight, bias=None, stride=1, padding=fir_length - 1, groups=u.shape[1])[..., :L]
        z_pre = z_pre + bias[None, :, None]
    if type(padding_mask) == torch.Tensor:
        z_pre = z_pre * padding_mask[:, None]
    if inference_params is not None:
        fir_state = u[..., -fir_length + 1 :]
    return z_pre, fir_state
```

**Parallel IIR with FFT:**

```111:215:external_repos/hyena/stripedhyena/engine.py
def parallel_iir(self, z_pre, h, D, L, poles, residues, t, dims, inference_params=None, prefill_style="fft", fftconv_fn=None, ...):
    x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)
    x1v = x1 * v
    if use_flashfft and (L % 2) == 0:
        y = fftconv_fn(x1v.to(dtype=torch.bfloat16).contiguous(), h.to(dtype=torch.float32))
    else:
        H = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
        X_s = torch.fft.fft(x1v.to(dtype=torch.float32), n=fft_size)
        X = X_s[..., : H.shape[-1]]
        y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]
    y = y.to(dtype=x1v.dtype)
    y = (y + x1v * D.unsqueeze(-1)) * x2
    return y, inference_params
```

### Attention Block (`stripedhyena/model.py`)

Attention blocks use FlashAttention-v2 with grouped query attention (GQA) and rotary positional embeddings. They provide targeted pattern recall capabilities that complement the Hyena blocks' long-context processing.

**Attention block structure:**

```26:82:external_repos/hyena/stripedhyena/model.py
class AttentionBlock(nn.Module):
    def __init__(self, config, layer_idx):
        self.pre_norm, self.post_norm = RMSNorm(config), RMSNorm(config)
        self.inner_mha_cls = MHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_heads_kv=config.num_attention_heads // self.proj_groups,
            rotary_emb_dim=config.hidden_size // config.num_attention_heads,
            causal=True, layer_idx=layer_idx, use_flash_attn=self.config.use_flash_attn,
        ).to(dtype=dtype)
        self.mlp = ParallelGatedMLP(config).to(dtype=mlp_dtype)
    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        u = self.inner_mha_cls(self.pre_norm(u), inference_params=inference_params) + u
        u = self.mlp(self.post_norm(u)) + u
        return u, None
```

### Gated Convolution Block (`stripedhyena/model.py`)

`ParallelGatedConvBlock` wraps the Hyena filter with input projections, output projections, and an MLP, following a standard transformer-like structure with residual connections.

**Gated convolution block:**

```277:321:external_repos/hyena/stripedhyena/model.py
class ParallelGatedConvBlock(nn.Module):
    def __init__(self, config, layer_idx):
        self.pre_norm, self.post_norm = RMSNorm(config).to(dtype=dtype), RMSNorm(config).to(dtype=dtype)
        self.filter = ParallelHyenaFilter(config, layer_idx).to(dtype=dtype)
        self.projections = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_filter_dense = nn.Linear(config.hidden_size, config.hidden_size).to(dtype)
        self.mlp = ParallelGatedMLP(config).to(dtype=mlp_dtype)
    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        z = self.proj_norm_fn(u)
        z, inference_params = self.filter(z, inference_params=inference_params, padding_mask=padding_mask)
        z_in = self.out_filter_dense(z) + u
        y = self.res_mlp_norm_fn(z_in)
        return y, inference_params
```

### Generation (`stripedhyena/generation.py`)

The `Generator` class handles autoregressive text generation with support for cached generation (recurrent mode) and standard generation. It manages inference parameters and state updates across tokens.

**Generation with caching:**

```14:158:external_repos/hyena/stripedhyena/generation.py
class Generator:
    def generate(self, device, input_string=None, input_ids=None, num_tokens=32, cached_generation=False, ...):
        if cached_generation:
            inference_params_dict_out = self.model.initialize_inference_params()
        for i in range(int(num_tokens)):
            post_prefill = cached_generation and i > 0
            if post_prefill:
                x = x[:, -1:]
                inference_params_dict_out["mha"].seqlen_offset += 1
                inference_params_dict_out["hyena"].seqlen_offset += 1
            with torch.no_grad():
                logits, inference_params_dict_out = self.model(x, inference_params_dict=inference_params_dict_out)
            last_logits = logits[:, -1]
            new_idx = sample(last_logits, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature)
            if post_prefill:
                x = new_idx[:, None]
            else:
                x = torch.cat([x, new_idx[:, None]], dim=-1)
        return generation[:, : i + 1], scores[:, : i + 1]
```

### Configuration (`configs/7b-sh-32k-v1.yml`)

The YAML config specifies layer indices for attention vs. Hyena blocks, filter parameters, and inference settings. The default 7B model alternates attention and Hyena blocks.

**Configuration structure:**

```1:53:external_repos/hyena/configs/7b-sh-32k-v1.yml
model_name: sh-7b-32k-v1
vocab_size: 32000
hidden_size: 4096
num_filters: 4096
attn_layer_idxs: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
hyena_layer_idxs: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
num_layers: 32
short_filter_length: 3
num_attention_heads: 32
state_size: 2
rotary_emb_base: 500000
proj_groups: 4
hyena_filter_groups: 1
max_seqlen: 32768
use_flash_attn: True
use_flash_rmsnorm: True
prefill_style: fft
```

## Integration Hooks

- **Embedding extraction.** The model returns logits `[B, L, vocab_size]` from `forward()`. To extract embeddings, access hidden states before the final norm: `hidden_states = model.embedding_layer.embed(input_ids)` then pass through blocks manually, or modify `forward()` to return `x` before `self.unembed.unembed(x)`.^[```358:371:external_repos/hyena/stripedhyena/model.py```]
- **Pooling strategies.** Mean pooling across sequence length yields `[B, hidden_size]` representations. For long sequences, consider pooling only over the last N tokens or using attention-weighted pooling. The alternating block structure means embeddings capture both local (Hyena) and global (attention) patterns.^[```381:387:external_repos/hyena/stripedhyena/model.py```]
- **State management for long contexts.** When processing sequences longer than training length, use `cached_generation=True` to enable recurrent mode. The inference parameters (`InferenceParams` for attention, `RecurrentInferenceParams` for Hyena) manage KV caches and FIR/IIR state across generation steps.^[```389:402:external_repos/hyena/stripedhyena/model.py```]
- **Projection to shared latent space.** Map pooled `[B, hidden_size]` embeddings (hidden_size=4096 for 7B) to a 512-D multimodal space using a lightweight projector similar to other foundation models. The Hyena blocks' long-context capabilities make this architecture suitable for processing extended genomic or brain imaging sequences.^[```358:371:external_repos/hyena/stripedhyena/model.py```]
- **Filter precomputation for efficiency.** For fixed-length inputs, call `model.precompute_filters(L, device)` before training/inference to cache the IIR filter `h` and avoid recomputing it on each forward pass. This is especially beneficial when batch processing many sequences of the same length.^[```404:420:external_repos/hyena/stripedhyena/model.py```]
- **Alternating block design.** The model alternates between Hyena and attention blocks, with Hyena blocks handling the majority of computation. This design allows the model to leverage both efficient long-context processing (Hyena) and targeted pattern recall (attention), making it suitable for diverse sequence modeling tasks in neuro-omics applications.^[```324:333:external_repos/hyena/stripedhyena/model.py```]


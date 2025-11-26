# Flamingo Code Walkthrough

> **KB references:** [Integration strategy](../integration/integration_strategy.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
OpenFlamingo implements a few-shot visual-language model that interleaves pretrained vision encoders (CLIP ViT) with causal language models (MPT, RedPajama, LLaMA, OPT) via gated cross-attention layers. The architecture conditions language generation on visual features through a Perceiver resampler and sparse cross-attention blocks inserted every N decoder layers, enabling in-context learning for vision-language tasks without task-specific fine-tuning.^[```74:100:external_repos/flamingo/README.md```][```17:56:external_repos/flamingo/open_flamingo/src/flamingo.py```]

## At-a-Glance
| Architecture | Params | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| CLIP ViT encoder + PerceiverResampler + GatedCrossAttentionBlock + causal LM decoder^[```17:56:external_repos/flamingo/open_flamingo/src/flamingo.py```][```68:133:external_repos/flamingo/open_flamingo/src/helpers.py```] | 3B–9B (vision frozen, only cross-attn/perceiver trainable)^[```104:111:external_repos/flamingo/open_flamingo/src/factory.py```] | Interleaved image-text sequences with `<image>` and `<|endofchunk|>` tokens^[```178:196:external_repos/flamingo/README.md```] | `vision_x`: `[B, T_img, F, C, H, W]`; `lang_x`: tokenized text with media markers^[```60:122:external_repos/flamingo/open_flamingo/src/flamingo.py```] | Few-shot captioning, VQA, image-text generation via `generate()`; FSDP training with gradient checkpointing^[```200:220:external_repos/flamingo/README.md```][```202:277:external_repos/flamingo/open_flamingo/src/flamingo.py```] | [github.com/mlfoundations/open_flamingo](https://github.com/mlfoundations/open_flamingo) |

### Environment & Hardware Notes
- **Installation.** Base package via `pip install open-flamingo`; training/eval extras via `pip install open-flamingo[training]` or `pip install open-flamingo[eval]`. Conda environment available via `conda env create -f environment.yml`.^[```28:51:external_repos/flamingo/README.md```]
- **FSDP wrapping strategy.** The `wrap_fsdp()` method manually wraps vision encoder, perceiver, gated cross-attention layers, and LM embeddings with double-wrapped FSDP to enable parameter offloading. Decoder layers are unfrozen for FSDP compatibility but excluded from the optimizer to effectively freeze them.^[```202:301:external_repos/flamingo/open_flamingo/src/flamingo.py```]
- **Gradient checkpointing.** Both the perceiver and decoder layers support gradient checkpointing when `gradient_checkpointing=True` is passed to `init_flamingo()`, reducing memory during training at the cost of recomputation.^[```26:58:external_repos/flamingo/open_flamingo/src/flamingo.py```]

## Key Components

### Model Factory (`open_flamingo/src/factory.py`)

`create_model_and_transforms()` instantiates a CLIP vision encoder, loads a causal LM, extends it with `FlamingoLMMixin`, and wires cross-attention layers at configurable intervals. Special tokens (`<image>`, `<|endofchunk|>`) are added to the tokenizer, and all parameters are frozen except the perceiver, gated cross-attention layers, and optionally LM embeddings.

**Model initialization with frozen backbones:**

```11:119:external_repos/flamingo/open_flamingo/src/factory.py
def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    ...
):
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(...)
    vision_encoder.visual.output_tokens = True
    text_tokenizer = AutoTokenizer.from_pretrained(...)
    text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
    lang_encoder = AutoModelForCausalLM.from_pretrained(...)
    extend_instance(lang_encoder, FlamingoLMMixin)
    model = Flamingo(vision_encoder, lang_encoder, ...)
    model.requires_grad_(False)
    model.perceiver.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
```

### Vision Encoding & Perceiver Resampling (`open_flamingo/src/flamingo.py`, `open_flamingo/src/helpers.py`)

The vision encoder extracts patch features from images, which are then resampled by the Perceiver into a fixed number of latent tokens per image. The Perceiver uses cross-attention to compress variable-length visual sequences into a consistent representation.

**Vision encoding and conditioning:**

```177:200:external_repos/flamingo/open_flamingo/src/flamingo.py
def _encode_vision_x(self, vision_x: torch.Tensor):
    assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
    b, T, F = vision_x.shape[:3]
    assert F == 1, "Only single frame supported"
    vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
    with torch.no_grad():
        vision_x = self.vision_encoder(vision_x)[1]
    vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
    vision_x = self.perceiver(vision_x)
    for layer in self.lang_encoder._get_decoder_layers():
        layer.condition_vis_x(vision_x)
```

**Perceiver resampler architecture:**

```68:132:external_repos/flamingo/open_flamingo/src/helpers.py
class PerceiverResampler(nn.Module):
    def __init__(self, *, dim, depth=6, dim_head=64, heads=8, num_latents=64, ...):
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            ]) for _ in range(depth)
        ])
    def forward(self, x):
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)
```

### Gated Cross-Attention (`open_flamingo/src/helpers.py`, `open_flamingo/src/flamingo_lm.py`)

Gated cross-attention layers enable text tokens to attend to visual features at specific media locations. The gating mechanism (tanh-activated learnable scalars) controls the contribution of cross-modal information, allowing the model to learn when to rely on visual context versus pure language modeling.

**Gated cross-attention block:**

```236:279:external_repos/flamingo/open_flamingo/src/helpers.py
class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, *, dim, dim_visual, dim_head=64, heads=8, ...):
        self.attn = MaskedCrossAttention(dim=dim, dim_visual=dim_visual, ...)
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))
    def forward(self, x, media, media_locations=None, use_cached_media=False):
        x = self.attn(x, media, media_locations=media_locations, use_cached_media=use_cached_media) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x
```

**Media location conditioning:**

```136:233:external_repos/flamingo/open_flamingo/src/helpers.py
class MaskedCrossAttention(nn.Module):
    def forward(self, x, media, media_locations=None, use_cached_media=False):
        if exists(media_locations):
            text_time = media_locations.cumsum(dim=-1)
            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)
```

### Flamingo Layer Wrapper (`open_flamingo/src/flamingo_lm.py`)

`FlamingoLayer` wraps each decoder block with an optional gated cross-attention layer, conditionally applying visual conditioning based on media token locations. The mixin pattern allows retrofitting any HuggingFace causal LM with Flamingo capabilities.

**Flamingo layer structure:**

```6:66:external_repos/flamingo/open_flamingo/src/flamingo_lm.py
class FlamingoLayer(nn.Module):
    def __init__(self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False):
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
    def forward(self, lang_x, attention_mask=None, **decoder_layer_kwargs):
        if self.gated_cross_attn_layer is not None:
            lang_x = self.gated_cross_attn_layer(lang_x, self.vis_x, media_locations=self.media_locations, use_cached_media=self.use_cached_media)
        lang_x = self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        return lang_x
```

**Flamingo LM mixin initialization:**

```83:126:external_repos/flamingo/open_flamingo/src/flamingo_lm.py
def init_flamingo(self, media_token_id, lang_hidden_size, vis_hidden_size, cross_attn_every_n_layers, gradient_checkpointing):
    self.old_decoder_blocks = self._get_decoder_layers()
    self.gated_cross_attn_layers = nn.ModuleList([
        GatedCrossAttentionBlock(dim=lang_hidden_size, dim_visual=vis_hidden_size)
        if (layer_idx + 1) % cross_attn_every_n_layers == 0
        else None
        for layer_idx, _ in enumerate(self._get_decoder_layers())
    ])
    self.init_flamingo_layers(gradient_checkpointing)
```

### Forward Pass & Generation (`open_flamingo/src/flamingo.py`)

The forward pass encodes vision inputs, conditions decoder layers on media locations, and runs the language model. Generation caches visual features and reuses them across autoregressive steps to avoid redundant encoding.

**Forward pass with media conditioning:**

```60:122:external_repos/flamingo/open_flamingo/src/flamingo.py
def forward(self, vision_x, lang_x, attention_mask=None, labels=None, clear_conditioned_layers=True, past_key_values=None, use_cache=False):
    if not self.lang_encoder._use_cached_vision_x:
        self._encode_vision_x(vision_x=vision_x)
        self._condition_media_locations(input_ids=lang_x)
    output = self.lang_encoder(input_ids=lang_x, attention_mask=attention_mask, labels=labels, past_key_values=past_key_values, use_cache=use_cache)
    if clear_conditioned_layers:
        self.lang_encoder.clear_conditioned_layers()
    return output
```

**Generation with cached media:**

```124:175:external_repos/flamingo/open_flamingo/src/flamingo.py
def generate(self, vision_x, lang_x, attention_mask=None, **kwargs):
    num_beams = kwargs.pop("num_beams", 1)
    if num_beams > 1:
        vision_x = vision_x.repeat_interleave(num_beams, dim=0)
    self.lang_encoder._use_cached_vision_x = True
    self._encode_vision_x(vision_x=vision_x)
    output = self.lang_encoder.generate(input_ids=lang_x, attention_mask=attention_mask, eos_token_id=eos_token_id, num_beams=num_beams, **kwargs)
    self.lang_encoder.clear_conditioned_layers()
    self.lang_encoder._use_cached_vision_x = False
    return output
```

### Training Entry Point (`open_flamingo/train/train.py`)

The training script supports distributed training with FSDP, mixed precision, gradient checkpointing, and multi-dataset mixing (MMC4, LAION). Loss multipliers allow balancing contributions from different data sources.

**Training configuration:**

```51:150:external_repos/flamingo/open_flamingo/train/train.py
parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
parser.add_argument("--cross_attn_every_n_layers", type=int, default=1)
parser.add_argument("--batch_size_mmc4", type=int, default=128)
parser.add_argument("--batch_size_laion", type=int, default=128)
parser.add_argument("--loss_multiplier_mmc4", type=float, default=1.0)
parser.add_argument("--loss_multiplier_laion", type=float, default=1.0)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--precision", choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"], default="fp32")
```

## Integration Hooks

- **Vision encoder outputs.** The CLIP encoder returns patch features `[B*T, num_patches, vis_dim]` which are reshaped to `[B, T, F, num_patches, vis_dim]` before Perceiver resampling. The resampler compresses these to `[B, T, num_latents, vis_dim]` where `num_latents=64` by default.^[```177:200:external_repos/flamingo/open_flamingo/src/flamingo.py```]
- **Media caching for evaluation.** The `cache_media()` method pre-encodes images and conditions layers, enabling efficient log-likelihood evaluation on fixed visual contexts without re-encoding on each forward pass.^[```315:331:external_repos/flamingo/open_flamingo/src/flamingo.py```]
- **Cross-attention interval control.** The `cross_attn_every_n_layers` parameter determines how frequently cross-attention layers are inserted. Published models use intervals of 1, 2, or 4, trading off parameter efficiency versus visual conditioning density.^[```64:68:external_repos/flamingo/open_flamingo/src/factory.py```]
- **Modality projection for neuro-omics.** To adapt Flamingo for brain imaging, replace the CLIP encoder with a brain encoder (e.g., BrainLM, BrainMT) and adjust `vis_dim` to match the brain encoder's output dimension. The Perceiver resampler will automatically adapt to the new feature space.^[```48:56:external_repos/flamingo/open_flamingo/src/flamingo.py```]
- **Text generation with brain context.** The generation API accepts interleaved sequences where `<image>` tokens mark brain scan embeddings. The model can generate text descriptions conditioned on brain features, enabling applications like scan summarization or clinical report generation.^[```124:175:external_repos/flamingo/open_flamingo/src/flamingo.py```]


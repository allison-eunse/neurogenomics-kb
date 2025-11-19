# BAGEL Code Walkthrough

> **KB references:** [BAGEL paper note](../generated/kb_curated/papers-md/bagel_2025.md)

## Overview
BAGEL couples a Qwen2-style Mixture-of-Transformer decoder, a SigLIP NaViT encoder, and a latent VAE so a single 7B active-parameter model can interleave text reasoning, visual understanding, and diffusion-style image synthesis. The public release ships with checkpoints, quantized inference paths, training scripts, and evaluation kits spanning understanding, text-to-image, and editing.^[```50:188:external_repos/bagel/README.md```][```153:198:external_repos/bagel/README.md```]

## At-a-Glance
| Architecture | Params / Scale | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| Qwen2 MoT decoder (packed attention, NaiveCache) + SigLIP-NaViT encoder + VAE; modality connectors align latent patches and ViT tokens with the LLM space.^[```27:229:external_repos/bagel/modeling/bagel/bagel.py```] | 7B active / 14B total parameters, trained on trillions of interleaved multimodal tokens; outperforms Qwen2.5-VL and rivals SD3 on benchmarks.^[```50:188:external_repos/bagel/README.md```] | Unified understanding, text-to-image, image editing, and “world-modeling” tasks surfaced through Gradio, CLI scripts, and evaluation benches.^[```50:200:external_repos/bagel/README.md```][```85:151:external_repos/bagel/app.py```] | Packed batches contain text token ids, ViT patches, VAE latents, per-token positions, attention masks, and per-modality loss selectors built by `PackedDataset`.^[```45:305:external_repos/bagel/data/dataset_base.py```] | Training entrypoint wires configurable branches (visual_gen / visual_und), FSDP wrapping, EMA, dataset mixing, and MFU logging.^[```98:870:external_repos/bagel/train/pretrain_unified_navit.py```] | `external_repos/bagel` |

### Environment & Hardware Notes
- Follow the Quick Start: Python 3.10 env, `pip install -r requirements.txt`, then `pip install flash_attn==2.5.8` before downloading checkpoints via `huggingface_hub.snapshot_download`. Modes 1–3 in `app.py` toggle full-precision, NF4, or INT8 pipelines for 12–80 GB GPUs.^[```107:151:external_repos/bagel/README.md```][```25:151:external_repos/bagel/app.py```]
- Training relies on CUDA + NCCL with FSDP; `pretrain_unified_navit.py` auto-detects device TFLOPs for MFU calculation and exposes switches for freezing LLM/ViT/VAE weights, enabling FLEX packing, or running EMA-only resumes.^[```98:418:external_repos/bagel/train/pretrain_unified_navit.py```]
- Inference hyperparameters (`cfg_text_scale`, `cfg_img_scale`, `cfg_interval`, `timestep_shift`, renorm mode, steps) are surfaced both in the README and the Gradio UI so you can script KB experiments consistently.^[```90:151:external_repos/bagel/README.md```][```160:357:external_repos/bagel/app.py```]

## Key Components

### Unified Forward Pass (`modeling/bagel/bagel.py`)
`Bagel` hosts the three branches: (1) language tokens (always on), (2) ViT patches for understanding, and (3) VAE latent patches for generation. It projects modality features into the LLM embedding space, injects learned positional/timestep embeddings, and multiplexes MoT experts via packed index tensors. Losses are computed per-branch (CE for text, Smooth L1/MSE for latents) and returned side-by-side.

```101:229:external_repos/bagel/modeling/bagel/bagel.py
    def forward(..., packed_text_ids, packed_text_indexes, sample_lens, packed_position_ids,
                ..., packed_vit_tokens=None, ..., padded_latent=None, ..., packed_timesteps=None,
                mse_loss_indexes=None):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        ...
        if self.config.visual_und:
            packed_vit_token_embed = self.vit_model(... )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed + vit_pos_embed
        if self.config.visual_gen:
            ... # patchify VAE latents, inject timestep + position, place into packed sequence
            packed_sequence[packed_vae_token_indexes] = packed_latent
        last_hidden_state = self.language_model(..., packed_sequence=packed_sequence, ...)
        if self.config.visual_gen:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            mse = (packed_mse_preds - target[has_mse]) ** 2
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")
        return dict(mse=mse, ce=ce)
```

The same class also defines cache-friendly helpers (`prepare_prompts`, `prepare_vit_images`, `prepare_vae_latent`, `generate_image`, `generate_text`) so both training and inference reuse identical packing rules.^[```232:907:external_repos/bagel/modeling/bagel/bagel.py```]

### PackedDataset & Sequence Plans (`data/dataset_base.py`)
`PackedDataset` streams heterogenous samples, applies conditional dropout (`text_cond_dropout_prob`, etc.), and emits a single packed tensor blob per batch. Each `sequence_plan` step can insert text spans, ViT patches, or VAE tensors, automatically managing BOS/EOS vision tokens, per-split attention modes, and modality-specific losses.^[```45:400:external_repos/bagel/data/dataset_base.py```]

```187:305:external_repos/bagel/data/dataset_base.py
    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            ...
        )
        if len(sequence_status['vae_image_tensors']) > 0:
            data['padded_images'] = padded_images
            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])
```

The `pack_sequence` routine adds `<|im_start|> / <|im_end|>` sentinels, calls `patchify` for ViT patches, records `packed_timesteps` for diffusion supervision, and scales CE loss weights by token length so batches with different numbers of captions remain balanced.^[```306:724:external_repos/bagel/data/dataset_base.py```]

### Training Entry Point (`train/pretrain_unified_navit.py`)
Three dataclasses (`ModelArguments`, `DataArguments`, `TrainingArguments`) expose practically every toggle: source checkpoints, positional interpolation, dropout per modality, packed-data limits, sharding strategy, EMA decay, LR schedule, and loss weights.^[```98:405:external_repos/bagel/train/pretrain_unified_navit.py```] The `main()` routine then:
- Parses args, initializes NCCL, seeds, and W&B logging.
- Loads or restores Qwen2/SigLIP/AE weights (optionally HF checkpoints) and wires them into `BagelConfig`.
- Builds `PackedDataset` via YAML-specified groups, enabling FLEX packing or resume-friendly overflow buffers.
- Wraps the model in FSDP + activation checkpointing, sets up EMA mirrors, optimizer, scheduler, gradient clipping, and MFU telemetry.^[```408:775:external_repos/bagel/train/pretrain_unified_navit.py```]
- Periodically logs CE/MSE/token throughput, tracks dataset sampling state for deterministic resumes, and checkpoints both base + EMA weights alongside optimizer/scheduler state.^[```658:867:external_repos/bagel/train/pretrain_unified_navit.py```]

### Inference Stack (`app.py` + `inferencer.py`)
`app.py` bootstraps configs, shares layers across devices, and lets you choose full precision, NF4, or INT8 quantization before launching the Gradio UI. It wires UI sliders directly to CFG/timestep parameters so experiments match README defaults.^[```25:357:external_repos/bagel/app.py```]

`InterleaveInferencer` encapsulates the streaming generation algorithm: it grows `NaiveCache` instances as you interleave prompts/images, clones contexts for classifier-free guidance, and alternates between textual “thinking” chains and latent diffusion steps.

```22:284:external_repos/bagel/inferencer.py
class InterleaveInferencer:
    def init_gen_context(self):
        return {'kv_lens': [0], 'ropes': [0], 'past_key_values': NaiveCache(...)}

    def update_context_text(...):
        generation_input, kv_lens, ropes = self.model.prepare_prompts(...)
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)

    def update_context_image(...):
        if vae:
            generation_input = self.model.prepare_vae_images(...)
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        if vit:
            generation_input = self.model.prepare_vit_images(...)
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

    def gen_image(...):
        generation_input = self.model.prepare_vae_latent(...)
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(...)
        unpacked_latent = self.model.generate_image(..., cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img_scale, ...)
        return self.decode_image(unpacked_latent[0], image_shape)
```

Understanding vs. generation differ only in whether you keep emitting text (`understanding_output=True`) or call `gen_image` with CFG contexts cloned before the last prompt.^[```207:314:external_repos/bagel/inferencer.py```]

### Packed Qwen2-NaViT Layers (`modeling/bagel/qwen2_navit.py`)
`PackedAttention` and `PackedAttentionMoT` extend Hugging Face’s Qwen2 attention with flash-attention varlen kernels, optional flex-attention (for packed sparse masks), and modality-aware expert routing. `NaiveCache` stores per-layer KV tensors so inference can stream text/image blocks without re-encoding past context.^[```207:379:external_repos/bagel/modeling/bagel/qwen2_navit.py```]

## Integration Hooks
- **Dataset alignment.** `PackedDataset` already surfaces conditional dropout flags and CE-weight scalars; reuse them when aligning neuro-omics modality mixes (e.g., drop imaging tokens to train text-only adapters without rewriting loss code).
- **Modality toggles.** Training arguments `visual_gen`/`visual_und` plus freeze switches make it easy to run ablations (e.g., ViT-only understanding on KB datasets) while reusing the same packed loader.^[```212:405:external_repos/bagel/train/pretrain_unified_navit.py```]
- **CFG introspection.** The inferencer’s CFG contexts are plain dicts holding cloned caches (`cfg_text_precontext`, `cfg_img_precontext`), which means you can intercept them to log per-modality contributions or plug your own KB-guided conditioning signals.^[```120:172:external_repos/bagel/inferencer.py```]



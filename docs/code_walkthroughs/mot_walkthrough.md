# MoT Code Walkthrough

> **KB references:** [MoT paper note](../generated/kb_curated/papers-md/mot_2025.md)

## Overview
Mixture-of-Transformers (MoT) introduces modality-aware sparsity to every non-embedding block so that each modality owns its feed-forward, attention, and normalization routes while still sharing global self-attention. In practice this lets a 7B-text+image MoT hit dense-model quality with only 55.8 % of the FLOPs, extend to speech with 37.2 % of the dense compute, and run multi-branch generation faster on commodity A100s.^[```6:30:external_repos/MoT/README.md```]

## At-a-Glance
| Architecture | Params / FLOPs | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| Attach modality-untied feed-forward + attention experts to an existing transformer, using binary `modality_masks` to route tokens yet keeping shared global attention.^[```75:151:external_repos/MoT/README.md```][```16:151:external_repos/MoT/src/simple_ModalityUntiedAttention.py```] | 7B MoT (text+image) matches dense baselines at 55.8 % FLOPs; 443 M MoT (text+image+speech) hits dense speech quality at 37.2 % FLOPs.^[```15:23:external_repos/MoT/README.md```] | Chameleon (autoregressive text + raster image), Transfusion (text autoregressive + image diffusion), and broader “native multimodal” projects.^[```15:30:external_repos/MoT/README.md```] | Any packed token sequence as long as each token is tagged in `modality_masks`; examples show text/image/speech masks and detoured normalization rules.^[```129:137:external_repos/MoT/README.md```][```87:137:external_repos/MoT/src/simple_ModalityUntiedAttention.py```] | Step-by-step tutorial covering FFN experts, attention experts, and normalization placement so you can graft MoT onto proprietary stacks.^[```75:330:external_repos/MoT/README.md```] | `external_repos/MoT` |

### Environment & Integration Notes
- Designed as a playbook on top of *your* transformer—start from any stack that exposes attention/FFN modules and thread through MoT’s modality-specific replacements.^[```40:71:external_repos/MoT/README.md```]
- Efficient gains hinge on accurate routing masks; the README demonstrates simple boolean lists and emphasises deterministic routing per modality.^[```129:137:external_repos/MoT/README.md```]
- Norm placement matters: either keep residual norms inside the expert modules (preferred) or refactor your `TransformerBlock` to avoid double-normalizing.^[```307:330:external_repos/MoT/README.md```]

## Key Components

### Modality-Untied Feed-Forward (`src/simple_ModalityUntiedFeedForward.py`)
`SimpleModalityUntiedFeedForward` replicates a SiLU-gated MLP per modality, normalizes each expert’s output, then stitches results back into the original token order via `merge_modalities`. Swapping only this block already covers ≈67 % of non-embedding parameters, so most FLOP savings arrive after this step.^[```75:119:external_repos/MoT/README.md```]

```17:60:external_repos/MoT/src/simple_ModalityUntiedFeedForward.py
class SimpleModalityUntiedFeedForward(torch.nn.Module):
    def __init__(..., n_modalities: int = 2):
        ...
        self.local_experts = torch.nn.ModuleList([
            SimpleFeedForward(...)
            for _ in range(self.n_modalities)
        ])
        self.local_experts_ffn_norm = torch.nn.ModuleList(
            [SimpleRMSNorm(dim, eps=1e-5) for _ in range(self.n_modalities)]
        )

    def forward(self, x, modality_masks):
        expert_outputs = []
        for i in range(self.n_modalities):
            expert_input = x[modality_masks[i]]
            expert_output = self.local_experts[i](expert_input)
            expert_output = self.local_experts_ffn_norm[i](expert_output)
            expert_outputs.append(expert_output)
        return merge_modalities(expert_outputs, modality_masks)
```

Because experts only see their modality tokens, you can scale specialization (e.g., text-heavy vs. image-heavy hidden sizes) without perturbing other branches. `SimpleFeedForward` itself is the Lingua-style gated MLP that preserves tensor-parallel friendliness.^[```64:107:external_repos/MoT/src/simple_ModalityUntiedFeedForward.py```]

### Modality-Untied Attention (`src/simple_ModalityUntiedAttention.py`)
The attention module mirrors the FFN pattern: per-modality projections and RMSNorms for Q/K/V/outputs, shared global attention via `torch.nn.MultiheadAttention`, and a final per-modality projection back to the model dimension.^[```141:301:external_repos/MoT/README.md```][```16:151:external_repos/MoT/src/simple_ModalityUntiedAttention.py```]

```16:151:external_repos/MoT/src/simple_ModalityUntiedAttention.py
class SimpleModalityUntiedAttention(torch.nn.Module):
    def __init__(...):
        self.local_experts_wq = self._create_experts(dim, n_heads * head_dim)
        self.local_experts_wk = self._create_experts(dim, n_heads * head_dim)
        self.local_experts_wv = self._create_experts(dim, n_heads * head_dim)
        self.local_experts_wo = self._create_experts(n_heads * head_dim, dim)
        ...
        self.attention_comp = torch.nn.MultiheadAttention(
            head_dim=head_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
```

During `forward`, tokens are first split by mask, projected/normed per modality, concatenated back for standard attention, and finally projected/normed per modality again.^[```86:151:external_repos/MoT/src/simple_ModalityUntiedAttention.py```] Optional QK normalization reshapes tensors to `[*, num_heads, head_dim]` before applying `SimpleRMSNorm`, which keeps the rotary-scaled statistics stable.^[```112:174:external_repos/MoT/src/simple_ModalityUntiedAttention.py```]

### Utility Primitives (`src/utils.py`)
`merge_modalities` reconstructs the packed sequence according to mask order, so expert outputs can be arbitrarily sharded while still producing a contiguous tensor for the residual path. `SimpleRMSNorm` is the Lingua-derived RMSNorm variant used consistently across experts.^[```14:66:external_repos/MoT/src/utils.py```]

```14:34:external_repos/MoT/src/utils.py
def merge_modalities(expert_outputs, modality_masks):
    merged = torch.empty_like(expert_outputs[0])
    for i in range(len(expert_outputs) - 1, -1, -1):
        merged[modality_masks[i]] = expert_outputs[i]
    return merged
```

### Implementation Checklist
- Start from your baseline `TransformerBlock`, then replace FFN/attention classes with the modality-untied versions, ensuring the residual structure still performs `x + module(x)` as shown in the README.^[```307:330:external_repos/MoT/README.md```]
- Provide `modality_masks` for every forward pass. The README’s three-modality example demonstrates boolean masks; in production you can precompute these from tokenizer metadata or image/video region plans.^[```129:137:external_repos/MoT/README.md```]
- Keep norm layers inside the modality-specific modules to avoid double-scaling outputs; only keep block-level norms if your baseline requires them.^[```307:330:external_repos/MoT/README.md```]

## Integration Hooks
- **Routing data from KB assets.** When generating multimodal batches (e.g., fMRI tokens + gene tokens) build boolean masks once per modality and pass them into MoT blocks—only the masks need awareness of modality boundaries.
- **Progressive specialization.** Because experts are independent `nn.ModuleList` entries, you can freeze or reinitialize select modalities while fine-tuning others (useful when only one KB modality changes).
- **FLOP budgeting.** The FLOP savings callouts (55.8 % / 37.2 % / one-third) provide targets for profiling when adapting MoT to new neuro-omics settings.^[```15:30:external_repos/MoT/README.md```]



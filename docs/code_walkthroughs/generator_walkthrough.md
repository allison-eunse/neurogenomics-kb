# GENERator Code Walkthrough

> **KB references:** [Model card](../models/genetics/generator.md) · [Genomics feature spec](../integration/modality_features/genomics.md) · [Integration strategy](../integration/integration_strategy.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
GENERator wraps GPT-style causal decoders (1.2 B and 3 B parameters for both eukaryote and prokaryote checkpoints) with a strict 6-mer tokenizer and long-context optimizations—FlashAttention, Liger kernels, sliding-window decoding—so you can score or generate up to one million base pairs per prompt.^[```5:125:external_repos/generator/README.md```]

## At-a-Glance
| Architecture | Params | Context | Tokenization / Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| HuggingFace `AutoModelForCausalLM` decoder w/ optional ChunkEnsemble Llama heads^[```257:349:external_repos/generator/src/tasks/downstream/fine_tuning.py```][```508:688:external_repos/generator/src/tasks/downstream/sequence_understanding.py```] | 1.2 B & 3 B checkpoints for euk/prok.^[```52:118:external_repos/generator/README.md```] | 1 Mbp prompts via sliding windows + FlashAttention^[```84:99:external_repos/generator/README.md```][```612:667:external_repos/generator/src/tasks/downstream/sequence_understanding.py```] | 6-mer tokenizer; sequences must be multiples of 6, enforced in preprocessing^[```118:125:external_repos/generator/src/tasks/downstream/variant_effect_prediction.py```][```115:235:external_repos/generator/src/tasks/downstream/fine_tuning.py```] | Variant effect scoring, sequence recovery, classification/regression fine-tuning^[```141:406:external_repos/generator/src/tasks/downstream/variant_effect_prediction.py```][```400:687:external_repos/generator/src/tasks/downstream/sequence_understanding.py```] | [github.com/GenerTeam/GENERator](https://github.com/GenerTeam/GENERator) |

### Environment & Hardware Notes
- **Long-context dependencies.** For million-base contexts the README recommends installing the custom kernels explicitly:  
  `pip install liger-kernel`  
  `pip install flash-attn --no-build-isolation`^[```84:89:external_repos/generator/README.md```]
- **Gradient checkpointing flag.** When operating on >10 kbp sequences, the authors enable `model.gradient_checkpointing_enable()` to trade compute for memory.^[```420:424:external_repos/generator/README.md```]

## Key Components

### Tokenizer & Preprocessing (`variant_effect_prediction.py`, `fine_tuning.py`)
The downstream scripts consistently load the HF tokenizer with `trust_remote_code=True`, force pad tokens to EOS if missing, and either truncate or pad every sequence to the nearest 6-mer boundary (`pad_to_multiple_of_six` flag) so the 6-mer BPE never emits `<oov>` tokens.

```151:176:external_repos/generator/src/tasks/downstream/variant_effect_prediction.py
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
...
inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True)
```

```118:125:external_repos/generator/src/tasks/downstream/variant_effect_prediction.py
truncate_length = len(sequence) % 6
if truncate_length > 0:
    sequence = sequence[truncate_length:]
```

```208:243:external_repos/generator/src/tasks/downstream/fine_tuning.py
if pad_to_multiple_of_six:
    remainder = len(seq) % 6
    if remainder != 0:
        pad_len = 6 - remainder
        seq = seq + "A" * pad_len
tokenized = tokenizer(
    sequences,
    truncation=True,
    max_length=max_length,
    add_special_tokens=True,
    padding=False,
)
```

### Positional & Long-Context Handling (`sequence_understanding.py`)
`sequence_understanding.py` either scales RoPE via YaRN or injects sliding-window attention patches so you can extend Llama-based classifiers to >1 M tokens while staying numerically stable.

```596:666:external_repos/generator/src/tasks/downstream/sequence_understanding.py
elif length_extension_mode == "sliding_window":
    config.sliding_window = int(original_model_max_length_for_scaling)
    ...
    def _sliding_llama_forward(...):
        kwargs["sliding_window"] = self.config.sliding_window
        return _orig_forward(...)
    LlamaAttention.forward = _sliding_llama_forward
    attn_implementation = "flash_attention_2"
```

### Backbone Instantiation (`fine_tuning.py`, `sequence_understanding.py`)
Fine-tuning uses `AutoModelForCausalLM` with optional pad ID fixes, while sequence-understanding swaps in `AutoModelForSequenceClassification` or the ChunkEnsemble wrapper to keep a rolling window over million-token sequences.

```257:285:external_repos/generator/src/tasks/downstream/fine_tuning.py
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
)
if model.config.pad_token_id is None and hasattr(model.config, "eos_token_id"):
    model.config.pad_token_id = model.config.eos_token_id
```

```508:593:external_repos/generator/src/tasks/downstream/sequence_understanding.py
class ChunkEnsembleLlamaForSequenceClassification(LlamaPreTrainedModel):
    def forward(...):
        input_ids_chunks = input_ids.unfold(dimension=1, size=self.chunk_size, step=self.stride)
        ...
        chunk_eos_embedding = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths,
        ]
        stacked_embeddings = torch.stack(all_chunk_eos_embeddings, dim=1)
        final_representation = padded_embeddings.view(batch_size, -1)
        logits = self.classifier(final_representation)
```

### Objective & Training Loop (`fine_tuning.py`)
The script wraps everything in `transformers.Trainer` with `DataCollatorForLanguageModeling` (`mlm=False`) so causal LM losses line up with the HF training stack and distributed options (DeepSpeed, FSDP) set via CLI.

```351:390:external_repos/generator/src/tasks/downstream/fine_tuning.py
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
trainer.train()
```

### Inference Helpers (`variant_effect_prediction.py`)
Variant effect prediction shards ClinVar sequences across GPUs, caches logits, and computes per-base probabilities by summing over all tokens starting with ref/alt characters. This utility powers the headline ClinVar AUROC numbers.

```201:290:external_repos/generator/src/tasks/downstream/variant_effect_prediction.py
def compute_logits_parallel(...):
    num_gpus = torch.cuda.device_count()
    shards.append({'shard_id': i, 'sequences_data': sequences_data[start_idx:end_idx], ...})
    with ctx.Pool(processes=num_gpus) as pool:
        results = list(pool.imap(compute_logits_shard, args_list))
```

```292:383:external_repos/generator/src/tasks/downstream/variant_effect_prediction.py
def parallel_compute_probabilities(...):
    vocab = tokenizer.get_vocab()
    char_indices = get_char_indices(vocab)
    results = list(pool.imap(compute_prob, args_list, chunksize=chunksize))
    p_ref, p_alt = zip(*results)
```

### Embedding Extraction (`sequence_understanding.py`)
ChunkEnsemble accumulates the EOS vector from each sliding chunk, pads/truncates them to a fixed count, and flattens into a `[B, max_chunks * hidden]` representation before the classifier head—exactly what you can reuse for downstream alignment.

```446:505:external_repos/generator/src/tasks/downstream/sequence_understanding.py
stacked_embeddings = torch.stack(all_chunk_eos_embeddings, dim=1)
num_padding_chunks = self.max_chunks - stacked_embeddings.shape[1]
...
final_representation = padded_embeddings.view(batch_size, -1)
logits = self.classifier(final_representation)
```

### Sequence Constraints (`variant_effect_prediction.py`, `fine_tuning.py`)
Both inference and training enforce the 6-mer constraint by trimming or padding raw strings and, for dataset preprocessing, only accepting columns named `sequence`, `seq`, `dna_sequence`, etc., so you cannot silently feed invalid tokens.

```208:244:external_repos/generator/src/tasks/downstream/fine_tuning.py
if "sequence" in examples:
    sequences = examples["sequence"]
elif "seq" in examples:
    sequences = examples["seq"]
...
else:
    raise ValueError("No sequence column found in dataset.")
```

## Integration Hooks (Genetics ↔ Brain)

- **Embedding shapes.** GENERator decoders yield `[B, L_tokens, hidden]` tensors; ChunkEnsemble condenses them into `[B, max_chunks, hidden]` before flattening to `[B, max_chunks * hidden]`. You can stop just before the final classifier to grab the stacked embeddings for pooling.^[```446:505:external_repos/generator/src/tasks/downstream/sequence_understanding.py```]
- **Pooling strategies.** Use mean pooling along the chunk dimension for overall sequence summaries, max pooling for motif emphasis, or take the final chunk (equivalent to autoregressive “last token”). Because chunk embeddings correspond to non-overlapping windows, pooling behaves like low-resolution downsampling.
- **Projection to shared latent.** After pooling to `[B, H]` (H≈1536 for the 1.2 B model), apply a projector to map into the same 512-D space used by your brain encoder:

```python
import torch.nn as nn

class GeneratorProjector(nn.Module):
    def __init__(self, input_dim=1536, output_dim=512, dropout=0.1):
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

- **Normalization.** LayerNorm (as in the projector above) keeps token-averaged embeddings comparable to fMRI CLS tokens (BrainLM/SwiFT), especially before cosine-similarity objectives.
- **Sequence hygiene.** Reuse the `pad_to_multiple_of_six` logic or `ensure_6mer_compatible` helper whenever you extract embeddings outside the packaged scripts; otherwise, HF will inject `<oov>` tokens that shift chunk boundaries and misalign pooling.^[```118:125:external_repos/generator/src/tasks/downstream/variant_effect_prediction.py```]
- **Memory tips.** For million-token prompts, lean on ChunkEnsemble (`length_extension_mode="chunk_ensemble"`) or sliding-window RoPE to avoid editing HF internals; both paths keep per-chunk lengths manageable and let FlashAttention v2 handle the heavy lifting.^[```566:666:external_repos/generator/src/tasks/downstream/sequence_understanding.py```]

Following these steps yields `[B, 512]` genetic embeddings that can be concatenated with or contrastively aligned against brain-model outputs such as BrainLM CLS vectors or BrainMT/SwiFT pooled features.

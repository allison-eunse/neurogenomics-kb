# DNABERT-2 Code Walkthrough

> **KB references:** [Model card](../models/genetics/dnabert2.md) · [Genomics feature spec](../integration/modality_features/genomics.md) · [Integration strategy](../integration/integration_strategy.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
DNABERT‑2 swaps classic k-mer vocabularies for a DNA BPE tokenizer backed by ALiBi positional bias in a 117 M parameter BERT encoder, and the repo focuses on supervised fine-tuning utilities for the Genome Understanding Evaluation benchmark.^[```30:110:external_repos/dnabert2/README.md```]

## At-a-Glance
| Architecture | Params | Context | Tokenization / Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| BERT encoder with ALiBi bias + BPE tokenizer^[```30:110:external_repos/dnabert2/README.md```] | 117 M (`zhihan1996/DNABERT-2-117M`)^[```30:94:external_repos/dnabert2/README.md```] | 512 tokens (default `model_max_length`)^[```43:99:external_repos/dnabert2/finetune/train.py```] | Optional k-mer preprocessing plus HF tokenizer/padding^[```79:185:external_repos/dnabert2/finetune/train.py```] | Supervised fine-tuning, LoRA adapters, k-mer augmentation, evaluation^[```33:304:external_repos/dnabert2/finetune/train.py```] | [github.com/Zhihan1996/DNABERT2](https://github.com/Zhihan1996/DNABERT2) |

### Environment & Hardware Notes
- **Exact install command.** The README instructs you to run `python3 -m pip install -r requirements.txt` (after cloning) to pull in the exact transformer/torch stack used for the GUE benchmark; no additional kernels are required because contexts stay at 512 tokens.^[```54:68:external_repos/dnabert2/README.md```]

## Key Components

### Tokenizer & Dataset Pipeline (`finetune/train.py`)
`SupervisedDataset` ingests CSVs, optionally swaps sequences for cached k-mer strings, and tokenizes them with the HF tokenizer, honoring `model_max_length` and storing label counts for dynamic classifier heads.

```79:160:external_repos/dnabert2/finetune/train.py
class SupervisedDataset(Dataset):
    def __init__(..., kmer: int = -1):
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        ...
        if kmer != -1:
            texts = load_or_generate_kmer(data_path, texts, kmer)
        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        self.input_ids = output["input_ids"]
        self.labels = labels
```

### K-mer Utilities (`finetune/train.py`)
Helper functions create or cache k-mer corpora for experiments that compare raw BPE tokens vs explicit k-mer inputs.

```79:109:external_repos/dnabert2/finetune/train.py
def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])
...
def load_or_generate_kmer(...):
    if os.path.exists(kmer_path):
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            json.dump(kmer, f)
```

### Backbone & LoRA (`finetune/train.py`)
The trainer script loads `AutoModelForSequenceClassification`, optionally wraps LoRA adapters (with user-specified target modules), and defers the loss/eval loop to `transformers.Trainer`.

```235:304:external_repos/dnabert2/finetune/train.py
tokenizer = transformers.AutoTokenizer.from_pretrained(..., model_max_length=training_args.model_max_length, ...)
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    num_labels=train_dataset.num_labels,
    trust_remote_code=True,
)
if model_args.use_lora:
    lora_config = LoraConfig(...)
    model = get_peft_model(model, lora_config)
trainer = transformers.Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator)
trainer.train()
```

### Embedding Access (README)
You can grab `[CLS]` embeddings straight from `AutoModel`, as shown in the README example, to feed into multimodal projectors.

```98:110:external_repos/dnabert2/README.md
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
hidden_states = model(inputs)[0] # [1, sequence_length, 768]
cls_embedding = hidden_states[:, 0, :]
```

## Integration Hooks (Genetics ↔ Brain)

- **Embedding shape.** HF models return `last_hidden_state` shaped `[B, L_tokens, 768]`. Use the `[:, 0, :]` CLS token for classification-style features or mean-pool across tokens for regression-style features.^[```98:110:external_repos/dnabert2/README.md```]
- **Pooling choices.** CLS pooling mirrors the pretraining objective; mean pooling smooths noise, and max pooling highlights motifs. You can concatenate these pooled views before projection if you need richer features.
- **Projection to shared latent.** Map `[B, 768]` vectors into a 512-D brain space:

```python
import torch.nn as nn

class DNABERT2Projector(nn.Module):
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

- **Normalization.** LayerNorm (as above) or z-scoring per batch keeps embeddings compatible with BrainLM/BrainMT outputs that also end in norm layers.
- **Sequence hygiene.** Stick to uppercase A/C/G/T before tokenization; when enabling k-mer augmentation, remember the transformation shortens the sequence by `k-1`, so adjust `model_max_length` in `TrainingArguments` to avoid truncation.^[```79:109:external_repos/dnabert2/finetune/train.py```][```43:52:external_repos/dnabert2/finetune/train.py```]

With these steps you can turn DNABERT‑2 outputs into `[B, 512]` embeddings ready for concatenation or contrastive alignment with BrainLM CLS tokens, BrainMT CLS vectors, or SwiFT pooled representations.

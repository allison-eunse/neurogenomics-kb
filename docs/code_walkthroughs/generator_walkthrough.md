# GENERator Code Walkthrough

## Overview

GENERator is a long-context generative genomic foundation model based on transformer decoders, achieving state-of-the-art performance across genomic benchmarks while excelling at sequence generation tasks. It uses 6-mer tokenization and supports contexts up to **1 million base pairs** with sliding window attention.

**Key Features:**
- **Architecture**: GPT-style decoder with sliding-window attention
- **Context Length**: Up to 1,000,000 bp (with FlashAttention + Liger Kernel)
- **Tokenization**: 6-mer BPE (requires sequence lengths divisible by 6)
- **Model Sizes**: 1.2B and 3B parameters
- **Training Data**: RefSeq database (eukaryote and prokaryote variants)
- **Optimization**: FlashAttention, Liger Kernel, DeepSpeed, FSDP support

## Repository Structure

```
generator/
├── src/
│   └── tasks/
│       └── downstream/
│           ├── variant_effect_prediction.py
│           ├── sequence_recovery.py
│           ├── sequence_understanding.py
│           └── fine_tuning.py
├── configs/
│   ├── variant_effect_prediction.json
│   ├── sequence_recovery.json
│   ├── sequence_understanding.yaml
│   └── fine_tuning.yaml
├── requirements.txt
└── README.md
```

## Architecture

### 1. Core Model Components

**Decoder Architecture:**
```python
# GENERator uses standard GPT-style decoder with:
# - Sliding-window causal attention
# - 6-mer tokenization
# - FlashAttention for long contexts

# Typical usage:
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "GenerTeam/GENERator-v2-eukaryote-1.2b-base",
    torch_dtype="bfloat16"
)
tokenizer = AutoTokenizer.from_pretrained(
    "GenerTeam/GENERator-v2-eukaryote-1.2b-base"
)
```

**Critical 6-mer Requirement:**
```python
# Input sequences MUST be multiples of 6 bp
# Bad: "ACGTACG" (7 bp) → will append <oov> token
# Good: "ACGTAC" (6 bp) or "ACGTACGTAC" (12 bp)

def ensure_6mer_compatible(seq: str) -> str:
    """Pad or truncate sequence to multiple of 6."""
    remainder = len(seq) % 6
    if remainder != 0:
        # Left padding with 'A'
        seq = 'A' * (6 - remainder) + seq
    return seq
```

### 2. Sliding-Window Attention

**Long-Context Strategy:**
```python
# GENERator extends context to 1M bp using:
# 1. Sliding-window attention (local context)
# 2. FlashAttention (memory-efficient)
# 3. Liger Kernel (optimized operations)

# Install requirements:
# pip install liger-kernel
# pip install flash-attn --no-build-isolation

# Model automatically uses sliding-window attention
# for sequences exceeding base context length
```

### 3. Model Variants

**Available Checkpoints:**

| Model | Parameters | Data | Category | Context |
|-------|-----------|------|----------|---------|
| `GENERator-v2-eukaryote-1.2b-base` | 1.2B | 422B | Eukaryote | 1M bp |
| `GENERator-v2-eukaryote-3b-base` | 3B | 422B | Eukaryote | 1M bp |
| `GENERator-v2-prokaryote-1.2b-base` | 1.2B | 515B | Prokaryote | 1M bp |
| `GENERator-v2-prokaryote-3b-base` | 3B | 515B | Prokaryote | 1M bp |

## Downstream Tasks

### 1. Variant Effect Prediction

**Purpose**: Predict impact of genetic variants (ClinVar benchmark)

**Usage:**
```bash
# FP32 (default)
python src/tasks/downstream/variant_effect_prediction.py

# BF16 for faster inference (recommended)
python src/tasks/downstream/variant_effect_prediction.py --bf16
```

**Integration:**
```python
# In your pipeline:
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "GenerTeam/GENERator-v2-eukaryote-1.2b-base",
    torch_dtype=torch.bfloat16
).cuda()

def predict_variant_effect(ref_seq: str, alt_seq: str):
    """Compare likelihoods of reference vs alternative sequence."""
    # Ensure 6-mer compatibility
    ref_seq = ensure_6mer_compatible(ref_seq)
    alt_seq = ensure_6mer_compatible(alt_seq)
    
    # Compute log-likelihoods
    ref_tokens = tokenizer(ref_seq, return_tensors="pt").to("cuda")
    alt_tokens = tokenizer(alt_seq, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        ref_loss = model(**ref_tokens, labels=ref_tokens.input_ids).loss
        alt_loss = model(**alt_tokens, labels=alt_tokens.input_ids).loss
    
    return ref_loss - alt_loss  # Positive = deleterious
```

### 2. Sequence Recovery

**Purpose**: Evaluate model's ability to recover DNA sequences

**Usage:**
```bash
# FP32
python src/tasks/downstream/sequence_recovery.py

# BF16 (recommended)
python src/tasks/downstream/sequence_recovery.py --bf16
```

**Benchmark**: Uses [GenerTeam/sequence-recovery](https://huggingface.co/datasets/GenerTeam/sequence-recovery) dataset

### 3. Sequence Understanding

**Purpose**: Classification/regression on genomic sequences

**Supported Benchmarks:**
- **Gener Tasks**: Gene classification, taxonomic classification
- **NT Tasks**: Histone marks (H2AFZ, H3K27ac, etc.)
- **Genomic Benchmarks**: Human/worm species classification
- **DeepSTARR**: Enhancer activity prediction (regression)

**Usage:**
```bash
# Single GPU
python src/tasks/downstream/sequence_understanding.py \
    --model_name GenerTeam/GENERator-eukaryote-1.2b-base \
    --dataset_name GenerTeam/gener-tasks \
    --subset_name gene_classification \
    --batch_size 8 \
    --problem_type single_label_classification \
    --main_metrics accuracy \
    --bf16

# Multi-GPU (DDP)
torchrun --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    src/tasks/downstream/sequence_understanding.py \
    --model_name GenerTeam/GENERator-eukaryote-1.2b-base \
    --dataset_name GenerTeam/gener-tasks \
    --subset_name gene_classification \
    --batch_size 8 \
    --bf16

# DeepSpeed (multi-node)
torchrun --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=master_node:29500 \
    src/tasks/downstream/sequence_understanding.py \
    --distributed_type deepspeed \
    --bf16
```

**Regression Example:**
```bash
# Enhancer activity prediction
python src/tasks/downstream/sequence_understanding.py \
    --model_name GenerTeam/GENERator-eukaryote-1.2b-base \
    --dataset_name GenerTeam/DeepSTARR-enhancer-activity \
    --problem_type regression \
    --main_metrics pearson \
    --batch_size 16 \
    --bf16
```

### 4. Fine-Tuning for Generation

**Purpose**: Fine-tune for specific sequence generation tasks

**Datasets:**
- DeepSTARR Enhancer sequences
- Histone coding DNA sequences (CDS)
- Cytochrome P450 CDS

**Usage:**
```bash
# Single GPU
python src/tasks/downstream/fine_tuning.py \
    --model_name GenerTeam/GENERator-eukaryote-1.2b-base \
    --dataset_name GenerTeam/DeepSTARR-enhancer-activity \
    --batch_size 4 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --bf16

# Multi-GPU with FSDP
torchrun --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    src/tasks/downstream/fine_tuning.py \
    --model_name GenerTeam/GENERator-eukaryote-3b-base \
    --dataset_name GenerTeam/histone-cds \
    --batch_size 2 \
    --distributed_type fsdp \
    --bf16
```

## Embeddings Extraction

**Use Case**: Extract sequence embeddings for downstream ML tasks

**Approach 1: Last Hidden State**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "GenerTeam/GENERator-v2-eukaryote-1.2b-base",
    torch_dtype=torch.bfloat16,
    output_hidden_states=True
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "GenerTeam/GENERator-v2-eukaryote-1.2b-base"
)

def extract_embeddings(sequence: str, pooling: str = "mean"):
    """Extract embeddings from GENERator.
    
    Args:
        sequence: DNA sequence (will be padded to multiple of 6)
        pooling: 'mean', 'max', or 'last'
    """
    # Ensure 6-mer compatibility
    seq = ensure_6mer_compatible(sequence)
    
    inputs = tokenizer(seq, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get last hidden state: [batch, seq_len, hidden_dim]
    hidden_states = outputs.hidden_states[-1]
    
    if pooling == "mean":
        embedding = hidden_states.mean(dim=1)
    elif pooling == "max":
        embedding = hidden_states.max(dim=1).values
    elif pooling == "last":
        embedding = hidden_states[:, -1, :]
    
    return embedding.cpu().numpy()
```

**Approach 2: Batch Extraction for Large Datasets**
```python
from torch.utils.data import DataLoader
import numpy as np

def batch_extract_embeddings(sequences: list[str], batch_size: int = 16):
    """Extract embeddings for multiple sequences efficiently."""
    embeddings = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        # Ensure 6-mer compatibility
        batch = [ensure_6mer_compatible(seq) for seq in batch]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512 * 6  # 512 6-mers
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Mean pooling
            hidden = outputs.hidden_states[-1]
            batch_emb = hidden.mean(dim=1).cpu().numpy()
        
        embeddings.append(batch_emb)
    
    return np.vstack(embeddings)
```

## Training Configuration

**Recommended Hyperparameters:**
```yaml
# For fine-tuning on downstream tasks
learning_rate: 5e-5
batch_size: 4-8 (per GPU)
num_train_epochs: 10
warmup_ratio: 0.1
weight_decay: 0.01
gradient_accumulation_steps: 4
max_grad_norm: 1.0
lr_scheduler_type: cosine
bf16: true

# For long sequences (>10k bp)
gradient_checkpointing: true
sliding_window_size: 4096
flash_attention: true
```

## Integration with Multimodal Pipelines

### Genetic-Brain Alignment

**Goal**: Extract genetic embeddings for fusion with brain fMRI data

**Pipeline:**
```python
# 1. Extract genetic embeddings from genomic sequences
genetic_embeddings = batch_extract_embeddings(
    genomic_sequences,
    batch_size=16
)

# 2. Project to shared embedding space
from torch import nn

class GeneticProjector(nn.Module):
    def __init__(self, input_dim=1536, output_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.proj(x)

# 3. Combine with brain embeddings (from BrainLM/SwiFT)
genetic_proj = GeneticProjector().cuda()
shared_genetic = genetic_proj(torch.tensor(genetic_embeddings).cuda())

# 4. Align with contrastive learning (see integration cards)
```

## Performance Benchmarks

**Genomic Benchmarks (Classification Accuracy):**
- Human vs Worm: **98.2%**
- Promoter Classification: **94.5%**
- Enhancer Detection: **91.3%**

**Gener Tasks:**
- Gene Classification: **96.1%**
- Taxonomic Classification: **97.8%**

**Sequence Recovery (Next K-mer Prediction):**
- Eukaryote: **92.4% @ 512bp**
- Prokaryote: **94.1% @ 512bp**

**ClinVar (Variant Effect Prediction):**
- AUROC: **0.89**
- AUPRC: **0.85**

## Best Practices

### 1. Input Sequence Preparation
```python
# Always validate 6-mer compatibility
def prepare_sequence(seq: str) -> str:
    """Prepare sequence for GENERator."""
    # Remove non-ACGT characters
    seq = ''.join(c for c in seq.upper() if c in 'ACGT')
    # Pad to multiple of 6
    seq = ensure_6mer_compatible(seq)
    return seq
```

### 2. Memory Management for Long Sequences
```python
# Enable gradient checkpointing for >100k bp
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast
with autocast(dtype=torch.bfloat16):
    outputs = model(**inputs)
```

### 3. Distributed Training
```bash
# Use DeepSpeed ZeRO-3 for 3B models
accelerate config  # Select DeepSpeed ZeRO-3
accelerate launch src/tasks/downstream/fine_tuning.py --bf16
```

## Common Issues & Solutions

### Issue 1: `<oov>` Tokens in Generation
**Cause**: Input sequence not divisible by 6  
**Solution**: Always pad/truncate to multiples of 6

### Issue 2: OOM on Long Sequences
**Cause**: Long context without optimization  
**Solution**: Enable gradient checkpointing + FlashAttention

```bash
pip install flash-attn --no-build-isolation
pip install liger-kernel
```

### Issue 3: Poor Generation Quality
**Cause**: Default sampling parameters  
**Solution**: Adjust `top_k`, `top_p`, `temperature`

```python
# Better generation
output = model.generate(
    **inputs,
    max_new_tokens=600,
    top_k=4,           # Nucleus sampling
    temperature=0.9,   # Diversity
    do_sample=True,
    repetition_penalty=1.2
)
```

## Critical Files Reference

**Task Scripts:**
- `src/tasks/downstream/variant_effect_prediction.py` - ClinVar evaluation
- `src/tasks/downstream/sequence_recovery.py` - Next k-mer prediction
- `src/tasks/downstream/sequence_understanding.py` - Classification/regression
- `src/tasks/downstream/fine_tuning.py` - Causal LM fine-tuning

**Configs:**
- `configs/variant_effect_prediction.json` - VEP hyperparameters
- `configs/sequence_understanding.yaml` - Downstream task configs

## Links & Resources

- **Repository**: https://github.com/GenerTeam/GENERator
- **Models**: https://huggingface.co/GenerTeam
- **Paper**: https://arxiv.org/abs/2502.07272
- **Datasets**: 
  - https://huggingface.co/datasets/GenerTeam/gener-tasks
  - https://huggingface.co/datasets/GenerTeam/sequence-recovery
  - https://huggingface.co/datasets/GenerTeam/DeepSTARR-enhancer-activity

## Next Steps

1. **For Understanding Tasks**: Start with `sequence_understanding.py` on Gener Tasks
2. **For Generation**: Fine-tune with `fine_tuning.py` on domain-specific sequences
3. **For Variant Effects**: Run `variant_effect_prediction.py` on ClinVar
4. **For Embeddings**: Use extraction code above for multimodal integration
5. **For Multimodal Fusion**: See `kb/integration_cards/ukb_genetics_brain_alignment.yaml`

---

*Last Updated: 2025-11-15*  
*Model Card: `kb/model_cards/generator.yaml`*  
*Integration: `kb/integration_cards/genetics_embeddings_pipeline.yaml`*


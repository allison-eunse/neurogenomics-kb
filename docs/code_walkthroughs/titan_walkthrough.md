# TITAN Code Walkthrough

> **KB references:** Model card (pending) · [Integration strategy](../integration/integration_strategy.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
TITAN (Transformer-based pathology Image and Text Alignment Network) aggregates CONCH v1.5 patch embeddings into slide-level representations using a Transformer encoder aligned with pathology-report text via a CoCa-style captioning loss. The public release focuses on the slide & text encoders (decoder weights removed) and ships Hugging Face `trust_remote_code` modules for `encode_slide_from_patch_features`, plus fine-tuning and evaluation scaffolding for tasks like TCGA-OT linear probes and zero-shot slide retrieval.^[```1:110:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/README.md```][```25:111:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/finetune.py```]

## At-a-Glance
| Architecture | Params | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| CONCH v1.5 patch encoder → TITAN slide Transformer with learned spatial grids + vision-language alignment (CoCa-inspired).^[```9:110:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/README.md```] | Hugging Face `MahmoodLab/TITAN` exposes 768-d slide embeddings; fine-tuning head typically 768→num_classes MLP; training loop enables FP16/bfloat16 autocast + cosine sched.^[```25:50:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/finetune.py```][```137:199:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/finetune.py```] | Pretrained on 335,645 WSIs + 182K real reports + 423K synthetic captions; patch grids derived from Level-0 coordinates with `patch_size_lv0` (512 @20×, 1024 @40×).^[```14:99:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/README.md```] | `.h5` files containing `features` (N×768) and `coords` (N×2) arrays, plus `patch_size_level0` attribute; sample downloads provided via Hugging Face hub.^[```82:99:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/README.md```] | `finetune.py` (CustomSequential + cosine LR + GradScaler), `eval_linear_probe.py` (logistic regression), `titan/utils.py` (metrics/bootstrap), and TCGA config/prompts for zero-shot classification.^[```25:339:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/finetune.py```][```2:75:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/eval_linear_probe.py```][```13:154:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/utils.py```][```1:120:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/datasets/config_tcga-ot.yaml```] | [github.com/mahmoodlab/TITAN](https://github.com/mahmoodlab/TITAN) |

### Environment & Hardware Notes
- **Install & deps.** Clone the repo, `conda create -n titan python=3.9`, activate, upgrade pip, then `pip install -e .` (installs PyTorch 2.0.1, timm 1.0.3, h5py, sklearn, transformers 4.46).^[```25:42:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/README.md```][```5:18:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/setup.py```]
- **Model access.** Run `huggingface_hub.login()` before calling `AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)`; this downloads slide + text encoders as well as CONCH v1.5 patch encoder helpers.^[```43:59:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/README.md```]
- **Feature extraction options.** Either (a) use TRIDENT/CLAM pipelines for CONCH features or (b) load shared `.h5` demo files and call `model.encode_slide_from_patch_features(features, coords, patch_size_lv0)` directly; set `patch_size_lv0` to 1024 (40×) or 512 (20×) per slide metadata.^[```63:98:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/README.md```]

## Key Components

### Custom Sequential Wrapper (`titan/finetune.py`)
Fine-tuning wraps the frozen TITAN backbone with a lightweight MLP head. `CustomSequential` simply forwards tensor tuples into `encode_slide_from_patch_features`, then feeds slide embeddings into the classification head. `create_mlp` helps instantiate arbitrary hidden stacks.

```25:50:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/finetune.py
class CustomSequential(nn.Module):
    def __init__(self, model, mlp):
        super(CustomSequential, self).__init__()
        self.model = model
        self.mlp = mlp

    def forward(self, *args, **kwargs):
        x = self.model.encode_slide_from_patch_features(*args, **kwargs)
        x = self.mlp(x)
        return x


def create_mlp(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.0, out_dim=None, end_with_fc=True):
    layers = []
    if len(hid_dims) > 0:
        for hid_dim in hid_dims:
            layers.append(nn.Linear(in_dim, hid_dim))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            in_dim = hid_dim
    layers.append(nn.Linear(in_dim, out_dim))
    if not end_with_fc:
        layers.append(act)
        layers.append(nn.Dropout(dropout))
    mlp = nn.Sequential(*layers)
    return mlp
```

### Training Loop & Scheduler (`titan/finetune.py`)
`train` constructs two optimizer parameter groups (bias/LayerNorm vs. rest), applies cosine LR with warmup, leverages `torch.cuda.amp.GradScaler`, and evaluates on a validation loader with early stopping that tracks the best weights.

```137:199:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/finetune.py
    model.train()
    fp16_scaler = torch.cuda.amp.GradScaler()
    step = 0
    early_stopping = EarlyStopping(patience=2, verbose=True)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        preds_all = []
        targets_all = []
        total_train_loss = 0
        for features, coords, patch_size_lv0, label in tqdm(train_loader):
            lr_scheduler(step)
            features = features.to(device)
            coords = coords.to(device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(features, coords, patch_size_lv0.to(device), **kwargs)
                loss = loss_fn(logits, label.to(device))
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            optimizer.zero_grad()

            preds_all.append(logits.argmax(1).cpu().numpy())
            targets_all.append(label.numpy())
            step += 1
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        ...
            tqdm.write(f"epoch {epoch}, bacc: {np.round(bacc, 4):.4f}, bacc_val: {np.round(bacc_val, 4):.4f}, loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
```

### Linear Probe Evaluation (`titan/eval_linear_probe.py`)
For frozen-feature experiments, `train_and_evaluate_logistic_regression_with_val` sweeps log-spaced `C`, fits `LogisticRegression`, and reports metrics (balanced accuracy, Cohen’s κ, AUROC) via shared utilities.

```2:75:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/eval_linear_probe.py
def train_and_evaluate_logistic_regression_with_val(train_data, train_labels, val_data, val_labels, test_data, test_labels, log_spaced_values=None, max_iter=500):
    seed_torch(torch.device('cpu'), 0)
    
    metric_dict = {
        'bacc': 'balanced_accuracy',
        'kappa': 'cohen_kappa_score',
        'auroc': 'roc_auc_score',
    }
    
    if log_spaced_values is None:
        log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
    
    best_score = -float('inf')
    best_C = None
    logistic_reg_final = None
    for log2_coeff in tqdm(log_spaced_values, desc="Finding best C"):
        ...
        logistic_reg = LogisticRegression(
            C=1/log2_coeff,
            fit_intercept=True,
            max_iter=max_iter,
            random_state=0,
            solver="lbfgs",
        )
        logistic_reg.fit(train_data, train_labels)
        ...
    eval_metrics = get_eval_metrics(test_labels, test_preds, test_probs, roc_kwargs=roc_kwargs)
```

### Metrics, Bootstrap & Zero-Shot Templates (`titan/utils.py`)
`get_eval_metrics` reports accuracy/balanced accuracy/kappa/weighted F1 (+ AUROC/log-loss when probs are provided). The module also seeds reproducibility, merges dictionaries, and defines zero-shot text templates for class prompts.

```13:89:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/utils.py
# zeroshot prompt templates
TEMPLATES = [
    "CLASSNAME.",
    "an image of CLASSNAME.",
    ...
    "CLASSNAME is identified.",
]

def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    unique_classes: Optional[List[int]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    unique_classes = unique_classes if unique_classes is not None else np.unique(targets_all)
    bacc = balanced_accuracy_score(targets_all, preds_all) if len(targets_all) > 1 else 0
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    nw_kappa = cohen_kappa_score(targets_all, preds_all, weights="linear")
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0, labels=unique_classes)

    eval_metrics = {
        f"{prefix}/acc": acc,
        f"{prefix}/bacc": bacc,
        f"{prefix}/kappa": kappa,
        f"{prefix}/nw_kappa": nw_kappa,
        f"{prefix}/weighted_f1": cls_rep["weighted avg"]["f1-score"],
    }
```

### TCGA-OT Configuration & Prompts (`datasets/config_tcga-ot.yaml`)
The YAML describes label counts, OncoTree codes, class-specific textual prompts (supporting zero-shot CLIP-like scoring), and dataset metadata. Integrate these prompts with TITAN’s text encoder or other VLMs for retrieval tasks.

```1:60:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/datasets/config_tcga-ot.yaml
aggregation: slide
cohorts:
- TCGA
ext_cohorts: []
folds: 1
label_dict:
  AASTR: 0
  ACC: 1
  AOAST: 2
  ASTR: 3
  BLCA: 4
  CCRCC: 5
  CESC: 6
  CHRCC: 7
  COAD: 8
  DDLS: 9
  DSTAD: 10
  ESCA: 11
  ESCC: 12
  GBM: 13
...
prompts:
  AASTR:
  - anaplastic astrocytoma
  - astrocytoma, anaplastic
  - grade III astrocytoma
  - AASTR
  ACC:
  - adrenocortical carcinoma
  - adrenal cortical carcinoma
  - adrenal cortex carcinoma
  - ACC
```

## Integration Hooks (Pathology ↔ Multimodal KB)
- **Slide embeddings as shared latent vectors.** `encode_slide_from_patch_features` returns 768-d tensors suitable for concatenation with genetic or clinical embeddings before populating KB integration cards; store patch grids alongside metadata for reproducibility.^[```25:99:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/finetune.py```]
- **Prompt templates for zero-shot mapping.** The `TEMPLATES` list and TCGA label prompts can seed cross-modal retrieval experiments or provide natural-language anchors for other models (e.g., LLaVA-Med) to align with TITAN outputs.^[```13:39:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/utils.py```][```1:120:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/datasets/config_tcga-ot.yaml```]
- **Metrics + bootstrap interoperability.** Use `get_eval_metrics` / `bootstrap` outputs to populate KB evaluation summaries, ensuring consistent confidence intervals across modalities when comparing TITAN features against alternative encoders.^[```13:154:/Users/allison/Projects/neuro-omics-kb/external_repos/titan/titan/utils.py```]


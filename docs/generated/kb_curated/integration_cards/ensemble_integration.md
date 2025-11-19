---
title: Ensemble Integration (EI) — Integration Guidance Card
status: active
updated: 2025-11-19
tags: [integration, late-fusion, ensembles, interpretability]
---

# Ensemble Integration (EI)

**Source:** Li et al. (2022), Bioinformatics Advances  
**Type:** Late fusion integration pattern  
**Best for:** Heterogeneous feature spaces, small-to-medium datasets, interpretable multimodal fusion

---

## Problem It Solves

**Challenge:** How to integrate heterogeneous biomedical data modalities (genetics, brain imaging, clinical data) that have very different structures, scales, and semantics without losing modality-specific signals.

**Solution:** Ensemble Integration (EI) treats each modality as a first-class citizen by:
1. Training specialized models per modality with appropriate inductive biases
2. Combining modality predictions via heterogeneous ensembles (stacking, selection, averaging)
3. Providing interpretable feature rankings across all modalities

**Why traditional approaches fail:**
- **Early integration** (concatenate raw features) → loses modality-specific structure
- **Intermediate integration** (shared embeddings) → emphasizes agreement, suppresses modality-unique signals
- **Single-model approaches** → can't adapt architecture to each modality's semantics

---

## Core Mechanics

### 1. Modality-Specific Model Training

Train diverse base classifiers per modality using algorithms matched to data structure:

```python
# Genetics: sequence/graph data
gene_models = {
    'lr': LogisticRegression(C=0.01),
    'rf': RandomForestClassifier(n_estimators=100),
    'svm': SVC(kernel='rbf', probability=True),
    'xgb': XGBClassifier(max_depth=5)
}

# Brain: spatial/temporal features
brain_models = {
    'lr': LogisticRegression(C=0.1),
    'gbdt': LightGBMClassifier(num_leaves=31),
    'knn': KNeighborsClassifier(n_neighbors=5)
}
```

**Key insight:** Different modalities benefit from different inductive biases (trees for genetics, neighbors for imaging).

### 2. Late Fusion Strategies

**Simple averaging:**
```python
ensemble_pred = (gene_pred_proba + brain_pred_proba) / 2
```

**Ensemble selection** (Li et al. method):
- Iteratively add models that improve validation performance
- Greedy forward selection with replacement
- Automatically weights models by contribution

**Stacking with meta-learner:**
```python
# Stack predictions from all base models
meta_features = np.hstack([gene_preds, brain_preds])
meta_model = LogisticRegression()
meta_model.fit(meta_features, y_train)
```

⚠️ **Critical:** Stacking must be fold-proper to avoid leakage—train meta-learner only on out-of-fold base predictions.

### 3. Interpretability via Feature Ranking

**Cross-modality feature importance:**
1. For each base model, extract feature importances (coefficients, SHAP values, permutation importance)
2. Aggregate via ensemble weights
3. Rank features across all modalities

**Result:** Identify which genes AND which brain regions drive predictions, weighted by ensemble contribution.

---

## When to Use

✅ **Use Ensemble Integration when:**
- Modalities have **heterogeneous structures** (sequences, images, graphs, tables)
- Dataset size is **small-to-medium** (< 10k samples)
- **Missing data** is common (not all subjects have all modalities)
- **Interpretability** is critical for clinical translation
- **Baseline comparisons** needed (per-modality vs. fusion performance)
- **Computing resources** are limited (no end-to-end training needed)

✅ **Particularly well-suited for:**
- Gene-brain-behavior prediction in neuro-omics
- Multi-site cohort integration with batch effects
- Clinical decision support requiring feature-level explanations
- Research settings exploring which modalities contribute most

---

## When to Defer

⚠️ **Defer to more advanced methods when:**
- Modalities have **strong cross-modal dependencies** (e.g., paired image-text)
- **Large datasets** available (> 100k samples) enabling end-to-end joint training
- **Real-time deployment** required (ensemble overhead too high)
- **Shared representations** needed (e.g., cross-modal retrieval tasks)

⚠️ **Consider alternatives:**
- **Two-tower contrastive** if need aligned embedding space for retrieval
- **Early fusion** if modalities naturally align (e.g., multi-view same subject)
- **Mixture-of-Experts** if need learned routing by modality

---

## Adoption in Our Neuro-Omics Pipeline

### Current Implementation

**Per-modality models:**
- **Genetics:** LR + LightGBM on Caduceus/DNABERT-2 embeddings (512-D)
- **Brain:** LR + LightGBM on BrainLM/SwiFT embeddings (512-D)
- **Fusion:** Ensemble selection or simple stacking with LR meta-learner

**Workflow:**
```bash
# 1. Extract embeddings per modality
python extract_gene_embeddings.py --model caduceus --out gene_emb.npy
python extract_brain_embeddings.py --model brainlm --out brain_emb.npy

# 2. Train per-modality models
python train_per_modality.py --modality gene --models lr,gbdt
python train_per_modality.py --modality brain --models lr,gbdt

# 3. Ensemble integration
python ensemble_fusion.py --strategy stacking --meta_model lr
```

**Evaluation metrics:**
- Per-modality AUROC/AUPRC
- Fusion AUROC/AUPRC
- DeLong test: Fusion vs. max(Gene, Brain)
- Feature importance rankings

### Integration with ARPA-H BOM

EI provides the **baseline late fusion** in our escalation strategy:

```
1. Ensemble Integration (baseline) ✓ Current
    ↓ If fusion wins (p < 0.05)
2. Two-tower contrastive
    ↓ If gains plateau
3. EI stacking with hub tokens
    ↓ Last resort
4. Full early fusion (TAPE-style)
```

**Why start with EI:**
- Establishes **whether fusion helps at all** before complex architectures
- Provides **interpretable baseline** for regulatory/clinical validation
- Enables **per-modality ablations** to identify which data types matter
- **Computationally cheap** to iterate on cohort definitions and confounds

---

## Caveats and Best Practices

### ⚠️ Leakage Prevention

**Problem:** If meta-learner sees in-fold predictions, it overfits to noise.

**Solution:** Always use out-of-fold predictions for stacking:
```python
# WRONG: Train meta-learner on training predictions
meta_model.fit(base_preds_train, y_train)  # Leakage!

# RIGHT: Train meta-learner on out-of-fold predictions
oof_preds = cross_val_predict(base_model, X_train, y_train, cv=5)
meta_model.fit(oof_preds, y_train)
```

### ⚠️ Confound Control

**Problem:** Batch effects (site, scanner) can dominate modality signals.

**Solution:** Residualize **before** training base models:
```python
# Per modality, residualize confounds
gene_emb_residual = residualize(gene_emb, confounds=['age', 'sex', 'site', 'PCs'])
brain_emb_residual = residualize(brain_emb, confounds=['age', 'sex', 'site', 'FD'])
```

### ⚠️ Meta-Learner Simplicity

**Problem:** Complex meta-learners (deep NNs) can overfit ensemble predictions.

**Solution:** Use simple meta-learners (LR, Ridge) unless >10k samples:
```python
# Prefer: Regularized logistic regression
meta_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)

# Avoid: Unless large N
meta_model = MLPClassifier(hidden_layers=(64, 32))  # Risk overfitting
```

### ⚠️ Missing Modality Handling

**Problem:** Not all subjects have both gene and brain data.

**Solution:** Train modality-specific fallback models:
```python
if has_both_modalities(subject):
    pred = ensemble_model.predict(gene_emb, brain_emb)
elif has_gene_only(subject):
    pred = gene_model.predict(gene_emb)
elif has_brain_only(subject):
    pred = brain_model.predict(brain_emb)
```

---

## Practical Implementation Guide

### Step 1: Choose Base Models

**Diversity is key** — use algorithms with different inductive biases:

| Modality | Recommended Models | Rationale |
|----------|-------------------|-----------|
| Genetics (sequence) | LR, XGBoost, SVM-RBF | Linear + trees + kernels |
| Brain (imaging) | LR, LightGBM, k-NN | Linear + trees + locality |
| Behavior (tabular) | LR, RandomForest, Ridge | Simple + robust to correlation |

### Step 2: Train with Proper CV

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Per-modality training with OOF predictions
for modality in ['gene', 'brain']:
    oof_preds = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        
        model.fit(X_train, y_train)
        oof_preds.append(model.predict_proba(X_val)[:, 1])
    
    # Save OOF predictions for meta-learner
    save_oof_predictions(modality, np.concatenate(oof_preds))
```

### Step 3: Meta-Learner Training

```python
# Load OOF predictions
gene_oof = load_oof_predictions('gene')
brain_oof = load_oof_predictions('brain')

# Stack into meta-features
meta_X = np.column_stack([gene_oof, brain_oof])

# Train meta-learner
meta_model = LogisticRegression(C=1.0, max_iter=1000)
meta_model.fit(meta_X, y_train)

# Evaluate on held-out test set
test_preds = np.column_stack([
    gene_model.predict_proba(gene_test)[:, 1],
    brain_model.predict_proba(brain_test)[:, 1]
])
test_auc = roc_auc_score(y_test, meta_model.predict_proba(test_preds)[:, 1])
```

### Step 4: Feature Interpretation

```python
import shap

# Compute SHAP values for each base model
gene_shap = shap.TreeExplainer(gene_model).shap_values(gene_emb)
brain_shap = shap.TreeExplainer(brain_model).shap_values(brain_emb)

# Weight by ensemble contribution (meta-learner coefficients)
gene_weight = np.abs(meta_model.coef_[0][0])
brain_weight = np.abs(meta_model.coef_[0][1])

# Aggregate feature importance
weighted_gene_importance = gene_shap.mean(axis=0) * gene_weight
weighted_brain_importance = brain_shap.mean(axis=0) * brain_weight

# Rank across all features
all_importance = np.concatenate([weighted_gene_importance, weighted_brain_importance])
top_features = np.argsort(all_importance)[::-1][:20]
```

---

## Reference Materials

**Primary paper:**
- [Ensemble Integration (Li 2022)](../papers-md/ensemble_integration_li2022.md) — Full paper summary

**Related KB resources:**
- [Integration Strategy](../../integration/integration_strategy.md) — Overall fusion approach
- [Design Patterns](../../integration/design_patterns.md) — Pattern 1: Late Fusion
- [CCA + Permutation Recipe](../../integration/analysis_recipes/cca_permutation.md) — Statistical testing
- [Prediction Baselines](../../integration/analysis_recipes/prediction_baselines.md) — Comparison protocol

**Integration cards:**
- [Oncology Multimodal Review](oncology_multimodal_review.md) — Broader fusion taxonomy

**Model documentation:**
- [Genetics Models](../../models/genetics/) — Gene embedding extraction
- [Brain Models](../../models/brain/) — Brain embedding extraction

**Experiment configs:**
- `configs/experiments/02_prediction_baselines.yaml` — EI implementation template

---

## Next Steps in Our Pipeline

1. **Baseline EI implementation** — LR + GBDT per modality with stacking meta-learner
2. **Per-modality ablations** — Which modality contributes most? (Gene vs. Brain vs. Fusion)
3. **Feature interpretation** — Identify top predictive genes and brain regions
4. **Cohort extension** — Test EI on Cha Hospital developmental cohort
5. **Escalation decision** — If fusion wins significantly, move to two-tower contrastive

**Success criteria for escalation:**
- Fusion AUROC > max(Gene, Brain) with p < 0.05 (DeLong test)
- Gains observed across multiple phenotypes (cognitive, diagnostic)
- Stable across cross-validation folds (not driven by outliers)

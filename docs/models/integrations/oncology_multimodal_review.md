---
title: Oncology Multimodal Review — Integration Principles
status: active
updated: 2025-11-19
tags: [integration, principles, review, fusion-taxonomy, deep-learning]
---

# Oncology Multimodal Review — Integration Principles

**Source:** Waqas et al. (2024), Frontiers in Artificial Intelligence  
**Type:** Comprehensive review of multimodal fusion strategies  
**Scope:** Deep learning architectures (GNNs, Transformers) for cancer data integration  
**Relevance:** General principles applicable to neuro-omics gene-brain-behavior fusion

---

## Problem Context

**Domain:** Cancer/oncology multimodal learning (imaging + omics + clinical)  
**Relevance to Neuro-Omics:** While focused on cancer, the review's taxonomy and cautions apply directly to gene-brain integration, where we face similar challenges:

| Oncology Challenge | Neuro-Omics Equivalent |
|-------------------|------------------------|
| Radiology + histopathology + genomics | fMRI + sMRI + WES/WGS |
| Multi-site scanner heterogeneity | UK Biobank + HCP + Cha Hospital sites |
| Missing modalities per patient | Incomplete genetic/imaging coverage |
| Class imbalance (rare cancers) | Rare neurological disorders |
| Interpretability for clinicians | Explainability for genetic counseling |

---

## Key Taxonomy: Fusion Strategies

The review categorizes multimodal integration into three main patterns, with trade-offs for each:

### 1. Early Fusion

**Mechanism:** Concatenate raw or lightly processed features from all modalities before feeding into a single model.

```
Gene sequences    → |
                    | → Concatenate → Single Model → Prediction
Brain volumes     → |
Clinical features → |
```

**Advantages:**
- Simplest to implement
- Single model to train and deploy
- Can capture low-level cross-modal interactions

**Disadvantages:**
- **Heterogeneous semantics:** Hard to align sequences, images, and tables
- **Dimensionality explosion:** Concatenating all features creates huge input spaces
- **Loss of modality structure:** Throws away spatial (imaging) and sequential (genomics) structure
- **Dominance by one modality:** High-dimensional modality can drown out others

**When to use:**
- Modalities are **already aligned** (e.g., multi-view same organ)
- Small feature sets (< 1k features total)
- Quick prototyping to check if fusion helps at all

**When to avoid:**
- Large dimensional mismatch (e.g., 100 genes vs. 100k voxels)
- Heterogeneous data types (sequences vs. images vs. tables)
- **Our default:** Avoid for gene-brain fusion unless late fusion fails

### 2. Intermediate Fusion (Joint Embeddings)

**Mechanism:** Extract modality-specific embeddings, then fuse via learned projections or attention before final prediction.

```
Gene FM     → gene_embed [512-D]   → |
                                     | → Fusion layer (attention/concat) → Classifier
Brain FM    → brain_embed [512-D]  → |
```

**Variants:**
- **Concatenation + MLP:** `concat(gene_emb, brain_emb) → MLP → pred`
- **Cross-attention:** Query one modality with keys/values from another
- **Gated fusion:** Learn modality weights dynamically per sample
- **Graph fusion (GNN):** Model relationships among genes, brain regions, patients

**Advantages:**
- Preserves modality-specific structure in embeddings
- Can learn cross-modal interactions
- Balances early vs. late fusion

**Disadvantages:**
- **Over-smoothing (GNNs):** Deep GNN layers blur modality boundaries
- **Alignment challenges:** Requires paired data for joint training
- **Hyperparameter sensitivity:** Fusion architecture choices critical

**When to use:**
- Have large paired dataset (>10k samples)
- Need learned cross-modal interactions
- Foundation models provide good embeddings already

**Our escalation path:** Move here if late fusion (EI) demonstrates significant gains.

### 3. Late Fusion (Decision-Level)

**Mechanism:** Train separate models per modality, combine predictions via ensembles or simple averaging.

```
Gene model  → gene_pred [proba]  → |
                                    | → Ensemble (avg/stack) → Final pred
Brain model → brain_pred [proba] → |
```

**Advantages:**
- **Respects modality-specific semantics:** Each model uses appropriate architecture
- **Handles missing data gracefully:** Can fall back to single-modality predictions
- **Interpretable:** Can isolate which modality contributes most
- **Computationally efficient:** No joint training overhead

**Disadvantages:**
- **Limited cross-modal learning:** Models trained independently
- **Suboptimal if strong dependencies:** May miss synergistic interactions

**When to use:**
- **Baseline comparisons:** Always start here
- Heterogeneous modalities with different structures
- Small-to-medium datasets (< 10k samples)
- Interpretability critical for clinical translation

**Our default:** This is Pattern 1 in our [Design Patterns](../../integration/design_patterns/).

---

## Practical Cautions for Neuro-Omics

The review highlights six major pitfalls—all directly applicable to our gene-brain integration:

### 1. Heterogeneous Semantics

**Problem:** Genetics (sequences, graphs), brain imaging (spatial volumes), behavior (tables) have fundamentally different structures.

**Oncology example:** Mixing WSI pixels with RNA-seq counts  
**Neuro-omics risk:** Naïvely concatenating gene embeddings with fMRI voxels

**Mitigation:**
- Use **modality-specific foundation models** (Caduceus for genes, BrainLM for fMRI)
- Project to **common dimensionality** (512-D) before fusion
- **Normalize scales** (z-score per modality)

### 2. Alignment Across Scales

**Problem:** Modalities capture biology at different resolutions (molecular → cellular → tissue → organ).

**Oncology example:** Aligning genomic variants (bp-level) with CT scans (mm-level)  
**Neuro-omics risk:** Gene-level variants vs. voxel-level BOLD signal

**Mitigation:**
- **Aggregate to common level:** Gene-level embeddings ↔ Subject-level brain embeddings
- **Hierarchical fusion:** Match genomic pathways to brain networks
- **Avoid pixel-level alignment:** Use pretrained FMs to handle within-modality structure

### 3. Missing Modalities

**Problem:** Not all subjects have complete data (incomplete genetic sequencing, missing scans).

**Oncology example:** Some patients lack genomic profiling due to sample quality  
**Neuro-omics risk:** UKB has imaging for subset; HCP lacks genetics

**Mitigation:**
- **Late fusion with fallbacks:** Train per-modality models that work independently
- **Imputation:** Use modality-specific imputation (e.g., gene expression imputation from SNPs)
- **Multi-task learning:** Share representations where data overlaps
- **Avoid requiring all modalities:** Don't force early fusion that drops incomplete samples

### 4. Over-Smoothing (GNNs)

**Problem:** Deep graph neural networks blur distinctions between node types, losing modality-specific signals.

**Oncology example:** Patient-gene-image GNN collapses to uniform representations  
**Neuro-omics risk:** Gene-brain-subject heterogeneous graph loses modality boundaries

**Mitigation:**
- **Limit GNN depth:** Use 2-3 layers maximum
- **Modality-aware message passing:** Separate parameters for gene-gene vs. gene-brain edges
- **Prefer late fusion:** Avoid GNNs unless strong relational structure justifies complexity

### 5. Confounds and Batch Effects

**Problem:** Site, scanner, sequencing platform, demographics can dominate biological signal.

**Oncology example:** Multi-site TCGA data has strong batch effects  
**Neuro-omics risk:** UK Biobank imaging sites, genetic ancestry PCs, scanner upgrades

**Mitigation:**
- **Residualize before fusion:** Remove age, sex, site, motion (FD), genetic PCs per modality
- **Harmonization:** ComBat for imaging, surrogate variable analysis for genomics
- **Stratified CV:** Ensure train/val/test splits balance sites
- **Site-unlearning:** Adversarial debiasing if residualization insufficient

⚠️ **Critical:** Always residualize **before** extracting embeddings from FMs.

### 6. Data Leakage

**Problem:** Information from test set bleeds into training, inflating performance estimates.

**Oncology example:** Normalizing across train+test before split  
**Neuro-omics risks:**
- Fitting PCA on full dataset before CV split
- Selecting features based on full-cohort statistics
- Using in-fold predictions for stacking meta-learner

**Mitigation:**
- **Fit preprocessing only on training folds:** z-score, PCA, harmonization per fold
- **Out-of-fold predictions for stacking:** Use `cross_val_predict` for meta-learner
- **Strict fold boundaries:** No subject overlap between train/val/test
- **Validation gates:** Use `scripts/manage_kb.py` to check for leakage

---

## Practices We Adopt

Based on the review's recommendations and our neuro-omics context:

### ✅ Default Late Fusion (Pattern 1)

**Rationale:** Heterogeneous gene-brain modalities, small-to-medium cohorts (UKB ~40k imaging, HCP ~1k)

**Implementation:**
- Per-modality FMs: Caduceus/DNABERT-2 (genetics), BrainLM/SwiFT (brain)
- Project to 512-D per modality
- Ensemble Integration (see [EI card](../ensemble_integration/))
- Compare: Gene-only vs. Brain-only vs. Fusion

### ✅ Rigorous Confound Control

**Per modality, residualize:**
- **Genetics:** Age, sex, genetic PCs (ancestry), cohort
- **Brain:** Age, sex, site/scanner, motion (mean FD), intracranial volume (ICV)
- **Behavior:** Age, sex, SES, education

**Harmonization when needed:**
- **Brain imaging:** ComBat for site effects, MURD for T1/T2
- **Genetics:** Surrogate variable analysis for batch effects

### ✅ Stratified Cross-Validation

- **5-fold StratifiedKFold:** Preserve outcome class balance
- **Site stratification:** Ensure each fold has all sites represented
- **Subject-level split:** No leakage via related individuals (kinship matrix)

### ✅ Proper Significance Testing

- **DeLong test:** Compare Fusion AUROC vs. max(Gene, Brain) AUROC
- **Permutation tests:** Null distributions for CCA canonical correlations
- **Bootstrap CIs:** 95% confidence intervals on performance metrics
- **Bonferroni correction:** Adjust for multiple phenotype tests

### ✅ Interpretability-First

- **Feature importance:** SHAP values per modality, aggregated by ensemble weights
- **Ablation studies:** Which modality contributes most? (Gene vs. Brain)
- **Biological validation:** Top genes/regions mapped to known pathways/networks

---

## What We Defer (and Why)

The review documents advanced architectures (deep GNNs, Transformers, foundation models), but we defer these until baselines justify the complexity:

### 🚫 Early Fusion (Immediate Concatenation)

**Why defer:** Heterogeneous semantics (sequences vs. images), dimensionality mismatch  
**When reconsider:** If late fusion completely fails (unlikely)

### 🚫 Deep Graph Neural Networks

**Why defer:** Over-smoothing risk, limited interpretability, requires careful graph construction  
**When reconsider:** If explicit gene-brain relationship modeling justified (e.g., gene-pathway-brain-network hierarchies)

### 🚫 End-to-End Joint Training

**Why defer:** Requires large paired datasets (>50k), computationally expensive, risk of overfitting  
**When reconsider:** If move to two-tower contrastive (Pattern 2) and have sufficient data

### 🚫 Foundation Model Fine-Tuning

**Why defer:** Gene/brain FMs pretrained on massive corpora; fine-tuning risks losing generalization  
**When reconsider:** If task-specific adaptation needed (e.g., pediatric-specific brain FM for Cha cohort)

---

## Direct Implications for Our Project

### Phase 1: Late Fusion Baselines (Current)

**Goal:** Establish whether gene-brain fusion helps at all

**Methods:**
- Ensemble Integration (EI) with LR + GBDT per modality
- CCA + permutation tests for cross-modal structure
- Per-modality vs. fusion AUROC comparisons

**Success criteria:** Fusion significantly outperforms max(Gene, Brain) on cognitive/diagnostic tasks

### Phase 2: Escalation to Intermediate Fusion (If Phase 1 succeeds)

**Goal:** Learn cross-modal interactions if late fusion wins

**Methods:**
- Two-tower contrastive alignment (Pattern 2)
- Cross-attention fusion
- EI stacking with learned modality weights

**Trigger:** Fusion AUROC > max(Gene, Brain) + 0.03, p < 0.05 across multiple phenotypes

### Phase 3: Advanced Architectures (Long-term)

**Goal:** Unified Brain-Omics Model (BOM) for ARPA-H vision

**Methods:**
- Mixture-of-Transformers (MoT) with modality-specific experts
- Hub tokens / TAPE-style early fusion
- LLM as semantic bridge for gene-brain-behavior reasoning

**Trigger:** Intermediate fusion demonstrates consistent gains, large paired dataset (>50k) available

---

## Architecture Decision Tree

```
Start: Gene + Brain data
    ↓
Q1: Does fusion help at all?
    → Run EI baseline (LR + GBDT per modality, stacking)
    → Compare Fusion vs. max(Gene, Brain) AUROC
    
    If NO (Fusion ≤ max + 0.01):
        → Stop: Use best single-modality model
        → Document: Modalities independent for this phenotype
    
    If YES (Fusion > max + 0.03, p < 0.05):
        ↓
Q2: Do we have large paired dataset (>10k)?
    
    If NO:
        → Continue with EI
        → Add interpretability: Which modality contributes?
    
    If YES:
        ↓
Q3: Are modalities strongly coupled?
        → Run CCA + permutation: Significant canonical correlations?
        
        If NO:
            → Continue with EI (late fusion optimal)
        
        If YES:
            ↓
            → Escalate to Two-Tower Contrastive (Pattern 2)
            → If gains plateau, consider EI stacking with hub tokens
            → If still improving, consider early fusion (Pattern 5)
```

---

## Reference Materials

**Primary paper:**
- [Oncology Multimodal Review (Waqas 2024)](../../generated/kb_curated/papers-md/oncology_multimodal_waqas2024/) — Full paper summary

**Related integration cards:**
- [Ensemble Integration (EI)](ensemble_integration/) — Late fusion implementation details

**KB integration guides:**
- [Integration Strategy](../../integration/integration_strategy/) — Overall fusion approach
- [Design Patterns](../../integration/design_patterns/) — 5 patterns with escalation criteria
- [Multimodal Architectures](../../integration/multimodal_architectures/) — Clinical/multimodal model patterns

**Analysis recipes:**
- [CCA + Permutation](../../integration/analysis_recipes/cca_permutation/) — Cross-modal structure testing
- [Prediction Baselines](../../integration/analysis_recipes/prediction_baselines/) — Fusion vs. single-modality comparison
- [Partial Correlations](../../integration/analysis_recipes/partial_correlations/) — Confound-aware associations

**Data governance:**
- [Governance & QC](../../data/governance_qc/) — Quality control protocols
- [UKB Data Map](../../data/ukb_data_map/) — Cohort definitions and confounds

**Model documentation:**
- [Genetics Models](../../models/genetics/) — Foundation models for gene embeddings
- [Brain Models](../../models/brain/) — Foundation models for brain embeddings
- [Multimodal Models](../../models/multimodal/) — Examples of fusion architectures

---

## Key Takeaways

1. **Start simple:** Late fusion (EI) is the principled baseline—establish whether fusion helps before complexity
2. **Control confounds:** Residualize age, sex, site, motion **before** fusion—batch effects dominate biology
3. **Respect heterogeneity:** Don't force early fusion on sequences + images—use modality-specific FMs
4. **Handle missing data:** Late fusion supports per-modality fallbacks—don't drop incomplete samples
5. **Test significance:** DeLong tests and permutations quantify fusion gains—avoid over-interpreting noise
6. **Prioritize interpretability:** Feature rankings and ablations guide biological discovery and clinical trust
7. **Escalate conditionally:** Only move to intermediate/early fusion if late fusion wins and data supports it

**Bottom line:** The oncology review validates our default late fusion strategy and provides clear criteria for when (and when not) to escalate to more complex architectures.

---
title: Modality Features — Genomics
status: ready
updated: 2025-11-16
---

# Genomics Features

Embedding hygiene
- RC-equivariant models (Caduceus/Evo/Evo2): compute forward and reverse-complement embeddings; average.
- DNABERT-2/GENERator:
  - RC at nucleotide level first; then tokenize (k-mer/BPE) deterministically.
  - Maintain k-mer frame; prefer mean pooling if CLS unstable.

Pooling and subject aggregation
- Mean or CLS pooling to token → exon → gene → subject aggregation as per study design.
- Standardize features, residualize covariates, project to 512.

Attribution
- LOGO: ΔAUC with Wilcoxon across folds; FDR control.

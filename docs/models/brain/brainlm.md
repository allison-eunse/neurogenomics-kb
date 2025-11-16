---
title: BrainLM — Model Card (light)
status: draft
updated: 2025-11-16
---

# BrainLM

Purpose
- Self-supervised masked autoencoding for fMRI parcels; produces subject/session embeddings.

Key inductive biases
- Masked reconstruction across parcels/time; large multi-site pretraining encourages site-invariant structure.

Inputs/constraints
- Parcel time series; consistent preprocessing; fixed or harmonized TR.

Embedding strategy
- Mean pooling over latent tokens to subject vector; project to 512 for downstream.

Strengths / limitations
- Strong linear probing; robust across sites; requires fMRI preprocessing pipeline.

Representative results
- Generalization across UKB/HCP; good reconstruction R².

Implications for us
- Candidate for Option B fMRI embeddings; late integration with gene features.

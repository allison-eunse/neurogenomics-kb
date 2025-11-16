---
title: Brain-JEPA — Model Card (light)
status: draft
updated: 2025-11-16
---

# Brain-JEPA

Purpose
- Joint-Embedding Predictive Architecture for fMRI latents; emphasizes semantic consistency.

Inductive biases
- Functional gradient positioning; spatiotemporal masking (Cross-ROI/Time).

Embedding strategy
- Token latents → pooled subject vectors; compact projections to 512 for downstream.

Notes
- Better linear-probe performance than MAE in reports; engineering heavier than FC.

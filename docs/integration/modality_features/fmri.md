---
title: Modality Features — fMRI
status: ready
updated: 2025-11-16
---

# fMRI Features

Option A (fast)
- Atlas (e.g., Schaefer-400) → time series → FC matrix → Fisher z → vectorize upper triangle → PCA to 100–256 → standardize/residualize (include FD, site) → project to 512 if needed.

Option B (FMs later)
- BrainLM/Brain-JEPA/Harmony subject embeddings.
- TR/site handling: Harmony TAPE for heterogeneous TRs; hub tokens for multimodal fusion.

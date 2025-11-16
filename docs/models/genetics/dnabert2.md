---
title: DNABERT-2 — Model Card (light)
status: draft
updated: 2025-11-16
---

# DNABERT-2

Tokenization
- BPE/k-mer hybrids; ensure RC is applied before tokenization; maintain frame.

Pooling
- Mean or CLS; verify linear-probe stability.

Notes
- Not strictly RC-equivariant; averaging forward/RC stabilizes.

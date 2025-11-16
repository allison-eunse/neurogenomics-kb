---
title: GENERator — Model Card (light)
status: draft
updated: 2025-11-16
---

# GENERator

Purpose
- Long-context generative DNA LM.

Hygiene
- Tokenizer sensitivity (k-mer/BPE) and RC handling; apply RC at nucleotide level pre-tokenization.

Implications
- Be explicit about tokenization scheme; test RC averaging stability.

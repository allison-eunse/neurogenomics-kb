# Model Card Template

> **Reminder:** Include a license note such as “This walkthrough references `<repo>` under `<license>`,” plus links to the repo, latest tag/commit, the associated `kb/model_cards/*.yaml`, and the generated PDF export (stored under `artifacts/pdf/code_walkthroughs/` or released assets).

## Metadata
- **Model name:** `<Friendly alias>`
- **External repo:** `<URL>`
- **Latest tag / commit:** `<tag-or-sha>`
- **License:** `<e.g., Apache-2.0>`
- **Model card YAML:** ``kb/model_cards/<id>.yaml``
- **Download PDF:** `<artifact link>`

## Purpose & Scope
- What the model is designed for (e.g., DNA sequence embeddings, rs-fMRI latents, multimodal fusion).
- Intended tasks / datasets; out-of-scope uses.

## Architecture & Inductive Biases
- Brief bullets on backbone, depth/width, notable blocks (e.g., JEPA, MAE, Swin, RC-equivariant BiMamba).
- Any modality-specific design (hub tokens, TAPE, gradient positional encodings).

## Tokenization & Input Constraints
- Tokenizer type (character, k-mer, BPE, voxel patches, ROI tensors).
- Context length / TR windows / voxel grids.
- Required preprocessing (sorting genes, TR normalization, motion censoring).

## Pooling & Subject-Level Embeddings
- How to pool token embeddings (mean, CLS, hub-token average).
- RC handling (average forward/RC) or TR alignment notes.
- Aggregation to subject/session level (e.g., exon → gene → subject).

## Training Data & Checkpoints
- Source datasets, sample counts, preprocessing assumptions.
- Checkpoint paths / download links; expected placement under `external_repos/<repo>/checkpoints`.

## Recommended Embedding Procedure
1. Preprocess inputs (tokenization, z-score, residualization).
2. Run encoder with key flags (e.g., gradient checkpointing, mask configs).
3. Pool + project to 512-D (PCA or tiny MLP) with fold-aware fitting.
4. Persist embeddings + covariates for downstream analyses.

## Integration Hooks
- 512-D projector config (`Linear → GELU → Dropout → Linear → LayerNorm`).
- Late-fusion guidance (LR/GBDT baselines, CCA requirements).
- LOGO / attribution tips (for gene models).

## Strengths & Limitations
- Where the model excels (e.g., long-context DNA, heterogeneous TRs).
- Known failure modes (site sensitivity, large VRAM needs, tokenizer quirks).

## References & Links
- Paper citation(s) with DOI/arXiv.
- Repo link, docs, issue tracker.
- Related KB cards (dataset, integration strategy, analysis recipes).


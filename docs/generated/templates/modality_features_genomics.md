# Modality Features: Genomics

## Source Inputs
- Candidate gene/exon list: `<link to table or YAML>`
- Sequence extraction pipeline: `<script / notebook>`
- Reference genome build: `<GRCh38/...>`

## Tokenization & RC Hygiene
- Specify tokenizer per model (character, k-mer length, BPE vocab).
- For RC-equivariant models (Caduceus, Evo/Evo2): compute forward + reverse-complement embeddings and average.
- For k-mer/BPE models (DNABERT-2, GENERator): RC raw sequence first, then re-tokenize to keep alignment; log tokenizer version and casing.
- Enforce consistent padding/truncation rules; document how Ns/ambiguous bases handled.

## Embedding Procedure
1. Load pretrained checkpoints (paths under `external_repos/<repo>/checkpoints`).
2. Apply pooling (mean or CLS) per exon/gene; annotate what constitutes a subject-level aggregation (e.g., mean across exons).
3. Normalize embeddings with fold-specific scaler.
4. Project to 512-D via PCA or small MLP; store projector weights.

## Covariates & Residualization
- Always residualize against age, sex, ancestry PCs (≥10), sequencing batch, and other study-specific covariates.
- Document covariate sources and residualization scripts; output to `artifacts/generated/confounds/`.

## LOGO / Attribution
- Outline Leave-One-Gene-Out loop (nested CV, ΔAUC, Wilcoxon signed-rank, Benjamini–Hochberg FDR).
- Capture tables of ΔAUC (mean ± SD) with p/q-values.
- Store plots showing top contributing genes.

## Integration Hooks
- Provide projector config reference (e.g., `projectors/genomics_pca512.yaml`).
- Note how to combine with brain embeddings for late fusion (concatenate 512-D vectors).
- Mention any pathways/enrichment analyses tied to CCA or LOGO outputs.

## References
- Caduceus, DNABERT-2, Evo2, GENERator papers.
- Internal notebooks for sequence extraction, tokenizer QA, RC sanity checks.


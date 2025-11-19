# Genetics Foundation Models

This section documents the **DNA sequence foundation models** used for genomic representation learning in the Neuro-Omics KB. These models are used to extract gene-level embeddings from raw genomic sequences (DNA/RNA) for downstream integration with brain imaging, behavioral phenotypes, and clinical outcomes.

## Overview

All genetics FMs documented here:

- **Operate on nucleotide sequences** (A, C, G, T) rather than pre-computed variant calls or SNP arrays
- **Support gene-level embeddings** via forward/reverse-complement (RC) averaging and pooling strategies
- **Enable interpretability** through attribution methods like LOGO ΔAUC
- **Are pretrained on large genomic corpora** (human reference genomes, multi-species datasets, or RefSeq)

## Model registry

| Model | Architecture | Key feature | Integration role |
|-------|-------------|-------------|------------------|
| [Caduceus](caduceus.md) | Mamba (BiMamba) + RC-equivariance | Strand-robust, efficient long-context | Primary gene encoder for UK Biobank WES |
| [DNABERT-2](dnabert2.md) | BERT (multi-species pretraining) | BPE tokenization, cross-species transfer | Alternative gene encoder; comparison baseline |
| [Evo 2](evo2.md) | StripedHyena (1M context) | Ultra-long-range dependencies | Exploratory; regulatory region capture |
| [GENERator](generator.md) | Generative 6-mer LM | Generative modeling, sequence design | Reference for generative vs discriminative tradeoffs |

## Usage workflow

1. **Extract sequences** from reference genome (hg38) for target genes
2. **Tokenize** using model-specific vocabularies (k-mers, BPE, or single-nucleotide)
3. **Embed** forward and reverse-complement sequences
4. **Pool** to gene-level representation (mean/CLS depending on model)
5. **Project** to 512-D for cross-modal alignment with brain embeddings

## Key considerations

### RC-equivariance
DNA has no inherent directionality; models like **Caduceus** enforce BiMamba RC-equivariance to avoid strand bias. For non-equivariant models, manually average forward and RC embeddings.

### Variant handling
Foundation models operate on **reference sequences by default**. To incorporate subject-specific variants:

- Patch reference with VCF alleles
- Re-embed variant sequences
- Compare ΔAUC between reference and variant embeddings (exploratory)

### Attribution
Use **LOGO (Leave-One-Gene-Out)** ΔAUC to assess which genes contribute most to downstream prediction tasks (e.g., MDD risk, cognitive scores). See [Yoon et al. BioKDD 2025](../../generated/kb_curated/papers-pdf/yoon_biokdd2025.pdf) for protocol details.

## Integration targets

Genetics embeddings are integrated with:

- **sMRI** IDPs (structural phenotypes) via CCA, late fusion, or contrastive alignment
- **fMRI** embeddings (e.g., BrainLM, Brain-JEPA) for gene–brain–behaviour triangulation
- **Behavioral phenotypes** (cognitive scores, psychiatric diagnoses) via multimodal prediction

See [Integration Strategy](../../integration/integration_strategy.md) for fusion protocols and [Modality Features: Genomics](../../integration/modality_features/genomics.md) for preprocessing specs.

## Source repositories

All genetics FM source code lives in `external_repos/`:

- `external_repos/caduceus/` — [kuleshov-group/caduceus](https://github.com/kuleshov-group/caduceus)
- `external_repos/dnabert2/` — [Zhihan1996/DNABERT2](https://github.com/Zhihan1996/DNABERT2)
- `external_repos/evo2/` — [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2)
- `external_repos/generator/` — [GenerTeam/GENERator](https://github.com/GenerTeam/GENERator)

Each model page includes walkthrough links to `docs/code_walkthroughs/` and structured YAML cards in `kb/model_cards/`.

## Next steps

- Validate gene embedding reproducibility across cohorts (UK Biobank WES, Cha Hospital panel sequencing)
- Benchmark LOGO ΔAUC stability under different embedding projection dimensions (256, 512, 1024)
- Explore regulatory region embeddings (enhancers, promoters) with long-context models like Evo 2

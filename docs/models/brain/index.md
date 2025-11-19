# Brain Foundation Models

This section documents the **neuroimaging foundation models** used for brain representation learning in the Neuro-Omics KB. These models extract embeddings from structural MRI (sMRI), functional MRI (fMRI), and other brain imaging modalities for downstream integration with genomic data, behavioral phenotypes, and clinical outcomes.

## Overview

All brain FMs documented here:

- **Operate on neuroimaging data** (volumetric MRI, parcel time series, or raw BOLD signals)
- **Support subject-level embeddings** via aggregation across spatial regions or temporal windows
- **Are pretrained on large multi-site datasets** (UK Biobank, HCP, ABCD, etc.)
- **Enable cross-modal alignment** with genomic and behavioral representations

## Model registry

| Model | Modality | Architecture | Key feature | Integration role |
|-------|----------|-------------|-------------|------------------|
| [BrainLM](brainlm.md) | fMRI | ViT-MAE | Masked autoencoding of parcel time series | Primary fMRI encoder; site-robust embeddings |
| [Brain-JEPA](brainjepa.md) | fMRI | JEPA | Joint-embedding prediction; no reconstruction loss | Alternative fMRI encoder; lower-latency option |
| [Brain Harmony](brainharmony.md) | sMRI + fMRI | ViT + TAPE | Multi-modal fusion via target-aware projection ensemble | Cross-modal sMRI+fMRI fusion; TAPE for multi-task |
| [BrainMT](brainmt.md) | sMRI (+ fMRI planned) | Hybrid Mamba-Transformer | Efficient long-range dependencies for 3D volumes | sMRI encoder; Mamba for computational efficiency |
| [SwiFT](swift.md) | fMRI | Swin Transformer | Hierarchical windows for spatiotemporal fMRI | Exploratory fMRI encoder; sequence-free modeling |

## Usage workflow

### For fMRI models (BrainLM, Brain-JEPA, SwiFT)

1. **Preprocess** rs-fMRI: parcellation (Schaefer/AAL), bandpass filter, motion scrubbing
2. **Tokenize** parcel time series (or 4D volumes for SwiFT)
3. **Embed** via pretrained encoder
4. **Pool** to subject-level representation (mean over tokens/time)
5. **Project** to 512-D for cross-modal alignment

### For sMRI models (BrainMT, Brain Harmony)

1. **Run** FreeSurfer or FSL FAST for tissue segmentation
2. **Extract** IDPs (cortical thickness, subcortical volumes) or feed raw T1w volumes
3. **Embed** via pretrained encoder
4. **Pool** to subject-level representation
5. **Project** to 512-D for fusion

## Key considerations

### Site/scanner harmonization
Multi-site pretraining (e.g., BrainLM on UKB+HCP) improves site robustness, but **residualize scanner/site effects** before fusion:

- Regress site dummy variables from embeddings
- Use ComBat or similar harmonization if needed (see [Integration Strategy](../../integration/integration_strategy.md))

### Motion artifacts
fMRI embeddings are sensitive to head motion. **Quality control:**

- Exclude high-motion frames (FD > 0.5 mm)
- Regress mean FD as confound in downstream prediction
- Report motion distributions stratified by diagnosis (e.g., ADHD vs TD)

### Multimodal fusion
**Brain Harmony** natively fuses sMRI and fMRI via TAPE (Target-Aware Projection Ensemble). For other models, use **late fusion** (concatenate embeddings) or **two-tower contrastive** alignment (see [Design Patterns](../../integration/design_patterns.md)).

## Integration targets

Brain embeddings are integrated with:

- **Genetics** embeddings (Caduceus, DNABERT-2) for gene–brain association discovery
- **Behavioral phenotypes** (cognitive scores, psychiatric diagnoses) via multimodal prediction
- **Clinical data** (longitudinal assessments, EHR records) for developmental trajectories

See [Integration Strategy](../../integration/integration_strategy.md) for fusion protocols and modality-specific feature specs:

- [Modality Features: sMRI](../../integration/modality_features/smri.md)
- [Modality Features: fMRI](../../integration/modality_features/fmri.md)

## Source repositories

All brain FM source code lives in `external_repos/`:

- `external_repos/brainlm/` — [vandijklab/BrainLM](https://github.com/vandijklab/BrainLM)
- `external_repos/brainjepa/` — [janklees/brainjepa](https://github.com/janklees/brainjepa)
- `external_repos/brainharmony/` — [hzlab/Brain-Harmony](https://github.com/hzlab/Brain-Harmony)
- `external_repos/brainmt/` — [arunkumar-kannan/brainmt-fmri](https://github.com/arunkumar-kannan/brainmt-fmri)
- `external_repos/swift/` — [Transconnectome/SwiFT](https://github.com/Transconnectome/SwiFT)

Each model page includes walkthrough links to `docs/code_walkthroughs/` and structured YAML cards in `kb/model_cards/`.

## Next steps

- Validate brain embedding reproducibility across cohorts (UK Biobank, Cha Hospital developmental cohort)
- Benchmark fMRI encoder stability across different parcellation schemes (Schaefer 100/200/400, AAL)
- Explore **EEG/EPhys** foundation models for pediatric/clinical settings (e.g., LaBraM, TBD)
- Integrate **diffusion MRI** embeddings for white matter microstructure (exploratory)

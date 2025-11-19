# Paper Cards

Structured YAML metadata for all key papers referenced in the KB. Each card includes:
- Citation metadata (title, authors, year, venue, DOI)
- Summary and key takeaways
- Methods/architecture details (for FM papers)
- Implications for our project
- Links to related code walkthroughs, model cards, and integration docs

## Integration Principles

| Paper | Card | Key Contribution |
|-------|------|------------------|
| Ensemble Integration (Li et al. 2022) | [ensemble_integration_li2022.yaml](ensemble_integration_li2022.yaml) | Late fusion rationale; stacking with diverse base learners |
| Oncology Multimodal Review (Waqas et al. 2024) | [oncology_multimodal_waqas2024.yaml](oncology_multimodal_waqas2024.yaml) | Confound control; evaluation discipline; fusion taxonomy |

## Foundation Models — Genetics

| Model | Card | Architecture | Context |
|-------|------|--------------|---------|
| Caduceus | [caduceus_2024.yaml](caduceus_2024.yaml) | BiMamba/Hyena RC-equivariant | Long-range |
| Evo2 | [evo2_2024.yaml](evo2_2024.yaml) | StripedHyena 2 | 1M tokens |
| GENERator | [generator_2024.yaml](generator_2024.yaml) | Transformer decoder | 6-mer tokenization |

## Foundation Models — Brain

| Model | Card | Architecture | Modalities |
|-------|------|--------------|------------|
| BrainLM | [brainlm_2024.yaml](brainlm_2024.yaml) | ViT-MAE (Nystromformer) | fMRI |
| Brain-JEPA | [brainjepa_2024.yaml](brainjepa_2024.yaml) | JEPA + gradient positioning | fMRI |
| Brain Harmony | [brainharmony_2025.yaml](brainharmony_2025.yaml) | TAPE + hub tokens | sMRI + fMRI |
| BrainMT | [brainmt_2025.yaml](brainmt_2025.yaml) | Hybrid Mamba-Transformer | fMRI |

## Foundation Models — Multimodal / Unified FMs

| Model | Card | Architecture | Modalities |
|-------|------|--------------|------------|
| Mixture-of-Transformers (MoT) | [mot_2025.yaml](mot_2025.yaml) | Sparse modality-aware transformer | Text · Image · Speech |
| BAGEL | [bagel_2025.yaml](bagel_2025.yaml) | Unified decoder-only MoT-based FM | Text · Image · Video · Web |

## Methods & Prior Work

| Paper | Card | Contribution |
|-------|------|--------------|
| Yoon et al. BIOKDD'25 | [yoon_biokdd2025.yaml](yoon_biokdd2025.yaml) | MDD classification with gene embeddings; LOGO attribution protocol |
| PRS Guide (Choi & O'Reilly 2019) | [prs_guide.yaml](prs_guide.yaml) | Polygenic risk score computation and validation |
| GWAS Diverse Populations (Peterson et al. 2019) | [gwas_diverse_populations.yaml](gwas_diverse_populations.yaml) | Ancestry control and stratification mitigation |

## Usage

```bash
# View a paper card
cat kb/paper_cards/caduceus_2024.yaml

# Search for papers by tag
grep -l "integration" kb/paper_cards/*.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('kb/paper_cards/ensemble_integration_li2022.yaml'))"
```

## Verification Status

All cards are marked `verification_status: "needs_human_review"`. After reading each paper and verifying the summary/takeaways:
1. Update `verification_status: "verified"`
2. Add any missing details
3. Commit the change

## Related Docs

- [Code Walkthroughs](../../docs/code_walkthroughs/) — Implementation guides for each FM
- [Model Cards](../model_cards/) — Structured metadata for models
- [Integration Strategy](../../docs/integration/integration_strategy.md) — How these papers inform our approach
- [Experiment Configs](../../configs/experiments/) — Ready-to-run templates based on these methods

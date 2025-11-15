Neurogenomics KB

## Purpose

A **documentation-focused knowledge base** for genetics and brain foundation models and their multimodal integration. This repository contains structured documentation, YAML metadata cards, code walkthroughs, and integration strategies.

**Note:** This is a **knowledge base only** - no implementation code. For actual model training/inference, refer to the `external_repos/` directories or the original model repositories.

## What's Inside

### 📚 Documentation (`docs/`)
- **Code Walkthroughs**: In-depth guides for 7 foundation models
  - Genomics: Caduceus, DNABERT-2, GENERator, Evo 2
  - Brain: BrainLM, BrainMT, SwiFT
- **Integration Playbooks**: Strategies for multimodal fusion
- **Data Schemas**: UK Biobank, HCP, and other datasets
- **Decision Logs**: Architectural and research choices

### 🏷️ Metadata (`kb/`)
- **Model Cards** (`model_cards/*.yaml`): Structured metadata for each FM
- **Dataset Cards** (`datasets/*.yaml`): Data specifications and schemas
- **Integration Cards** (`integration_cards/*.yaml`): Multimodal fusion strategies
- **Paper Cards** (`paper_cards/*.yaml`): Research paper references

### 🔗 External Repos (`external_repos/`)
- Cloned source code for reference (read-only)
- Links to original repositories for actual implementation

### 🔧 KB Management (`scripts/`)
- `manage_kb.py`: Validation and catalog generation tools
- No implementation scripts (moved to separate repos)

## Build & Serve Locally

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Serve documentation site
mkdocs serve

# Validate YAML cards
python scripts/manage_kb.py validate models
python scripts/manage_kb.py validate datasets
```

## Codex Quality Gate (2-Cycle Workflow)

Use `scripts/codex_gate.py` to enforce an automatic pass/fail gate whenever you run Codex in two consecutive cycles. The gate bundles domain tests (`manage_kb` validators + MkDocs), Python linting (`ruff`), and static type checks (`mypy`), then records the outcome so Cycle 2 can compare itself to Cycle 1.

```
# Cycle 1 – quick sanity before giving Codex control
python scripts/codex_gate.py --mode fast --label cycle1 --since origin/main

# Cycle 2 – full sweep before handing work back
python scripts/codex_gate.py --mode full --label cycle2 --since HEAD~1
```

- `--mode fast` skips the MkDocs build for a faster signal; `--mode full` runs everything.
- `--since` scopes checks to paths that changed versus the provided git ref (fallback: run all).
- Results are stored under `~/.cache/codex_gate/neurogenomics-kb` (override via `--state-dir`).
- `--fail-fast` stops on the first failure, and `--list-checks` shows the exact commands.

If Cycle 2 introduces a regression, the gate exits with a non-zero status so the automation can halt before launching the next Codex pass.

## Usage

### Explore Model Cards
```bash
# List all models
ls kb/model_cards/*.yaml

# View a specific model
cat kb/model_cards/caduceus.yaml
```

### Read Code Walkthroughs
```bash
# Open in browser after serving
mkdocs serve
# Visit: http://localhost:8000/code_walkthroughs/caduceus_walkthrough/
```

### Reference External Code
```bash
# External repos are for reference only
cd external_repos/caduceus
# Follow original repo instructions for training/inference
```

## Contribution Guidelines

This is a **documentation repository**. Contributions should focus on:

✅ **Do:**
- Add/update model cards with accurate metadata
- Write comprehensive code walkthroughs
- Document integration strategies
- Add decision logs for architectural choices
- Improve data schemas and benchmarks

❌ **Don't:**
- Add implementation code (training scripts, inference pipelines)
- Include custom model variants
- Create helper scripts for embeddings extraction
- Add experimental code

### YAML Card Guidelines
- Keep `verified: false` until human review
- Include all required fields (see `kb/*/template.yaml`)
- Reference external repos for implementation details
- Use links for code examples, not inline code

### Documentation Style
- Concise with citations
- Link to original papers and repos
- Include practical integration examples
- Focus on "how to use" not "how to implement"

## Related Repositories

- **PDF/Markdown Converter**: [pdf-md-ai-summaries](https://github.com/allison-eunse/pdf-md-ai-summaries)
- **Model Implementations**: See links in individual model cards
- **Datasets**: UK Biobank (restricted), HCP, OpenGenome2

## Contact

**Maintainer**: Allison Eun Se You  
**Purpose**: Knowledge base for neurogenomics foundation model research  
**Scope**: Documentation, metadata, integration strategies (no implementation)

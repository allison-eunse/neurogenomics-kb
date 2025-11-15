# Neurogenomics-KB Quick Reference

## 📍 Repository Locations

### 1. **Neurogenomics KB** (Documentation & Metadata)
**Location**: `/Users/allison/.cursor/worktrees/neurogenomics-kb/2wmpo/`  
**GitHub**: (your existing repo)  
**Purpose**: Knowledge base - documentation, YAML cards, walkthroughs

### 2. **PDF Converter** (Separate Utility)
**Location**: `/Users/allison/Projects/pdf<->md;ai-summaries/`  
**GitHub**: https://github.com/allison-eunse/pdf-md-ai-summaries  
**Purpose**: PDF↔MD conversion + AI summaries

---

## 🎯 Neurogenomics-KB Structure

### Documentation
```
docs/code_walkthroughs/
├── brainlm_walkthrough.md      (BrainLM MAE guide)
├── brainmt_walkthrough.md      (BrainMT Mamba+Transformer)
├── caduceus_walkthrough.md     (RC-equivariant Hyena)
├── dnabert2_walkthrough.md     (BPE tokenization)
├── evo2_walkthrough.md         (StripedHyena 1M context)
├── generator_walkthrough.md    (6-mer generative model)
├── swift_walkthrough.md        (Swin 4D fMRI)
└── index.md
```

### Metadata Cards
```
kb/
├── model_cards/       (7 models: all valid YAML)
├── datasets/          (10 datasets: UKB, HCP, OpenGenome2, etc.)
├── integration_cards/ (2 cards: embeddings + alignment)
└── paper_cards/       (templates ready)
```

### Management
```
scripts/manage_kb.py    (validation tool)
```

---

## 🚀 Common Commands

### Neurogenomics-KB

```bash
# Navigate to KB
cd /Users/allison/.cursor/worktrees/neurogenomics-kb/2wmpo

# Serve documentation
mkdocs serve

# Validate model cards
python scripts/manage_kb.py validate models

# Validate dataset cards
python scripts/manage_kb.py validate datasets

# Check YAML syntax
python -c "import yaml; from pathlib import Path; \
[print(f'✓ {f.stem}') for f in Path('kb/model_cards').glob('*.yaml') \
if f.stem != 'template' and yaml.safe_load(f.read_text())]"

# Codex gate (fast / full)
python scripts/codex_gate.py --mode fast --label cycle1 --since origin/main
python scripts/codex_gate.py --mode full --label cycle2 --since HEAD~1
```

### PDF Converter

```bash
# Navigate to PDF repo
cd ~/Projects/pdf\<-\>md\;ai-summaries

# Install dependencies
pip install -r requirements.txt

# Convert PDF to Markdown
python pdf_to_markdown.py paper.pdf output.md

# Generate AI summary
python summary_generator.py paper.md summary.md

# Convert to aesthetic PDF
python markdown_to_pdf.py summary.md output.pdf

# Full pipeline
python pdf_to_markdown.py paper.pdf paper.md && \
python summary_generator.py paper.md summary.md && \
python markdown_to_pdf.py summary.md summary.pdf
```

---

## 📊 Repository Stats

### Neurogenomics-KB
- **Code Walkthroughs**: 8 complete guides
- **Model Cards**: 7 validated YAML files
- **Dataset Cards**: 10 specifications
- **Integration Cards**: 2 multimodal strategies
- **External Repos**: 7 reference implementations

### PDF Converter
- **Scripts**: 3 Python tools
- **Features**: PDF↔MD conversion, AI summarization, aesthetic PDFs
- **Theme**: Baby blue, lavender, wine red

---

## ✅ What's Clean

- ✅ No implementation scripts in KB repo
- ✅ All YAML cards parse successfully
- ✅ Generator walkthrough fixed (was empty)
- ✅ README clarified (KB-only purpose)
- ✅ PDF tools moved to separate repo
- ✅ All walkthroughs complete

---

## 🎨 PDF Converter Features

**Color Scheme:**
- Baby Blue (#89CFF0) - Main headers
- Lavender (#B695C0) - Subheaders
- Wine Red (#722F37) - Accents
- Times New Roman - Body font

**Capabilities:**
- PDF → Markdown with structure detection
- Markdown → PDF with professional styling
- AI-powered summary extraction (key points, methods, results, limitations)

---

## 📝 Next Steps

### For Neurogenomics-KB:
1. Continue documenting models and integration strategies
2. Add paper cards to `kb/paper_cards/`
3. Expand integration playbooks in `docs/integration/`
4. Keep YAML cards updated with new checkpoints

### For PDF Converter:
1. Push to GitHub: `cd ~/Projects/pdf<->md;ai-summaries && git push`
2. Test with your research papers
3. Customize colors if desired (edit `markdown_to_pdf.py`)

---

## 🔗 Quick Links

- **Model Cards**: `kb/model_cards/`
- **Walkthroughs**: `docs/code_walkthroughs/`
- **Integration**: `docs/integration/`
- **KB Script**: `scripts/manage_kb.py`
- **PDF Tools**: `~/Projects/pdf<->md;ai-summaries/`

---

**Last Updated**: November 15, 2025  
**Organized by**: Allison Eun Se You  
**Status**: ✅ Clean, validated, and ready to use


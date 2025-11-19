# Contributing to Neuro-Omics KB

Thank you for your interest in contributing! This knowledge base thrives on community contributions of documentation, structured metadata, and integration strategies.

## 🎯 What We're Looking For

This is a **documentation repository**. We welcome:

✅ **Model cards** - Add new FM cards with embedding recipes and integration hooks  
✅ **Code walkthroughs** - Step-by-step guides for using foundation models  
✅ **Integration strategies** - New fusion methods, harmonization approaches  
✅ **Paper cards** - Structured summaries of relevant research  
✅ **Dataset cards** - Documentation for new cohorts or datasets  
✅ **Analysis recipes** - Reproducible workflows (CCA, prediction, attribution, etc.)  
✅ **Bug fixes** - Typos, broken links, outdated information  

❌ We generally don't accept:
- Implementation code (training/inference scripts)
- Custom model variants
- Experimental helper scripts
- Large binary files

---

## 🚀 Getting Started

### 1. Fork and Clone
```bash
git clone https://github.com/YOUR-USERNAME/neuro-omics-kb.git
cd neuro-omics-kb
git checkout -b feature/my-contribution
```

### 2. Set Up Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Make Your Changes

#### Adding a model card:
```bash
cp kb/model_cards/template.yaml kb/model_cards/my_model.yaml
# Edit the file following the template structure
python scripts/manage_kb.py validate models
```

#### Adding a walkthrough:
```bash
cp docs/generated/templates/model_card_template.md docs/code_walkthroughs/my_model_walkthrough.md
# Write your walkthrough
mkdocs serve  # Preview at http://localhost:8000
```

#### Adding a dataset card:
```bash
cp kb/datasets/template.yaml kb/datasets/my_dataset.yaml
python scripts/manage_kb.py validate datasets
```

#### Adding a paper card:
```bash
cp kb/paper_cards/template.yaml kb/paper_cards/my_paper.yaml
# Fill in structured metadata
python scripts/manage_kb.py validate papers
```

### 4. Test Your Changes
```bash
# Validate YAML syntax
python scripts/manage_kb.py validate models
python scripts/manage_kb.py validate datasets

# Build documentation
mkdocs build --strict

# Serve locally to preview
mkdocs serve
```

### 5. Submit a Pull Request

1. Push your branch: `git push origin feature/my-contribution`
2. Open a PR on GitHub
3. Describe your changes clearly:
   - What does this add/fix?
   - Which files are affected?
   - Any related issues?

---

## 📋 Style Guidelines

### Documentation (Markdown)
- Use clear, concise language
- Include links to original papers/repos
- Add code examples where helpful
- Follow existing structure and tone
- Use proper heading hierarchy (# → ## → ###)

### YAML Cards
- Follow the template structure exactly
- Keep `verified: false` until human review
- Include all required fields
- Link to external resources, don't inline large blocks
- Use consistent indentation (2 spaces)

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- Be descriptive but concise
- Reference issues if applicable: "Fix #123: Update BrainLM card"

### Code Examples
- Use markdown code fences with language tags:
  ```python
  # Example code here
  ```
- Ensure examples are runnable or clearly marked as pseudocode

---

## 🔍 What to Document

### For Model Cards

**Required fields:**
- `id`: Unique identifier (lowercase, underscores)
- `name`: Full model name
- `modality`: genetics, brain, multimodal, etc.
- `summary`: 2-3 sentence overview
- `arch`: Architecture details (type, backbone, parameters)
- `embedding_recipe`: How to extract embeddings
- `repo`: GitHub link
- `license`: Code and weights licenses
- `tasks`: Downstream applications
- `verified`: false (until reviewed)
- `last_updated`: YYYY-MM-DD

**Optional but encouraged:**
- `weights`: Hugging Face or artifact links
- `checkpoints`: Pre-trained model paths
- `how_to_infer`: Code snippets for inference
- `integrations`: Links to integration strategies

### For Code Walkthroughs

**Structure:**
1. **Overview** - What the model does, key innovations
2. **At-a-Glance Table** - Architecture, params, context, capabilities
3. **Environment & Hardware** - Setup requirements, GPU notes
4. **Key Components** - Code snippets with explanations
5. **Integration Hooks** - How to extract embeddings for KB use
6. **KB References** - Links to model cards, integration recipes

**Best practices:**
- Use code references with line numbers where applicable
- Include actual file paths from `external_repos/`
- Add practical integration examples
- Link to relevant KB sections

### For Paper Cards

**Required fields:**
- `title`: Full paper title
- `authors`: List of authors
- `year`: Publication year
- `venue`: Journal/conference
- `pdf_source`: URL to paper
- `local_pdf_path`: Path in repo (if stored)
- `summary`: Key contributions (3-5 sentences)
- `key_contributions`: Bullet points
- `implications_for_project`: How it informs KB work
- `related_to`: Links to walkthroughs, cards
- `tags`: Categorization tags

### For Dataset Cards

**Required fields:**
- `id`: Unique dataset ID
- `name`: Full dataset name
- `cohort`: Population description
- `modalities`: List of data types
- `subjects`: Sample size
- `access`: How to obtain data
- `license`: Usage restrictions
- `tasks`: Supported analyses

---

## 🧪 Testing Checklist

Before submitting:

- [ ] YAML files validate without errors
- [ ] MkDocs builds successfully (`mkdocs build --strict`)
- [ ] All internal links work (no 404s)
- [ ] Code examples are syntactically correct
- [ ] Followed style guidelines
- [ ] Added entry to relevant index files (if needed)
- [ ] Updated `last_updated` dates in modified files

---

## 🤝 Code of Conduct

- Be respectful and constructive
- Focus on documentation quality
- Give credit where due (cite papers, acknowledge repos)
- Help others learn
- Keep discussions professional and on-topic

---

## 📧 Questions?

- Open an issue for questions or discussions
- Tag issues appropriately: `documentation`, `model-card`, `bug`, etc.
- Check existing issues before creating duplicates

---

## 🎓 Learning Resources

### Understanding Foundation Models
- [Genetics Models Overview](https://allison-eunse.github.io/neuro-omics-kb/models/genetics/)
- [Brain Models Overview](https://allison-eunse.github.io/neuro-omics-kb/models/brain/)
- [Integration Strategy](https://allison-eunse.github.io/neuro-omics-kb/integration/integration_strategy/)

### Documentation Best Practices
- [MkDocs Material docs](https://squidfunk.github.io/mkdocs-material/)
- [YAML syntax guide](https://yaml.org/spec/1.2.2/)
- [Markdown guide](https://www.markdownguide.org/)

---

## 🏆 Recognition

Contributors will be acknowledged in:
- Git commit history
- Pull request descriptions
- Future releases (if significant contributions)

---

Thank you for helping make this knowledge base better! 🎉

**Questions?** Open an issue or reach out to the maintainers.


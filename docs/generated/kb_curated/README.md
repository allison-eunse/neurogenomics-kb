---
title: AI Summaries and KB
status: draft
updated: 2025-11-16
---

# AI Summaries → KB Cards

- Use pdf_to_markdown.py to extract paper Markdown.
- Run summary_generator.py to produce brief summaries.
- Curate into KB cards in kb/ (datasets/, integration_cards/, model_cards/) using the templates.

Workflow

1) Convert: pdf_to_markdown.py input/*.pdf → build/*.md
2) Summarize: summary_generator.py build/paper.md → build/paper_summary.md
3) Curate: Copy key takeaways into kb/integration_cards/ or kb/model_cards/ using templates.
4) Publish: Commit and sync to neurogenomics-kb/docs if desired.

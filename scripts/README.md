# scripts/

This folder contains automation used by the Neuro-Omics KB:

- `manage_kb.py` – Typer-based CLI for metadata catalogs, validation, CI checks,
  walkthrough drafting, and RAG index generation. Example commands:
  - `python scripts/manage_kb.py catalog models --out docs/models/catalog.md`
  - `python scripts/manage_kb.py validate models`
  - `python scripts/manage_kb.py ci docs`
  - `python scripts/manage_kb.py rag-index`
  - `python scripts/manage_kb.py ops embeddings caduceus`

All commands assume you installed `requirements.txt` and run from the repo root.

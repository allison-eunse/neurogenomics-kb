#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[worktree-setup] %s\n' "$*"
}

ROOT_DIR="${ROOT_WORKTREE_PATH:-}"
WORKTREE_DIR="$(pwd)"

log "Starting (Unix) in ${WORKTREE_DIR}"
if [[ -n "${ROOT_DIR}" ]]; then
  log "ROOT_WORKTREE_PATH=${ROOT_DIR}"
else
  log "ROOT_WORKTREE_PATH=<unset>"
fi
log "Node: $(command -v node || echo 'missing')"
log "Python: $(command -v python3 || echo 'missing')"

mkdir -p .cursor

# Python environment setup (prefer uv, fall back to python3 venv)
if command -v uv >/dev/null 2>&1; then
  log "Using uv for Python deps (fast)"
  uv venv .venv >/dev/null 2>&1 || true
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    if [[ -f requirements.txt ]]; then
      log "Installing Python deps via uv"
      uv pip install -r requirements.txt >/dev/null 2>&1 || log "uv install failed (non-blocking)"
    fi
  fi
else
  PY_BIN=${PY_BIN:-python3}
  if command -v "$PY_BIN" >/dev/null 2>&1; then
    if [[ ! -d .venv ]]; then
      "$PY_BIN" -m venv .venv || true
    fi
    if [[ -f .venv/bin/activate ]]; then
      # shellcheck disable=SC1091
      source .venv/bin/activate
      python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
      if [[ -f requirements.txt ]]; then
        pip install -r requirements.txt >/dev/null 2>&1 || log "pip install failed (non-blocking)"
      fi
    fi
  else
    log "WARNING: python3 not found; skipping Python setup"
  fi
fi

# Node dependencies (best-effort)
if [[ -f package.json ]]; then
  if command -v pnpm >/dev/null 2>&1; then
    log "Installing Node deps with pnpm"
    pnpm install --frozen-lockfile >/dev/null 2>&1 || pnpm install >/dev/null 2>&1 || log "pnpm install failed (non-blocking)"
    pnpm run build >/dev/null 2>&1 || true
  elif command -v bun >/dev/null 2>&1; then
    log "Installing Node deps with bun"
    bun install >/dev/null 2>&1 || log "bun install failed (non-blocking)"
    bun run build >/dev/null 2>&1 || true
  elif command -v npm >/dev/null 2>&1; then
    log "Installing Node deps with npm"
    npm ci >/dev/null 2>&1 || npm install >/dev/null 2>&1 || log "npm install failed (non-blocking)"
    npm run build >/dev/null 2>&1 || true
  else
    log "WARNING: No Node package manager found"
  fi
fi

# Environment files
if [[ -f .env.example && ! -f .env ]]; then
  cp .env.example .env && log "Created .env from .env.example"
fi
if [[ -n "${ROOT_DIR}" && -f "${ROOT_DIR}/.env.example" && ! -f .env ]]; then
  cp "${ROOT_DIR}/.env.example" .env && log "Seeded .env from ROOT_WORKTREE_PATH/.env.example"
fi

# Git tools and hooks
if [[ -f .gitmodules ]]; then
  log "Initializing git submodules"
  git submodule update --init --recursive >/dev/null 2>&1 || log "Submodule update failed (non-blocking)"
fi
if command -v pre-commit >/dev/null 2>&1; then
  log "Installing pre-commit hooks"
  pre-commit install --install-hooks >/dev/null 2>&1 || log "Pre-commit install failed (non-blocking)"
fi

# Ensure expected directories exist for agents
mkdir -p rag/vectordb kb/papers_fulltext docs/.worktree-cache >/dev/null 2>&1 || true

# Build docs (best-effort)
if command -v mkdocs >/dev/null 2>&1; then
  log "Building mkdocs site preview"
  mkdocs build --strict --site-dir .cursor/site-preview >/dev/null 2>&1 || log "mkdocs build failed (non-blocking)"
fi

# Knowledge base validation + indices (best-effort)
PYTHON_BIN="$(command -v python 2>/dev/null || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
fi

if [[ -n "${PYTHON_BIN}" && -f scripts/manage_kb.py ]]; then
  log "Validating KB metadata"
  "${PYTHON_BIN}" scripts/manage_kb.py validate models >/dev/null 2>&1 || log "model validation failed (non-blocking)"
  "${PYTHON_BIN}" scripts/manage_kb.py validate datasets >/dev/null 2>&1 || log "dataset validation failed (non-blocking)"
  "${PYTHON_BIN}" scripts/manage_kb.py validate links >/dev/null 2>&1 || log "link validation failed (non-blocking)"
  log "Rebuilding RAG index"
  "${PYTHON_BIN}" scripts/manage_kb.py rag-index >/dev/null 2>&1 || log "RAG index build failed (non-blocking)"
fi

# Generate manifest for parallel agents
if [[ -n "${PYTHON_BIN}" ]]; then
  "${PYTHON_BIN}" <<'PY' || log "Manifest generation skipped"
from __future__ import annotations
import json
from pathlib import Path

root = Path(".").resolve()

def list_files(base: Path, pattern: str) -> list[str]:
    if not base.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in base.rglob(pattern))

manifest = {
    "docs_markdown": list_files(root / "docs", "*.md"),
    "kb_cards": {
        "model_cards": list_files(root / "kb" / "model_cards", "*.yaml"),
        "datasets": list_files(root / "kb" / "datasets", "*.yaml"),
        "integration_cards": list_files(root / "kb" / "integration_cards", "*.yaml"),
        "paper_cards": list_files(root / "kb" / "paper_cards", "*.yaml"),
    },
    "scripts": list_files(root / "scripts", "*.py"),
    "external_repos": {},
    "parallel_tasks": [
        {
            "id": 1,
            "label": "Docs & MkDocs site",
            "paths": ["docs/", "mkdocs.yml"],
        },
        {
            "id": 2,
            "label": "Knowledge base YAML cards",
            "paths": ["kb/model_cards/", "kb/datasets/", "kb/integration_cards/"],
        },
        {
            "id": 3,
            "label": "External research repos",
            "paths": ["external_repos/"],
        },
        {
            "id": 4,
            "label": "RAG + orchestration scripts",
            "paths": ["scripts/", "rag/", "kb/rag/"],
        },
    ],
}

ext_root = root / "external_repos"
if ext_root.exists():
    for repo in sorted(p for p in ext_root.iterdir() if p.is_dir()):
        summary = {
            "readme": str(repo.relative_to(root) / "README.md") if (repo / "README.md").exists() else None,
            "requirements": str(repo.relative_to(root) / "requirements.txt") if (repo / "requirements.txt").exists() else None,
            "has_tests": (repo / "tests").exists(),
        }
        manifest["external_repos"][repo.name] = summary

cursor_dir = root / ".cursor"
cursor_dir.mkdir(parents=True, exist_ok=True)
(cursor_dir / "agent-manifest.json").write_text(json.dumps(manifest, indent=2))
print("[worktree-setup] Wrote .cursor/agent-manifest.json")
PY
else
  log "Skipping manifest generation (python unavailable)"
fi

log "Done (Unix)."


Param()

function Log($message) {
  Write-Host "[worktree-setup] $message"
}

$worktreePath = Get-Location
$rootPath = $env:ROOT_WORKTREE_PATH

Log "Starting (Windows) in $worktreePath"
if ($rootPath) {
  Log "ROOT_WORKTREE_PATH=$rootPath"
} else {
  Log "ROOT_WORKTREE_PATH=<unset>"
}

$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if ($nodeCmd) { Log "Node: $($nodeCmd.Source)" } else { Log "Node: missing" }

$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd) { $pyCmd = Get-Command python3 -ErrorAction SilentlyContinue }
if ($pyCmd) { Log "Python: $($pyCmd.Source)" } else { Log "Python: missing" }

New-Item -ItemType Directory -Force -Path ".\.cursor" | Out-Null

# Python environment setup
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCmd) {
  Log "Using uv for Python deps (fast)"
  & $uvCmd.Source venv .venv | Out-Null
  if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
    if (Test-Path ".\requirements.txt") {
      Log "Installing Python deps via uv"
      & $uvCmd.Source pip install -r requirements.txt | Out-Null
    }
  }
} elseif ($pyCmd) {
  if (-not (Test-Path ".\.venv")) {
    & $pyCmd.Source -m venv .venv | Out-Null
  }
  if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip setuptools wheel | Out-Null
    if (Test-Path ".\requirements.txt") {
      pip install -r requirements.txt | Out-Null
    }
  }
} else {
  Log "WARNING: python not found; skipping Python setup"
}

# Node dependencies (best-effort)
if (Test-Path ".\package.json") {
  $pnpm = Get-Command pnpm -ErrorAction SilentlyContinue
  $bun = Get-Command bun -ErrorAction SilentlyContinue
  $npm = Get-Command npm -ErrorAction SilentlyContinue
  if ($pnpm) {
    Log "Installing Node deps with pnpm"
    & $pnpm.Source install --frozen-lockfile | Out-Null
    if ($LASTEXITCODE -ne 0) { & $pnpm.Source install | Out-Null }
    & $pnpm.Source run build | Out-Null
  } elseif ($bun) {
    Log "Installing Node deps with bun"
    & $bun.Source install | Out-Null
    & $bun.Source run build | Out-Null
  } elseif ($npm) {
    Log "Installing Node deps with npm"
    & $npm.Source ci | Out-Null
    if ($LASTEXITCODE -ne 0) { & $npm.Source install | Out-Null }
    & $npm.Source run build | Out-Null
  } else {
    Log "WARNING: No Node package manager found"
  }
}

# Environment files
if ((Test-Path ".\.env") -eq $false -and (Test-Path ".\.env.example")) {
  Copy-Item ".\.env.example" ".\.env"
  Log "Created .env from .env.example"
}
if ($rootPath -and (Test-Path "$rootPath\.env.example") -and (Test-Path ".\.env") -eq $false) {
  Copy-Item "$rootPath\.env.example" ".\.env"
  Log "Seeded .env from ROOT_WORKTREE_PATH\.env.example"
}

# Git utilities
if (Test-Path ".\.gitmodules") {
  Log "Initializing git submodules"
  git submodule update --init --recursive | Out-Null
}
if (Get-Command pre-commit -ErrorAction SilentlyContinue) {
  Log "Installing pre-commit hooks"
  pre-commit install --install-hooks | Out-Null
}

# Ensure directory scaffolding exists
New-Item -ItemType Directory -Force -Path ".\rag\vectordb" | Out-Null
New-Item -ItemType Directory -Force -Path ".\kb\papers_fulltext" | Out-Null
New-Item -ItemType Directory -Force -Path ".\docs\.worktree-cache" | Out-Null

# Build docs (best-effort)
if (Get-Command mkdocs -ErrorAction SilentlyContinue) {
  Log "Building mkdocs site preview"
  mkdocs build --strict --site-dir .cursor/site-preview | Out-Null
}

$pyRuntime = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyRuntime) { $pyRuntime = Get-Command python3 -ErrorAction SilentlyContinue }

# Knowledge base validation + RAG index
if ($pyRuntime -and (Test-Path ".\scripts\manage_kb.py")) {
  Log "Validating KB metadata"
  & $pyRuntime.Source scripts/manage_kb.py validate models | Out-Null
  if ($LASTEXITCODE -ne 0) { Log "model validation failed (non-blocking)" }
  & $pyRuntime.Source scripts/manage_kb.py validate datasets | Out-Null
  if ($LASTEXITCODE -ne 0) { Log "dataset validation failed (non-blocking)" }
  & $pyRuntime.Source scripts/manage_kb.py validate links | Out-Null
  if ($LASTEXITCODE -ne 0) { Log "link validation failed (non-blocking)" }
  Log "Rebuilding RAG index"
  & $pyRuntime.Source scripts/manage_kb.py rag-index | Out-Null
  if ($LASTEXITCODE -ne 0) { Log "RAG index build failed (non-blocking)" }
}

# Manifest for parallel agents
if ($pyRuntime) {
  $manifestScript = @'
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
        {"id": 1, "label": "Docs & MkDocs site", "paths": ["docs/", "mkdocs.yml"]},
        {"id": 2, "label": "Knowledge base YAML cards", "paths": ["kb/model_cards/", "kb/datasets/", "kb/integration_cards/"]},
        {"id": 3, "label": "External research repos", "paths": ["external_repos/"]},
        {"id": 4, "label": "RAG + orchestration scripts", "paths": ["scripts/", "rag/", "kb/rag/"]},
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
'@
  & $pyRuntime.Source "-c" $manifestScript
} else {
  Log "Skipping manifest generation (python unavailable)"
}

Log "Done (Windows)."


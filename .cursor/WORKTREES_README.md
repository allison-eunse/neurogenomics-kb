# Worktrees Configuration for neurogenomics-kb

This configuration enables **parallel agents mode** in Cursor, allowing multiple AI agents to work simultaneously in isolated worktrees without conflicts.

## 🚀 Quick Reference

### System Packages Auto-Installed

| Platform | Package Manager | Auto-Installs |
|----------|----------------|---------------|
| **macOS** | Homebrew | ✅ python@3.11, git, git-lfs, uv |
| **Linux** | apt/yum/dnf | ⚠️ Shows instructions only (needs sudo) |
| **Windows** | Chocolatey/winget | ✅ python, git |

### Python Packages (via pip/uv)

- mkdocs, mkdocs-material
- pyyaml, typer, rich, rapidfuzz
- pre-commit (if not system-installed)

### What Gets Validated

- ✅ Model cards (`kb/model_cards/*.yaml`)
- ✅ Dataset cards (`kb/datasets/*.yaml`)
- ✅ Integration references

## What This Configuration Does

### 🔧 Automatic Setup on Worktree Creation

When Cursor creates a new worktree for parallel agents, it will automatically:

1. **System Package Management** ✨ **[NEW]**
   - **macOS**: Uses Homebrew to install missing dependencies
     - Automatically installs: `python@3.11`, `git`, `git-lfs`, `uv`
     - Runs `brew shellenv` to configure PATH
   - **Linux**: Detects package manager (apt-get, yum, dnf)
     - Shows instructions for missing packages
   - **Windows**: Uses Chocolatey or winget
     - Automatically installs: `python`, `git`
   - All system installations are **non-blocking** (won't fail setup if they fail)

2. **Python Environment**
   - Creates a `.venv` virtual environment
   - Installs dependencies from `requirements.txt` using:
     - `uv` (ultra-fast) if available or auto-installed via Homebrew
     - Standard `pip` as fallback
   - Packages: mkdocs, mkdocs-material, pyyaml, typer, rich, rapidfuzz
   - Installs `pre-commit` via pip if not available system-wide

3. **Node.js Environment** (if `package.json` exists)
   - Uses `pnpm`, `bun`, or `npm` (in order of preference)
   - Installs dependencies and runs build

4. **Environment Files**
   - Copies `.env.example` → `.env` if not present
   - Can seed from root worktree's `.env.example`

5. **Git Submodules**
   - Initializes and updates all submodules recursively
   - Non-blocking if it fails

6. **Pre-commit Hooks**
   - Installs `pre-commit` via pip if not already available
   - Installs pre-commit hooks if available

7. **MkDocs Site**
   - Builds the documentation site with `mkdocs build --strict`
   - Non-blocking (best-effort)

8. **Project-Specific Directories**
   - Creates `rag/vectordb` for RAG vector database
   - Creates `kb/papers_fulltext` for full-text papers

9. **KB Validation** ✨ **[Updated]**
   - Validates all model cards (`python scripts/manage_kb.py validate models`)
   - Validates all dataset cards (`python scripts/manage_kb.py validate datasets`)
   - Validates integration links (`python scripts/manage_kb.py validate links`)
   - Non-blocking if validation fails

## Security & Privacy 🔒

### What This Configuration Accesses:

✅ **System Package Managers** (New!)
- **macOS**: Homebrew (`/opt/homebrew` or `/usr/local/Homebrew`)
  - Used to install: python, git, git-lfs, uv
  - Only installs if packages are missing
- **Linux**: apt-get, yum, or dnf
  - Shows warnings for missing packages (doesn't auto-install, requires sudo)
- **Windows**: Chocolatey or winget
  - Used to install: python, git

✅ **Project-Scoped Operations**
- Uses relative paths (e.g., `./scripts/manage_kb.py`, `./.venv`)
- Virtual environment is isolated to each worktree
- Git operations limited to project repository

❌ **What This Does NOT Access:**
- No access to `~/Documents`, `~/Desktop`, or other personal folders
- No modification of system Python or global packages
- No access to files outside the project directory
- All installations are optional and non-blocking

### Safe Environment Variables:

- `ROOT_WORKTREE_PATH` - Points to the main worktree, used only to seed `.env` files
- `PATH` - Extended to include Homebrew binaries (macOS only)
- All other environment variables are scoped to the project

## Platform Support

- **macOS/Linux**: Uses `setup-worktree-unix` (bash commands)
- **Windows**: Uses `setup-worktree-windows` (PowerShell commands)
- **Fallback**: Generic `setup-worktree` for unsupported platforms

## How Parallel Agents Use This

When you use parallel agents in Cursor:

1. Cursor creates a git worktree at: `/Users/allison/.cursor/worktrees/neurogenomics-kb/<WORKTREE_ID>`
2. Each worktree gets its own isolated copy of the repo
3. This setup script runs automatically in each new worktree
4. Agents work independently without file conflicts
5. Changes can be merged back to main when complete

## Project Structure

```
neurogenomics-kb/
├── .cursor/
│   └── worktrees.json          # This configuration file
├── kb/
│   ├── model_cards/            # Model metadata (validated)
│   ├── datasets/               # Dataset metadata (validated)
│   ├── integration_cards/      # Integration metadata (validated)
│   └── papers_fulltext/        # Created automatically
├── docs/                       # MkDocs documentation
├── scripts/
│   └── manage_kb.py            # KB management & validation CLI
├── external_repos/             # Git submodules for external projects
├── rag/
│   └── vectordb/               # Created automatically
├── requirements.txt            # Python dependencies
└── mkdocs.yml                  # MkDocs configuration
```

## Testing the Configuration

To test this configuration manually:

```bash
cd /Users/allison/.cursor/worktrees/neurogenomics-kb/Z2G4h

# Unix/macOS
bash -c "$(cat .cursor/worktrees.json | jq -r '."setup-worktree-unix"[]')"

# Or manually run each step from the JSON
```

## Customization

To add custom setup steps, edit `.cursor/worktrees.json` and add commands to:
- `setup-worktree-unix` - for macOS/Linux
- `setup-worktree-windows` - for Windows

### Example: Add a custom validation

```json
{
  "setup-worktree-unix": [
    // ... existing commands ...
    "if [ -f my_custom_script.sh ]; then bash my_custom_script.sh; fi"
  ]
}
```

## Troubleshooting

### Homebrew not found (macOS)
- Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- After install, run: `eval "$(/opt/homebrew/bin/brew shellenv)"`
- Restart your terminal

### System packages not installing automatically
- **macOS**: Ensure Homebrew is installed and in PATH
- **Linux**: The setup shows instructions but doesn't auto-install (requires sudo)
  - Run suggested command: `sudo apt-get install python3 python3-pip python3-venv git`
- **Windows**: Ensure Chocolatey or winget is installed
  - Chocolatey: https://chocolatey.org/install
  - winget: Comes with Windows 11 by default

### Python dependencies not installing
- Check that Python 3 is in your PATH: `which python3`
- Try installing `uv` for faster setup: `brew install uv` (macOS) or `pip install uv`
- On macOS, if python3 is missing: `brew install python@3.11`

### Git submodules fail to initialize
- Ensure you have network access
- Check SSH keys if using SSH URLs: `ssh -T git@github.com`
- May need to authenticate with GitHub

### MkDocs build fails
- This is non-blocking, agents can still work
- Check `mkdocs.yml` for configuration errors
- Ensure all dependencies installed: `. .venv/bin/activate && pip install -r requirements.txt`

### KB Validation fails
- This is non-blocking, agents can still work
- Run manually: `python scripts/manage_kb.py validate models`
- Check model/dataset/integration card YAML syntax

### Permission denied errors
- On macOS/Linux: Package installations via Homebrew don't require sudo
- On Linux: System package installation requires sudo (shown as warning only)
- Virtual environment operations don't require special permissions

## Documentation References

- [Cursor Worktrees Documentation](https://cursor.com/docs/configuration/worktrees)
- [Git Worktrees](https://git-scm.com/docs/git-worktree)
- Project-specific KB management: `python scripts/manage_kb.py --help`


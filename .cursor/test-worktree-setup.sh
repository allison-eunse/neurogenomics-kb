#!/usr/bin/env bash
# Test script for worktree configuration
# Run this to verify your worktree setup works correctly

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Worktree Configuration Test for neurogenomics-kb   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}✓${NC} $1"
}

fail() {
    echo -e "${RED}✗${NC} $1"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Test 1: Check worktrees.json exists and is valid JSON
echo "Test 1: Checking worktrees.json..."
if [ -f ".cursor/worktrees.json" ]; then
    if jq empty .cursor/worktrees.json 2>/dev/null; then
        pass "worktrees.json exists and is valid JSON"
    else
        fail "worktrees.json is not valid JSON"
        exit 1
    fi
else
    fail "worktrees.json not found"
    exit 1
fi

# Test 2: Check OS detection
echo ""
echo "Test 2: Checking OS detection..."
OS=$(uname)
if [ "$OS" = "Darwin" ]; then
    pass "macOS detected"
    
    # Check Homebrew
    if command -v brew >/dev/null 2>&1; then
        pass "Homebrew is installed: $(brew --version | head -n1)"
    else
        warn "Homebrew not found - install from https://brew.sh"
    fi
elif [ "$OS" = "Linux" ]; then
    pass "Linux detected"
    
    # Check package manager
    if command -v apt-get >/dev/null 2>&1; then
        pass "apt-get package manager found"
    elif command -v yum >/dev/null 2>&1; then
        pass "yum package manager found"
    elif command -v dnf >/dev/null 2>&1; then
        pass "dnf package manager found"
    else
        warn "No supported package manager found"
    fi
else
    warn "Unrecognized OS: $OS"
fi

# Test 3: Check essential tools
echo ""
echo "Test 3: Checking essential tools..."

if command -v python3 >/dev/null 2>&1; then
    pass "Python 3: $(python3 --version)"
else
    fail "Python 3 not found"
fi

if command -v git >/dev/null 2>&1; then
    pass "Git: $(git --version)"
else
    fail "Git not found"
fi

if command -v git-lfs >/dev/null 2>&1; then
    pass "Git LFS: $(git-lfs --version | head -n1)"
else
    warn "Git LFS not found (optional)"
fi

if command -v uv >/dev/null 2>&1; then
    pass "uv: $(uv --version)"
else
    warn "uv not found (optional, but speeds up Python setup)"
fi

# Test 4: Check Python virtual environment
echo ""
echo "Test 4: Checking Python virtual environment..."
if [ -d ".venv" ]; then
    pass "Virtual environment exists"
    
    if [ -f ".venv/bin/activate" ]; then
        pass "Virtual environment activation script found"
        
        # Activate and check packages
        source .venv/bin/activate
        
        if command -v python >/dev/null 2>&1; then
            pass "Virtual environment Python: $(python --version)"
        fi
        
        # Check for required packages
        if python -c "import mkdocs" 2>/dev/null; then
            pass "mkdocs installed"
        else
            warn "mkdocs not installed"
        fi
        
        if python -c "import yaml" 2>/dev/null; then
            pass "pyyaml installed"
        else
            warn "pyyaml not installed"
        fi
        
        deactivate
    else
        warn "Virtual environment activation script not found"
    fi
else
    warn "Virtual environment not found (will be created on worktree setup)"
fi

# Test 5: Check project structure
echo ""
echo "Test 5: Checking project structure..."

REQUIRED_DIRS=(
    "kb"
    "kb/model_cards"
    "kb/datasets"
    "kb/integration_cards"
    "docs"
    "scripts"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        pass "Directory exists: $dir"
    else
        fail "Directory missing: $dir"
    fi
done

if [ -f "scripts/manage_kb.py" ]; then
    pass "KB management script exists"
else
    fail "scripts/manage_kb.py not found"
fi

if [ -f "requirements.txt" ]; then
    pass "requirements.txt exists"
else
    fail "requirements.txt not found"
fi

if [ -f "mkdocs.yml" ]; then
    pass "mkdocs.yml exists"
else
    fail "mkdocs.yml not found"
fi

# Test 6: Check git submodules
echo ""
echo "Test 6: Checking git submodules..."
if [ -f ".gitmodules" ]; then
    pass ".gitmodules file exists"
    
    SUBMODULES=$(git config --file .gitmodules --get-regexp path | wc -l)
    if [ "$SUBMODULES" -gt 0 ]; then
        pass "Found $SUBMODULES submodule(s)"
        
        # Check if submodules are initialized
        if git submodule status | grep -q "^-"; then
            warn "Some submodules are not initialized (will be done on worktree setup)"
        else
            pass "All submodules initialized"
        fi
    fi
else
    warn ".gitmodules not found (no submodules configured)"
fi

# Test 7: Check setup script commands
echo ""
echo "Test 7: Validating setup commands..."

UNIX_COMMANDS=$(jq -r '.["setup-worktree-unix"][]' .cursor/worktrees.json | wc -l)
if [ "$UNIX_COMMANDS" -gt 0 ]; then
    pass "Unix setup has $UNIX_COMMANDS commands"
else
    fail "Unix setup has no commands"
fi

WINDOWS_COMMANDS=$(jq -r '.["setup-worktree-windows"][]' .cursor/worktrees.json | wc -l)
if [ "$WINDOWS_COMMANDS" -gt 0 ]; then
    pass "Windows setup has $WINDOWS_COMMANDS commands"
else
    fail "Windows setup has no commands"
fi

# Check for Homebrew integration
if jq -r '.["setup-worktree-unix"][]' .cursor/worktrees.json | grep -q "brew"; then
    pass "Homebrew integration present in Unix setup"
else
    warn "Homebrew integration not found in Unix setup"
fi

# Check for KB validation
if jq -r '.["setup-worktree-unix"][]' .cursor/worktrees.json | grep -q "manage_kb.py"; then
    pass "KB validation present in Unix setup"
else
    warn "KB validation not found in Unix setup"
fi

# Test 8: Check documentation
echo ""
echo "Test 8: Checking documentation..."

if [ -f ".cursor/WORKTREES_README.md" ]; then
    pass "WORKTREES_README.md exists"
else
    warn "WORKTREES_README.md not found"
fi

if [ -f ".cursor/CHANGELOG.md" ]; then
    pass "CHANGELOG.md exists"
else
    warn "CHANGELOG.md not found"
fi

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                   Test Complete!                     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Your worktree configuration is ready for parallel agents mode!"
echo ""
echo "To use parallel agents:"
echo "  1. Open Cursor"
echo "  2. Enable parallel agents mode"
echo "  3. Cursor will automatically create worktrees and run setup"
echo ""
echo "For more information, see .cursor/WORKTREES_README.md"


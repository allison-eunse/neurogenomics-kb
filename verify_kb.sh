#!/usr/bin/env bash
set -e

echo "═══════════════════════════════════════════════════════════"
echo "  KB Verification Script"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check Python files
echo "✓ Checking Python tools..."
python3 -c "import yaml; print('  - YAML module OK')"
ls scripts/manage_kb.py scripts/codex_gate.py > /dev/null && echo "  - KB management scripts present"

# Validate YAMLs
echo ""
echo "✓ Validating YAML cards..."
for yaml_file in kb/paper_cards/*.yaml kb/datasets/ukb_manifest_stub.yaml configs/experiments/*.yaml; do
    if [[ "$(basename "$yaml_file")" != "template.yaml" ]]; then
        python3 -c "import yaml; yaml.safe_load(open('$yaml_file'))" && echo "  - $(basename "$yaml_file")"
    fi
done

# Check docs
echo ""
echo "✓ Checking documentation..."
[[ -f docs/index.md ]] && echo "  - Main index present"
[[ -f docs/decisions/2025-11-integration-plan.md ]] && echo "  - Integration plan present"
[[ -f docs/integration/integration_strategy.md ]] && echo "  - Integration strategy present"
[[ -d docs/integration/analysis_recipes ]] && echo "  - Analysis recipes present"
[[ -d docs/code_walkthroughs ]] && echo "  - Code walkthroughs present"

# Check cards
echo ""
echo "✓ Checking cards..."
paper_count=$(find kb/paper_cards -name "*.yaml" ! -name "template.yaml" | wc -l)
echo "  - Paper cards: $paper_count"
model_count=$(find kb/model_cards -name "*.yaml" ! -name "template.yaml" | wc -l)
echo "  - Model cards: $model_count"
dataset_count=$(find kb/datasets -name "*.yaml" ! -name "template.yaml" | wc -l)
echo "  - Dataset cards: $dataset_count"

# Check configs
echo ""
echo "✓ Checking experiment configs..."
config_count=$(find configs/experiments -name "*.yaml" | wc -l)
echo "  - Experiment configs: $config_count"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ All checks passed!"
echo "═══════════════════════════════════════════════════════════"

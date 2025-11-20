---
title: "{{CARD_TITLE}} — Integration Guidance Card"
status: draft
updated: {{DATE}}
tags: [integration, {{TAGS}}]
---

# {{CARD_TITLE}}

**Source:** {{AUTHORS}} ({{YEAR}}), {{VENUE}}  
**Type:** {{INTEGRATION_PATTERN_TYPE}}  
**Best for:** {{PRIMARY_USE_CASES}}

---

## Problem It Solves

**Challenge:** [Describe the integration challenge this approach addresses]

**Solution:** [High-level description of the solution approach]

**Why traditional approaches fail:** [Key limitations of naive or simpler methods]

---

## Core Mechanics

### 1. [Primary Mechanism Name]

[Detailed description of how the integration works]

```python
# Code example showing key implementation pattern
```

**Key insight:** [Main technical or conceptual contribution]

### 2. [Secondary Mechanisms if applicable]

[Additional details on variants, extensions, or related techniques]

---

## When to Use

✅ **Use this approach when:**
- [Condition 1: data characteristics]
- [Condition 2: dataset size requirements]
- [Condition 3: interpretability needs]
- [Condition 4: computational constraints]

✅ **Particularly well-suited for:**
- [Specific application 1]
- [Specific application 2]
- [Specific application 3]

---

## When to Defer

⚠️ **Defer to other methods when:**
- [Condition where this approach is suboptimal]
- [Alternative scenario]
- [Computational or data constraints]

⚠️ **Consider alternatives:**
- **[Alternative 1]:** [When to use instead]
- **[Alternative 2]:** [When to use instead]

---

## Adoption in Our Neuro-Omics Pipeline

### Current Implementation

**Per-modality setup:**
- **Genetics:** [FM choice, embedding dimensions, preprocessing]
- **Brain:** [FM choice, embedding dimensions, preprocessing]
- **Fusion:** [How modalities are combined]

**Workflow:**
```bash
# Step-by-step commands for implementation
```

**Evaluation metrics:**
- [Metric 1 with rationale]
- [Metric 2 with rationale]
- [Statistical test for fusion gain]

### Integration with ARPA-H BOM

[How this approach fits into the Brain-Omics Model escalation strategy]

```
[Escalation diagram showing where this fits]
```

**Why [start with / escalate to] this approach:**
- [Rationale 1]
- [Rationale 2]
- [Rationale 3]

---

## Caveats and Best Practices

### ⚠️ [Caveat 1 Name]

**Problem:** [Description of what can go wrong]

**Solution:** [How to avoid or mitigate the issue]
```python
# Code example showing correct vs. incorrect approach
```

### ⚠️ [Caveat 2 Name]

**Problem:** [Description]

**Solution:** [Mitigation]

[Repeat for additional caveats]

---

## Practical Implementation Guide

### Step 1: [Setup Phase]

[Detailed instructions for initial setup]

| Component | Configuration | Rationale |
|-----------|---------------|-----------|
| [Item 1] | [Config] | [Why] |
| [Item 2] | [Config] | [Why] |

### Step 2: [Training/Integration Phase]

```python
# Detailed code example for core integration step
```

### Step 3: [Evaluation Phase]

```python
# Code for evaluating integration performance
```

[Repeat for additional steps as needed]

---

## Reference Materials

**Primary paper:**
- Paper Title (Authors Year) — see `../../generated/kb_curated/papers-md/{{PAPER_SLUG}}.md`

**Related integration cards:**
- {{RELATED_CARD_1}} — Brief description (link once created)
- {{RELATED_CARD_2}} — Brief description (link once created)

**KB integration guides:**
- [Integration Strategy](../../integration/integration_strategy.md) — Overall fusion approach
- [Design Patterns](../../integration/design_patterns.md) — Pattern taxonomy
- [Multimodal Architectures](../../integration/multimodal_architectures.md) — Model examples

**Analysis recipes:**
- Reference recipe: `../../integration/analysis_recipes/{{RECIPE}}.md`
- Optional second recipe: `../../integration/analysis_recipes/{{RECIPE}}.md`

**Model documentation:**
- [Genetics Models](../../models/genetics/index.md) — Gene embedding extraction
- [Brain Models](../../models/brain/index.md) — Brain embedding extraction
- [Multimodal Models](../../models/multimodal/index.md) — Fusion architectures

---

## Next Steps in Our Pipeline

1. **[Phase 1]** — [Description and deliverable]
2. **[Phase 2]** — [Description and deliverable]
3. **[Phase 3]** — [Description and deliverable]
4. **[Phase 4]** — [Description and deliverable]
5. **[Phase 5]** — [Description and deliverable]

**Success criteria for escalation:**
- [Quantitative criterion 1]
- [Quantitative criterion 2]
- [Qualitative criterion]

---

## Key Takeaways

1. **[Takeaway 1]** — [Explanation]
2. **[Takeaway 2]** — [Explanation]
3. **[Takeaway 3]** — [Explanation]
4. **[Takeaway 4]** — [Explanation]
5. **[Takeaway 5]** — [Explanation]

**Bottom line:** [One-sentence summary of when and why to use this integration approach]

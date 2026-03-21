---
description: Extract and catalog all mathematical equations, formulas, and pseudocode from a research paper. Use this when you need precise mathematical details for implementation.
---

# Equation Extraction

Extract every equation from the paper and organize them for implementation.

## Output Format

Write to `output/summaries/<paper_name>_equations.md`:

```markdown
# Equations from <Paper Title>

## Core Equations

### Eq. 1: <Name/Description>
$$
<LaTeX equation>
$$
**Variables**:
- $x$ — input tensor, shape (batch, seq_len, d_model)
- $W$ — weight matrix, shape (d_model, d_out)

**Used in**: <which component/step>
**Implementation notes**: <any gotchas for coding this>

---
(repeat for each equation)
```

## Categorization

Group equations into:
1. **Loss Functions** — objectives to optimize
2. **Forward Pass** — computation graph equations
3. **Attention / Core Mechanism** — the paper's main contribution
4. **Normalization / Regularization** — LayerNorm, dropout, etc.
5. **Initialization** — weight init schemes
6. **Metrics** — evaluation formulas

## Rules
- Preserve exact notation from the paper.
- Include equation numbers as referenced in the paper.
- For pseudocode, convert to Python-style pseudocode.
- Flag any equations with ambiguous notation.
- Note numerical stability concerns (log-sum-exp, epsilon terms, etc.).

---
description: Produce a structured, implementation-focused summary of a research paper. Use this when reading any paper (PDF, markdown, or arXiv) to ensure consistent, actionable summaries.
---

# Structured Paper Summary

Generate an implementation-focused summary that a software engineer can use to build from.

## Output Format

For each paper, write a markdown file with exactly these sections:

### Header
```markdown
# <Paper Title>
**Source**: <file path or arXiv ID>
**Date Read**: <today's date>
**Relevance**: <HIGH / MEDIUM / LOW> to the stated objective
```

### Sections (all required)

1. **Core Idea** (2-3 sentences)
   - What problem does this paper solve?
   - What is the main contribution?

2. **Relevance to Objective**
   - How does this paper help achieve the user's goal?
   - Which specific techniques are applicable?

3. **Key Techniques** (bullet list)
   - Algorithm names and brief descriptions
   - Novel architectural components
   - Training or optimization strategies

4. **Implementation-Critical Details**
   - Model dimensions, layer counts, hidden sizes
   - Learning rates, batch sizes, optimizers
   - Data preprocessing steps
   - Loss functions with exact formulas
   - Activation functions and normalization

5. **Equations** (verbatim LaTeX)
   - Copy the 3-5 most important equations
   - Include variable definitions
   - Add brief plain-English explanation for each

6. **Architecture Diagram** (ASCII art)
   - Data flow through the model
   - Key components and their connections

7. **Limitations & Caveats**
   - Known failure modes
   - Dataset or domain constraints
   - Computational requirements

8. **Implementation Priority**
   - What to build first
   - What can be deferred
   - What requires clarification

## Rules
- Be precise with numbers — never say "a few layers", say "6 layers".
- If a detail is ambiguous in the paper, flag it explicitly.
- If the paper is not relevant, keep the summary to sections 1 and 7 only.

---
description: Verify that the generated code faithfully implements the algorithms and techniques described in the source research papers. Use this during code review to check paper-to-code alignment.
---

# Paper Alignment Check

Systematically verify that code matches the paper's described methods.

## Checklist

### 1. Algorithm Fidelity
For each algorithm in the paper:
- [ ] Implementation matches pseudocode step-by-step
- [ ] Loop structures and conditionals match paper description
- [ ] Variable names map clearly to paper notation

### 2. Equation Implementation
For each equation in `output/summaries/*_equations.md`:
- [ ] Equation is implemented in code
- [ ] Operator precedence is correct
- [ ] Broadcasting/shapes match mathematical dimensions
- [ ] Numerical stability measures added where needed

### 3. Hyperparameter Defaults
| Parameter | Paper Value | Code Default | Match? |
|-----------|-------------|--------------|--------|
| (fill for each hyperparameter) |

### 4. Architecture Match
- [ ] Layer count matches paper
- [ ] Hidden dimensions match paper
- [ ] Activation functions match paper
- [ ] Normalization placement matches paper (pre-norm vs post-norm)
- [ ] Residual connections present where paper specifies

### 5. Training Pipeline (if applicable)
- [ ] Optimizer matches paper (Adam, SGD, etc.)
- [ ] Learning rate schedule matches paper
- [ ] Loss function matches paper
- [ ] Data augmentation matches paper

## Scoring

Rate alignment on a 1-10 scale:
- **9-10**: Faithful reproduction, minor notation differences only
- **7-8**: Correct overall, some implementation choices differ from paper
- **5-6**: Core idea correct, significant deviations in details
- **3-4**: Loosely inspired by paper, major components missing
- **1-2**: Does not implement the paper's techniques

## Output

Write alignment report to `output/reviews/alignment.md` with the filled checklist and score.

## Rules
- Cross-reference summaries, not just the code.
- Flag any deviation, even if the deviation is arguably better.
- Be specific: cite paper section numbers and code line numbers.

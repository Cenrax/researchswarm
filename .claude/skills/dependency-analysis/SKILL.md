---
description: Analyze and select third-party libraries and frameworks needed to implement the research paper. Use this when planning dependencies to avoid version conflicts and bloat.
---

# Dependency Analysis

Determine the minimal set of dependencies required for the implementation.

## Steps

1. Read the paper summaries and identify required capabilities:
   - Tensor operations → PyTorch, NumPy
   - Model architectures → specific framework features
   - Data loading → datasets, preprocessing libraries
   - Visualization → matplotlib, tensorboard
   - API serving → FastAPI, Flask

2. For each dependency, evaluate:
   - **Why**: Which paper technique requires it
   - **Version**: Minimum compatible version
   - **Size**: Install footprint
   - **Alternatives**: Lighter or more common options
   - **License**: Compatibility with MIT

3. Check for conflicts between dependencies.

4. Produce `output/plans/dependencies.md`:

```markdown
# Dependencies

## Required
| Package | Version | Purpose | Paper Reference |
|---------|---------|---------|-----------------|
| torch   | >=2.0   | Tensor ops, autograd | All sections |
| numpy   | >=1.24  | Array manipulation | Data preprocessing |

## Optional
| Package | Version | Purpose |
|---------|---------|---------|
| tensorboard | >=2.14 | Training visualization |

## Conflicts
- None found

## Install Command
pip install torch>=2.0 numpy>=1.24
```

## Rules
- Prefer PyTorch over TensorFlow unless the paper is TF-specific.
- Pin minimum versions, not exact versions.
- Flag any dependency over 500MB install size.
- Prefer stdlib over third-party where possible.

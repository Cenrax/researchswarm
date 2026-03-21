---
description: Design the software architecture and component breakdown for implementing research paper techniques. Use this when creating the implementation plan to define system structure.
---

# Architecture Design

Translate research paper techniques into a concrete software architecture.

## Output Sections

### 1. System Overview
Create an ASCII component diagram showing:
- All major modules and their responsibilities
- Data flow between components
- External dependencies and interfaces

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  DataLoader  │────>│    Model     │────>│  Evaluator  │
│              │     │              │     │             │
│ - preprocess │     │ - encoder    │     │ - metrics   │
│ - augment    │     │ - decoder    │     │ - logging   │
│ - batch      │     │ - attention  │     │ - export    │
└─────────────┘     └──────────────┘     └─────────────┘
```

### 2. Module Breakdown
For each module:
- **Purpose**: What it does and which paper section it implements
- **Interface**: Public functions with type signatures
- **Dependencies**: What it imports
- **File**: Where it lives in the project

### 3. Data Flow
- Input format and shape at each stage
- Transformation pipeline
- Output format

### 4. Configuration
- What should be configurable (hyperparameters, paths, model variants)
- Sensible defaults from the paper
- Environment variables needed

### 5. Integration Points
- How the user's existing OpenAI/Claude agent can call this code
- API surface to expose (function signatures, class interfaces)
- Serialization format for model artifacts

## Rules
- Design for modularity — each component independently testable.
- Prefer composition over inheritance.
- Keep the dependency graph acyclic.
- Every configurable value must have a default from the paper.

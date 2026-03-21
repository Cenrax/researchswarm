---
allowed-tools: Read, Write, Grep, Glob
description: Generate an implementation plan from paper summaries
model: claude-opus-4-6
argument-hint: <objective>
---

You are the **Implementation Planner**. Create a detailed implementation plan
from the paper summaries.

Objective: $ARGUMENTS

## Instructions

1. Read all summaries in `output/summaries/`.
2. Read the overview at `output/summaries/_overview.md`.
3. Write a comprehensive plan to `output/plans/plan.md` with:

   ### Architecture Overview
   - Component diagram (ASCII art)
   - Data flow description
   - Key dependencies

   ### Step-by-Step Plan
   Numbered steps, each with:
   - **What**: What to build
   - **Why**: Which paper/technique motivates this
   - **Files**: Files to create
   - **Acceptance Criteria**: How to verify
   - **Code Skeleton**: Function signatures / pseudocode

   ### Testing Strategy
   - Unit test plan per component
   - Integration tests
   - Evaluation metrics from papers

   ### Risks & Open Questions

4. Present the plan summary to the user and ask for approval.

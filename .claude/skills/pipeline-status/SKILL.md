---
description: Check the current state of the research-to-code pipeline by inspecting output directories for summaries, plans, code, and reviews. Use this when you need to understand what stages have been completed and what's pending.
---

# Pipeline Status Check

Inspect the output directories to determine the current state of the pipeline.

## Steps

1. Use `Glob` to list files in each output directory:
   - `output/summaries/` — paper summaries and `_overview.md`
   - `output/plans/` — implementation plan (`plan.md`)
   - `output/code/` — generated source code files
   - `output/reviews/` — review reports (`review.md`)

2. For each directory, report:
   - Number of files present
   - Key file names
   - Whether the stage appears complete

3. Produce a status summary in this format:

```
Pipeline Status:
  [x] Stage 1 — Read:    3 summaries + overview
  [x] Stage 2 — Plan:    plan.md (47 lines)
  [ ] Stage 3 — Code:    empty
  [ ] Stage 4 — Review:  empty
  Next: Stage 3 (Code)
```

4. If a stage has partial output (e.g., summaries but no overview), flag it.

## Rules
- Only read file metadata and line counts — do NOT read full file contents.
- Be concise — this is a status check, not a deep analysis.

---
description: Send code back to the coder agent with reviewer feedback for a rework iteration. Use this when the reviewer reports NEEDS_CHANGES or FAIL, and the user approves a rework cycle.
---

# Iterate with Reviewer Feedback

Orchestrate a rework loop: take the reviewer's findings and send them back to the coder agent for fixes.

## Steps

1. Read the review report at `output/reviews/review.md`.
2. Extract all issues with severity **critical** or **major**.
3. Read the implementation plan at `output/plans/plan.md` for context.
4. Compose a focused prompt for the coder agent that includes:
   - The specific issues to fix (with file paths and line references)
   - The relevant section of the plan for each issue
   - Clear acceptance criteria for each fix
5. Dispatch the **coder** agent with this prompt.
6. After the coder finishes, dispatch the **reviewer** agent again to verify fixes.
7. Report the updated review status to the user.

## Prompt Template for Coder

```
The reviewer found the following issues in the code:

{{issues}}

For each issue:
1. Read the affected file
2. Fix the issue
3. Run the code in a sandbox to verify the fix
4. Move to the next issue

Reference the plan at output/plans/plan.md for intended behavior.
```

## Rules
- Only iterate on critical and major issues — skip nits.
- Maximum 3 rework cycles before escalating to the user.
- After each cycle, present a diff summary to the user.

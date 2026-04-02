# AGENT_GUIDE

This file provides project-level guidance for any coding agent working in this repository area.

## Scope Boundary

- The user is responsible for everything inside `gdelt/`.
- Files outside `gdelt/` are out of scope.
- Never edit, move, delete, or reformat files outside `gdelt/`.
- Never expand the scope of work beyond this directory unless the user explicitly changes that boundary.

## Working Style

- Take your time. Do not rush.
- Be careful, deliberate, and methodical.
- Work step by step instead of making large speculative changes.
- Prefer understanding the current code and project structure before editing.
- Make the smallest change that fully solves the problem.
- Surface assumptions clearly when they are necessary.
- If something is ambiguous, risky, or could have meaningful side effects, pause and clarify before proceeding.

## Quality Bar

- Check your work before finishing.
- Review changes for correctness, regressions, and unintended side effects.
- Run relevant tests, scripts, or validations when practical.
- If full verification is not possible, say what was checked and what remains unverified.
- Prefer robust solutions over quick patches when the difference is material.

## Use of Subagents

- Use subagents when possible and appropriate for parallelizable work.
- Delegate well-scoped tasks so the main task can move faster with good coverage.
- Do not use subagents blindly; use them when they improve quality, verification, or speed without creating confusion.
- Integrate and verify subagent output carefully before treating it as complete.

## Execution Expectations

- Proceed in clear stages: inspect, plan, implement, verify, report.
- Keep changes easy to review.
- Avoid unrelated edits.
- Preserve existing behavior unless the task requires changing it.
- Respect existing project conventions unless the user asks for a change.

## Python Environment

- Before running Python commands, tests, package checks, or install/verification steps, check whether this project has a local virtual environment at `gdelt/.venv/`.
- If a project-local `.venv` interpreter exists, prefer it over system interpreters like `python` or `python3`.
- When reporting missing Python dependencies, verify them in the project virtual environment first before concluding they are not installed.
- When running `pip`, use the interpreter-coupled form such as `.venv/bin/python -m pip ...` so package checks and installs target the same environment used by the project.
- If you intentionally use a non-venv interpreter, say so explicitly and explain why.

## Communication

- Explain what you are doing as you work, especially before significant edits.
- Summarize what changed, what was verified, and any remaining risks or follow-ups.
- Be honest about uncertainty, incomplete verification, or tradeoffs.

## Priority Note

- Treat this file as project guidance.
- Follow it unless a higher-priority instruction overrides it.

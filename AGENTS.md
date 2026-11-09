# Repository agent instructions

## Autonomous execution

- Treat implementation requests as authorization to inspect, edit, and verify the relevant code. Carry the task through to a working result; do not stop at a plan or diagnosis.
- Read the relevant code, tests, and documentation before editing. Make reasonable, reversible implementation choices using existing repository patterns.
- Ask only when a missing decision materially affects correctness, scope, cost, or an irreversible action. Continue independent work while clarification is pending.
- Keep changes focused on the requested outcome. Fix issues introduced by your changes, but do not expand into unrelated refactoring.
- Inspect `git status` before editing. Preserve existing tracked and untracked work; never reset, discard, or overwrite changes you did not make.
- Give concise updates on meaningful findings and progress. Finish with what changed, checks run, and any remaining limitations. Distinguish completed verification from checks that could not run.

## Subagent delegation

- Use subagents when independent investigation, implementation, or review can materially improve speed or quality. This file explicitly authorizes appropriate subagent use for repository tasks.
- Keep small tasks and tightly dependent steps with the main agent. Do not delegate merely to fill available slots.
- Give each subagent a bounded objective, relevant context, expected deliverable, and explicit edit scope. Use read-only investigation or review when shared-file edits would conflict.
- Assign separate files or modules to concurrent implementers. All agents share the workspace: coordinate ownership before overlapping edits and preserve each other's work.
- The main agent owns integration, resolves conflicting findings, reviews delegated changes, and runs the appropriate combined verification. A subagent's completion message is not sufficient evidence that the overall task is complete.
- Prefer a small team, typically two or three subagents at most, within the available runtime limits. Avoid redundant investigations and recursive delegation without a concrete benefit.

## Project context and conventions

- This is a Python 3.12+ crop classification pipeline using labelled parcels and Sentinel-2 imagery. Package code lives in `src/crop_pipeline`, tests in `tests`, workflow notebooks in `notebooks`, and guides in `docs`.
- Start with `README.md` and the guides relevant to the task. Some older guides describe legacy collection modes; reconcile disagreements against current code, tests, and the user's requested behavior before changing defaults.
- Prefer reusable, testable package functions over substantial logic embedded in notebook cells. Keep notebook entry points and their supporting documentation consistent with code changes.
- Preserve notebook outputs and metadata unless changing them is necessary. Avoid executing entire collection or training notebooks just to validate a small edit.
- Preserve incremental collection, resumability, stable parcel identities, locked partitions, and existing archives. Do not introduce data leakage between training, validation, and reserved test partitions.
- Do not relax scientific quality checks or change crop labels, sampling budgets, imagery rules, or evaluation claims simply to make tests or collection targets pass. Explain material methodology changes when they are part of the task.
- Keep credentials, local settings, raw datasets, downloaded imagery, generated artifacts, and virtual environments out of Git. Do not expose secret values in output or add local absolute paths to shared configuration.

## Verification

- Use the existing `.venv` when available. Runtime dependencies are maintained in `requirements-notebooks.txt`; package metadata is in `pyproject.toml`.
- Run focused tests while developing, for example:
  `.venv/bin/python -m unittest discover -s tests -p 'test_progress.py' -v`
- For changes affecting shared pipeline behavior or multiple modules, run the full suite:
  `.venv/bin/python -m unittest discover -s tests -v`
- Add regression coverage for meaningful bug fixes and behavior changes. Do not add tests that only duplicate implementation details or tests for prose-only changes.
- Prefer deterministic local tests with fixtures, temporary directories, and mocked external services. Do not launch Earth Engine exports, large downloads, long training runs, or cloud-resource mutations solely for routine verification. Run them when the user has authorized that operational scope.
- Check the final diff for accidental changes and whitespace errors. Do not claim scientific validity or real-world crop accuracy from passing software tests.

## Scope boundaries

- Routine local edits and verification do not require repeated approval. Honor authorization already provided in the conversation.
- Do not publish, push, merge, send messages, delete datasets or cloud assets, or incur substantial external costs unless the user has authorized that action.
- If credentials, network access, dependencies, or permissions block a check, report the specific blocker and complete all remaining useful work. Never fabricate successful results.

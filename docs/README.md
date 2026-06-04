# `docs/` — Project Documentation

Free-form documentation: master SOTA log, internal planning, and the
`superpowers/` design specs that drive the architectural decisions.

## Top-level files

| File | Purpose |
|------|---------|
| `SOTA_UPGRADE_LOG.md` | Running log of every method tried, what it scored, what was kept, what was discarded. Single source of truth for the project's empirical history. |

## Subfolders

| Folder | Purpose |
|--------|---------|
| `internal/` | Internal-only notes (planning scratchpads, reading lists, supervisor-meeting prep). Not part of the deliverable. |
| `superpowers/` | Design specs and implementation plans authored via the superpowers skill set. Each spec is dated and topic-named: `YYYY-MM-DD-<topic>-design.md`, `YYYY-MM-DD-<topic>-implementation-plan.md`, etc. The PA-CODE design, Path A1 plan, and OrganMoE-3D spec all live here. |

## How docs interact with the rest of the project

- Specs in `superpowers/` are the authoritative source for architectural intent. Code is expected to match the spec; if they disagree, the spec wins for review purposes.
- `SOTA_UPGRADE_LOG.md` complements `D:\Project\VoluFormer3D\MASTER_PROJECT_LOG.md` (active V4 log) — older milestones live here, newer milestones in the master log under `VoluFormer3D_V4/`.
- Thesis-side write-ups (chapters, abstract, supervisor summary) live under `D:\Project\thesis\`, not here.

## What does **not** belong here

- Per-run output files (those live in `logs/`, `reports/`, or `thesis/results/`).
- Code or configs (those live in `models/`, `configs/`, etc.).
- Checkpoint weights (those live in `checkpoints/`).

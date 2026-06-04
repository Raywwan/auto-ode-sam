# `utils/` — Small Shared Utilities

Three minimal helpers used everywhere — kept thin on purpose. Nothing
domain-specific lives here; that belongs in `models/`, `datasets/`, or
`training/`.

## Modules

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker. |
| `checkpoint.py` | Checkpoint save/load helpers: atomic write (to a temp path then rename), schema-versioned state-dict containers, "latest" symlink/copy management, and the safety rule that **refuses to overwrite a non-empty experiment dir** (see `memory/feedback_checkpoints.md`). |
| `logger.py` | Lightweight logger wrapper around the stdlib `logging` module. Unbuffered stdout for long-running training jobs; per-run log file under `logs/<exp>/train.log`. |

## What does **not** belong here

- Config parsing — lives in `train.py` / the trainer module.
- Metric computation — lives in `evaluation/`.
- File-format IO for medical images — lives in `datasets/` next to the loader that consumes it.
- Anything imported from only one place — promote it back inline and delete the indirection.

## Standing rule

Keep this folder thin. If a util module starts depending on
`models/` or `training/`, that's a sign it belongs there, not here.

# `ideas/` — Half-Baked Plans and Brainstorm Notes

Scratchpad for exploratory write-ups that pre-date their corresponding
implementation. Anything stable has been promoted to `docs/superpowers/`
or to the thesis itself; what remains here is exploratory.

## Files

| File | Purpose |
|------|---------|
| `ARCHITECTURES_DETAILED.md` | Long-form catalogue of architectural variants considered (ODE-SAM, ACM-SAM, FCA-SAM, multiscale-ISA, TriMamba-SAM, Stage-2 noISA). Each entry documents what was tried, what scored, and what was discarded. Useful as a record of the design-space exploration but **not** intended for citation. |

## Cross-references

- For the *currently active* architectural decisions, see `models/auto_ode_sam.py` and the thesis Chapter 3 (Methodology) at `D:\Project\thesis\chapters\03_method.tex` (read-only).
- For *negative results* with their diagnostic data, see `D:\Project\thesis\results\pa_code_drift\` and `D:\Project\Dr_Aram\results\pa_code_negative_result\`.

## Don't add here

- Code, configs, or tests — those have their own top-level folders.
- Authoritative design specs — those belong in `docs/superpowers/specs/`.

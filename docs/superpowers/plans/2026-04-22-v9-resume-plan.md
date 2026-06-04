# V9 — Resume Plan (as of 2026-04-22 00:20 AST)

**Author:** Claude (auto-generated)
**Status at save time:** Stage 2 v5 training task `b8sr01fhb` is at epoch 9/30. Refiner plateaued. Run will terminate when the CC session closes (local_bash task).

---

## Where we stand

- **Stage 1 proposer** is DONE. `checkpoints/v9_stage1/last.pt` → val_dice_3d_present = **0.8433**.
- **Stage 2 v5** alignment fix worked: fused=0.70 (up from 0.03 pre-fix). But refiner has **plateaued at 0.09-0.13** across 9 epochs, so cascade currently DILUTES proposer (0.70 < 0.84).
- All four 2026-04-21 alignment bugs are fixed and verified (see `MASTER_PROJECT_LOG.md` and `feedback_v9_alignment_lessons.md`).

## Decision tree on resume

```
Step 1: Is task b8sr01fhb still alive?
  YES → check latest epoch; if ep ≥ 15 and refiner still <0.20, proceed to Step 2
        if ep < 15, wait or skip to Step 2 with latest on-disk ckpt
  NO  → proceed to Step 2 with the highest checkpoint in checkpoints/v9_stage2/
        (as of save time: epoch_010.pt at 00:10 AST)
```

```
Step 2: Run 3D sliding-window eval on PROPOSER ALONE
        Why: this is the number that appears in the paper, regardless of cascade fate
        How: python evaluation/eval_v9_3d.py --ckpt checkpoints/v9_stage1/last.pt \
                                              --config configs/v9_tierB.yaml \
                                              --stage 1
        Expected time: ~2h on 30 val volumes
        Expected number: 0.80-0.85 present-only, 0.65-0.72 all-channel
```

```
Step 3: Decide cascade fate based on Stage 2 v5 result
  Scenario A: refiner never climbs (Dice < 0.30 at ep 30)
    → Accept cascade as failed novelty
    → Paper reframe: "Strong proposer + honest cascade ablation"
    → Skip Stage 3 joint; go to Tier 1/2 gains in SOTA plan

  Scenario B: refiner climbs to 0.30-0.50 at ep 30
    → Cascade partially works; run two ablations
       (a) organ_id=0 null conditioning
       (b) loss on conditioning channel only
    → If either ablation lifts refiner significantly, rebuild and retrain

  Scenario C: refiner climbs to >0.50 at ep 30
    → Cascade is alive; move to Stage 3 joint fine-tune
    → Expected +0.01-0.03 from joint training
```

## Concrete command list (ready to paste)

```bash
# Activate env
source /c/Users/Raywa/Desktop/LiteSAM3D/.venv/Scripts/activate

# Tier 0: 3D eval of proposer (the paper number)
cd /c/Users/Raywa/Desktop/VoluFormer3D_V4
python evaluation/eval_v9_3d.py \
    --ckpt checkpoints/v9_stage1/last.pt \
    --config configs/v9_tierB.yaml \
    --stage 1 \
    --output reports/v9_proposer_3d_eval.json

# Tier 0b: evaluate Stage 2 ckpts per-organ (no 3D SW, just slab eval)
python scripts/eval_refiner_per_organ.py \
    --ckpt checkpoints/v9_stage2/epoch_010.pt \
    --config configs/v9_tierB.yaml

# Tier 0c: gate-bias sweep to check if fusion can be rescued by bias tuning
python scripts/eval_gate_sensitivity.py \
    --ckpt checkpoints/v9_stage2/epoch_010.pt \
    --config configs/v9_tierB.yaml \
    --bias-sweep 0 1 2 3 4
```

## If the run terminated during logout

The Stage 2 v5 task (`b8sr01fhb`) is `local_bash`, so **it dies when the CC session ends.** Checkpoints are saved every 2 epochs, so at worst we lose 1-2 epochs of compute. Resume by either:

1. **Restarting Stage 2 from highest ckpt** — requires `--resume` flag support in `train_v9.py` (verify line ~311 area). If missing, add it: load `model.state_dict()` from the Stage 2 ckpt, load `opt.state_dict()` too, set `start_epoch = ckpt["epoch"] + 1`.
2. **Or just stop Stage 2** — refiner is plateaued anyway; further training will not lift it. Go straight to 3D eval of proposer (Step 2 above).

Recommended: option 2. No value in resuming a plateau.

## File checklist (don't lose these)

- `checkpoints/v9_stage1/last.pt` — Stage 1 winner, 0.8433. **SACRED. Do not overwrite.**
- `checkpoints/v9_stage1/epoch_050.pt` — same weights, redundant backup.
- `checkpoints/v9_stage2/epoch_0{02..10}.pt` — Stage 2 v5 partial run.
- `checkpoints/v9_stage2_buggy_preFix/` — archived.
- `checkpoints/v9_stage2_buggy_preAlign/` — archived.
- `MASTER_PROJECT_LOG.md` — full dated timeline. APPEND after any new run.
- `configs/v9_tierB.yaml` — the only config in use. Do not fork without renaming.
- `datasets/amos22_v9.py` — contains the 4-bug-fix alignment logic. **Do not edit without re-running alignment test.**
- `models/voluformer_v9.py:58` — gate bias init = +2.0. Keep.
- `training/train_v9.py` line 207, 372 — slab-local midplane indexing fix. Keep.
- `training/trainer_v9.py` line 118, 141 — same fix for boundary/infonce losses.

## Tier 1-5 SOTA plan

Full roadmap saved to `memory/project_v9_sota_plan.md` — read that file on resume. Summary: 3D eval now → TTA → ensembles → teacher distill → BTCV/TotalSeg → writeup.

## Expected next-session flow

1. Read MEMORY.md (auto-loaded).
2. Read `project_v9_status.md` and `project_v9_sota_plan.md`.
3. Read this file (2026-04-22-v9-resume-plan.md).
4. Check if `b8sr01fhb` is alive; if not, read `checkpoints/v9_stage2/` for highest ckpt.
5. Launch Tier 0 (3D eval of proposer).
6. Update MASTER_PROJECT_LOG.md with the 3D eval results.

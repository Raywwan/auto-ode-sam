# Pre-trained Checkpoints

This directory is **excluded from git** (see `.gitignore`). The weights are
hosted externally; download them locally and place them so the layout below
matches before running `evaluate_3d.py` or any ensemble script.

## Download

> **Note:** download URLs will be filled in once the public release is staged.
> Until then, the bit-equal weights are available on the project's local
> bundle at `D:\Project\checkpoints\` (and the seed=42 weight at
> `D:\Project\legacy_v3\checkpoints\phase3_odesam_v2_256px_liver\`).
>
> Planned hosts (TBD):
> - **Primary:** HuggingFace Hub — `raywandlawar/auto-ode-sam`
> - **Mirror:** Zenodo (DOI to be assigned at thesis submission)

After download, verify integrity with the SHA-256 table below.

## Expected layout

```
checkpoints/
├── phase3_odesam_v2_256px_liver/             ← seed = 42 (V3-era headline)
│   ├── phase3_odesam_v2_256px_liver_best.pt
│   ├── phase3_odesam_v2_256px_liver_latest.pt
│   ├── phase3_odesam_v2_256px_liver_epoch020.pt
│   ├── phase3_odesam_v2_256px_liver_epoch015.pt
│   └── config.yaml
│
├── phase3_odesam_v2_seed43_40ep/             ← seed = 43 (ensemble member)
│   └── phase3_odesam_v2_seed43_40ep_best.pt
├── phase3_odesam_v2_seed44_40ep/             ← seed = 44 (ensemble member)
│   └── phase3_odesam_v2_seed44_40ep_best.pt
├── phase3_odesam_v2_seed45_40ep/             ← seed = 45 (ensemble member)
│   └── phase3_odesam_v2_seed45_40ep_best.pt
│
├── phase3_odesam_v2_seed43_bdou_ft5ep/       ← BD-DoU fine-tune (sec:abl-stability)
├── phase3_odesam_v2_fwdonly_40ep/            ← Unidirectional ODE (sec:abl-bidi)
├── phase3_odesam_v2_hdloss_rw_256px_liver/   ← HD-loss retrain
│
├── path_a1_pa_code/                          ← PA-CODE Run #1 (negative, sec:pa-code)
├── path_a1_pa_code_lr1e4/                    ← PA-CODE Run #2 (LR sweep)
├── path_a1_pa_code_gbound/                   ← PA-CODE Run #3 (bounded-γ, terminated ep6)
├── path_a1_big_solid/                        ← A1 multi-organ attempt (superseded)
│
├── medsam2/                                  ← MedSAM-2 zero-shot comparator (tab:comparison)
├── pretrained/                               ← TinyViT + VoCo pretrained backbones
└── teachers/                                 ← SAM-2 Hiera-Large teacher
```

## SHA-256 integrity (load-bearing weights)

| Seed | Path | SHA-256 |
|---|---|---|
| 42 | `phase3_odesam_v2_256px_liver/phase3_odesam_v2_256px_liver_best.pt` | `dfec2350355e8becc57f1bbfa700a6da8694331cb3e8614966b632de65b2cd76` |
| 43 | `phase3_odesam_v2_seed43_40ep/phase3_odesam_v2_seed43_40ep_best.pt`   | `5b54bc4e4c13a1bbfb6691ce442b0b2487346554fcd08be2afb669bef6cdde81` |
| 44 | `phase3_odesam_v2_seed44_40ep/phase3_odesam_v2_seed44_40ep_best.pt`   | `f686287d35fdeef19e5a25fe494ebb4eafa4904cb3a41bc5ff10a0962a854d72` |
| 45 | `phase3_odesam_v2_seed45_40ep/phase3_odesam_v2_seed45_40ep_best.pt`   | `643f57f5da2c41f562cd95c955f3719c642525ebdc4ee25d11ffcb2914ed65cd` |

Quick verify (PowerShell):

```powershell
Get-FileHash checkpoints\phase3_odesam_v2_seed43_40ep\phase3_odesam_v2_seed43_40ep_best.pt -Algorithm SHA256
```

Or (Unix-like):

```bash
sha256sum checkpoints/phase3_odesam_v2_seed43_40ep/phase3_odesam_v2_seed43_40ep_best.pt
```

## What each checkpoint is

| Checkpoint | Train config | Role | Cited at |
|---|---|---|---|
| `phase3_odesam_v2_256px_liver/*_best.pt` | (V3-era) `phase3_odesam_v2.yaml` equivalent | Canonical seed = 42, single-seed headline DSC 0.9351 | thesis ch. 4 |
| `phase3_odesam_v2_seed43_40ep/*_best.pt` | `configs/phase3_odesam_v2_seed43.yaml` | Ensemble member | sec:repro-n4 |
| `phase3_odesam_v2_seed44_40ep/*_best.pt` | `configs/phase3_odesam_v2_seed44.yaml` | Ensemble member | sec:repro-n4 |
| `phase3_odesam_v2_seed45_40ep/*_best.pt` | `configs/phase3_odesam_v2_seed45.yaml` | Ensemble member | sec:repro-n4 |
| `phase3_odesam_v2_seed43_bdou_ft5ep/`   | BD-DoU fine-tune from seed 43 best | Loss-curriculum ablation | sec:abl-stability |
| `phase3_odesam_v2_fwdonly_40ep/`        | seed = 43 with unidirectional ODE | Bidirectionality ablation | sec:abl-bidi |
| `phase3_odesam_v2_hdloss_rw_256px_liver/` | seed = 42 with HD loss | Loss-substitution ablation | sec:abl-loss |
| `path_a1_pa_code*/`                      | various LR / γ-bound sweeps | Negative result | sec:pa-code |
| `path_a1_big_solid/`                     | A1 multi-organ extension | Superseded — kept for ablation parity | sec:pa-code |
| `medsam2/`                               | (no training, zero-shot) | External comparator | tab:comparison |
| `pretrained/`                            | TinyViT, VoCo upstream | Backbone init | sec:method |
| `teachers/`                              | SAM-2 Hiera-Large | Future distillation work | discussion |

## Licenses

The weights inherit upstream backbone licenses:
- SAM-v1 mask decoder — Apache 2.0 (Meta)
- TinyViT image encoder — Apache 2.0 (Microsoft)
- SAM-2 Hiera-Large teacher — BSD-3-Clause (Meta)
- MedSAM-2 comparator — Apache 2.0 (Bowang Lab)
- VoCo pretrained backbones — Apache 2.0

The Auto-ODE-SAM-specific layers (the Neural-ODE block, decoder fine-tune)
are released under [MIT](../LICENSE).

# Design: Per-Organ Loss Weighting + Full-Stack Copy-Paste Augmentation

**Date:** 2026-04-12  
**Status:** Approved — ready for implementation  
**Target configs:** `phase3a_autoodesam_256px.yaml`, `phase3b_autoodesam_256px_smoke.yaml`

---

## Motivation

Auto-ODE-SAM V3 trains on 15 organs simultaneously. The 5 small/rare organs —
gallbladder (4), esophagus (5), right adrenal (11), left adrenal (12), duodenum (13) —
occupy 5–50× fewer voxels than the liver or spleen. Without explicit weighting:

- The loss is dominated by large-organ gradient signal.
- The model learns to segment liver at 0.94 DSC while adrenals stagnate at 0.60.
- Mean DSC looks acceptable but the thesis examiner will ask about per-organ results.

These two features directly target small-organ performance with low implementation risk.

**Expected gains (15-organ mean):**
- Per-organ weighting: +0.5–1.5% DSC (concentrated in small organs)
- Copy-paste augmentation: +0.5–2.0% DSC (small organs only — large organs unaffected)

---

## Feature 1: Per-Organ Loss Weighting

### Approach

Batch-mean scalar scaling: after computing the scalar `total_loss`, multiply by the
mean organ weight of all items in the current batch. This is the correct approach
given that AutoODESAM batches mix multiple organ types.

### Organ Weights

Justified by inverse approximate organ volume (smaller organ → higher weight):

| Organ | ID | Weight | Rationale |
|-------|-----|--------|-----------|
| spleen | 1 | 1.0 | Large |
| right kidney | 2 | 1.0 | Large |
| left kidney | 3 | 1.0 | Large |
| gallbladder | 4 | **2.0** | Small, irregular shape |
| esophagus | 5 | **2.0** | Small, tubular |
| liver | 6 | 1.0 | Very large |
| stomach | 7 | 1.0 | Large, variable |
| aorta | 8 | 1.0 | Medium |
| inferior vena cava | 9 | 1.0 | Medium |
| pancreas | 10 | **1.5** | Medium-small, challenging |
| right adrenal | 11 | **2.0** | Very small |
| left adrenal | 12 | **2.0** | Very small |
| duodenum | 13 | **2.0** | Small, C-shaped |
| bladder | 14 | 1.0 | Large when full |
| prostate/uterus | 15 | 1.0 | Medium |

Weights are hardcoded as a module-level constant `_ORGAN_LOSS_WEIGHTS` in `trainer.py`.
No config entry needed — these are thesis-motivated constants, not hyperparameters to tune.

### Implementation

**File:** `training/trainer.py`

**Step 1** — Add constant near top of file (after imports):
```python
# Per-organ loss weights: upweight small/rare organs (inverse approximate organ volume).
# Gallbladder, esophagus, adrenal glands, duodenum are 5-50x smaller than liver/spleen.
_ORGAN_LOSS_WEIGHTS: dict = {
    4: 2.0, 5: 2.0, 10: 1.5, 11: 2.0, 12: 2.0, 13: 2.0
    # All unlisted organs default to 1.0
}
```

**Step 2** — In `_train_epoch`, immediately after the loss computation block, before
`scaler.scale(total_loss).backward()` (or `total_loss.backward()`):
```python
# Per-organ loss weighting: scale loss by mean organ weight of this batch.
# organ_id is (B,) long tensor — already on device from the AutoODESAM branch.
if organ_id is not None:
    batch_weights = torch.tensor(
        [_ORGAN_LOSS_WEIGHTS.get(int(o), 1.0) for o in organ_id],
        dtype=total_loss.dtype, device=total_loss.device
    ).mean()
    total_loss = total_loss * batch_weights
```

**No config changes required.**

---

## Feature 2: Full-Stack Copy-Paste Augmentation

### Approach

In-dataset augmentation inside `AMOS22_3D_Dataset.__getitem__`. The donor organ
region is pasted across all D slices of the current sample's stack, giving the
ODE cross-slice module coherent 3D examples of rare organs.

Paste is restricted to:
- **Same organ type**: donor and target share `organ_id`. Prevents label confusion.
- **Small organs only**: `SMALL_ORGAN_IDS = {4, 5, 11, 12, 13}`. No benefit pasting liver.
- **Same spatial position**: donor bbox pasted at identical (y1:y2, x1:x2) coordinates.
  Anatomically valid because all volumes are resampled to the same spacing (1.5mm³).

### Data Flow

```
__getitem__(idx)
  ├── load image_stack (D, H, W) numpy          [existing]
  ├── load all_masks list of D numpy arrays      [existing]
  │
  ├── [NEW] if copypaste_prob > 0 and organ_id in _donor_index:
  │     call _copypaste_aug(image_stack, all_masks, organ_id)
  │     → returns modified (image_stack, all_masks)
  │     update center_mask = all_masks[center_local]
  │
  ├── normalise + resize stack                   [existing]
  └── return batch dict                          [existing]
```

### `__init__` additions

Accept new parameter:
```python
copypaste_prob: float = 0.0
```

After building `_small_fg_indices`, build donor index:
```python
SMALL_ORGAN_IDS = {4, 5, 11, 12, 13}
self.copypaste_prob = copypaste_prob
self._donor_index: Dict[int, List[int]] = {oid: [] for oid in SMALL_ORGAN_IDS}
for i, s in enumerate(self.samples):
    oid = s.get("organ_id", 0)
    if oid in self._donor_index:
        self._donor_index[oid].append(i)
```

### `_copypaste_aug` method

```
_copypaste_aug(image_stack: np.ndarray,   # (D, H, W) — modified in-place
               all_masks: List[np.ndarray], # D × (H, W) — modified in-place
               organ_id: int
               ) -> Tuple[np.ndarray, List[np.ndarray]]
```

Algorithm:
1. Pick random donor index from `self._donor_index[organ_id]`; avoid self-paste
   (`while donor_idx == current_idx`, max 3 retries — if same index, skip aug)
2. Load donor's image slices and binary masks for the same D-slice window:
   - Use same npy_cache logic already in `__getitem__` (same `_npy_cache_dir` helper)
   - Donor `sl_start`, `sl_end` computed from donor sample's `center_slice`
3. For each depth `d` in 0..D-1:
   - `donor_mask_d = (donor_label_d == organ_id).astype(np.uint8)`
   - If `donor_mask_d.sum() < 50`: skip (organ absent or too small in this slice)
   - Compute tight bbox: `y1,y2,x1,x2 = bbox_with_margin(donor_mask_d, margin=4)`
   - Clip bbox to image bounds
   - Overwrite: `image_stack[d, y1:y2, x1:x2] = donor_img_d[y1:y2, x1:x2]`
   - Union mask: `all_masks[d] = np.maximum(all_masks[d], donor_mask_d)`
4. Return `(image_stack, all_masks)`

Helper `_bbox_with_margin(mask, margin)`:
```python
rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
y1, y2 = rows.argmax(), len(rows) - rows[::-1].argmax()
x1, x2 = cols.argmax(), len(cols) - cols[::-1].argmax()
H, W = mask.shape
return max(0, y1-margin), min(H, y2+margin), max(0, x1-margin), min(W, x2+margin)
```

### Trainer change

When constructing `train_dataset` in `Trainer.__init__`, pass:
```python
copypaste_prob=getattr(cfg.training, "copypaste_prob", 0.0)
```
Val dataset always gets `copypaste_prob=0.0` (no augmentation at validation).

### Config changes

**`phase3a_autoodesam_256px.yaml`** — add under `training:`:
```yaml
copypaste_prob: 0.3    # paste small organs at 30% of small-organ samples
```

**`phase3b_autoodesam_256px_smoke.yaml`** — add under `training:`:
```yaml
copypaste_prob: 0.1    # lower for smoke test — reduce noise over 2 epochs
```

---

## Files Changed Summary

| File | Change |
|------|--------|
| `training/trainer.py` | Add `_ORGAN_LOSS_WEIGHTS` constant; 5-line weighting block in `_train_epoch`; pass `copypaste_prob` to train dataset |
| `datasets/amos22.py` | Add `copypaste_prob` param; build `_donor_index`; add `_bbox_with_margin` helper; add `_copypaste_aug` method; 6-line call in `__getitem__` |
| `configs/phase3a_autoodesam_256px.yaml` | Add `copypaste_prob: 0.3` |
| `configs/phase3b_autoodesam_256px_smoke.yaml` | Add `copypaste_prob: 0.1` |

**No changes to:** `losses.py`, `models/`, `inference/`, other configs.

---

## Constraints and Edge Cases

- **Donor == target**: If the randomly selected donor is the same sample as the current
  item, skip augmentation (up to 3 retries, then no-op). Prevents circular paste.
- **Organ absent from all D slices**: If the donor has no organ present in any of the D
  slices (rare edge case for small organs near volume boundary), the method returns the
  unmodified stack — no crash.
- **Bbox clipping**: All bbox coordinates are clipped to `[0, H)` and `[0, W)` to handle
  margin overflow at image edges.
- **Val set**: `_donor_index` is built but `copypaste_prob=0.0` — the index is never
  accessed during validation. Zero runtime cost.
- **SMALL_ORGAN_IDS consistency**: Uses `{4, 5, 11, 12, 13}` — identical to the existing
  `SMALL_ORGAN_IDS` in `__init__` (foreground oversampling). No divergence.

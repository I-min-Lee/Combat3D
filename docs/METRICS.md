# Metrics: which numbers are valid, and which are retracted

**Read this before quoting any number from an earlier draft, slide, or commit message.**
Three figures that circulated widely during development were produced by a silent
implementation bug and are **withdrawn**. Anything downstream of them — including one
headline claim — is also withdrawn.

---

## 1. Retracted numbers

| Number | Status | Why |
|---|---|---|
| ~~end-to-end 190.3 mm~~ | ❌ **retracted** | produced by the `calibrate()` axis bug — see the section below |
| ~~"173 mm差额"~~ | ❌ **retracted** | same bug |
| ~~jitter ratio 0.51~~ | ❌ **retracted** | same bug; true value **0.83–0.89** |
| ~~"residual rotation 30°"~~ | ❌ **retracted** | dirty fit; true value **3.7°** |
| ~~"173 mm全在对齐层"~~ (the claim) | ❌ **retracted** | the reconstructed number is 27.5 mm, and it is not all in the alignment layer |
| ~~104 mm (root head, 828 npz)~~ | ❌ **removed from the reported numbers** | superseded; reporting both datasets invites misreading. Report **9 mm** only. |
| ~~from-scratch baseline 120.3 / 149.1 mm~~ | ❌ **discarded** | measures data efficiency, not architecture — unfair protocol |
| ~~from-scratch MixSTE 338.5 mm~~ | ❌ **discarded, and not shipped** | same reason; the run is under-trained rather than representative. MixSTE is out of the table entirely — only ONNX weights are public, so the same-protocol warm start is impossible. |

**Separating warm-started from from-scratch runs at a glance.** A run warm-started via
`FT_INIT` begins near `loss ≈ 0.097`; a from-scratch run begins near `loss ≈ 0.72`. If your
retrain logs show the latter, `FT_INIT` did not take effect and the resulting number is not
the one reported here. The checkpoints of the discarded runs are listed (and excluded) in
[`../weights/README.md`](../weights/README.md).

## 2. The bug that caused the largest retraction

In `mb/kb/kb_train.py`, `calibrate()` takes a model output of shape `(B, T, 17, 3)` and
zeroes the root:

```python
out = out - out[:, 0:1, :]        # WRONG — dim 1 is TIME, not the joint axis
out = out - out[:, :, 0:1, :]     # correct
```

The wrong version subtracts **frame 0's entire skeleton from every frame**. The
least-squares global scale `s` then comes out at only 0.55–0.75× of its true value — and
every millimetre metric is multiplied by that `s`.

| | Before | After |
|---|---|---|
| end-to-end, 15 test takes | 190.3 mm | **27.5 mm** |
| fitted "residual rotation" | 25–35° | **3.7°** |
| frame-to-frame displacement ratio | 0.51 | **0.89** |

**Lesson, generalised:** on multi-rig datasets, the **per-`(take, view)` scale
convention** is the layer most likely to be silently wrong. It raises nothing, and its
output looks plausible.

Backup of the pre-fix file: `kb_train.py.bak_20261004_085904_calzeroaxis`.

## 3. Also check the *frame*, not just the number

| Trap | Consequence |
|---|---|
| `27.5 mm` on 15 test takes | includes 6 `sword3` takes that overlap training → **not** a held-out number. Use the **9-take** figure. |
| `18.4 mm` (val) | single validation take (`train01_hugging`); it is the model-selection set and is **optimistic by construction**. |
| arm-to-arm validation comparisons | **invalid** — each arm validates against its own npz labels. See the aggregation warning in `REPRODUCE.md`. |
| root-relative vs absolute | MPJPE/PA are root-relative and **cannot** see absolute position. Absolute placement needs the orthogonal λ metric. |

---

## 4. Valid numbers

### 4.1 Main table — 9 genuinely held-out takes, against **official** Harmony4D GT

Metric: MPJPE / PA-MPJPE in mm, root-relative, 13 shared joints,
per-`(take, view)` `(s, R)` fitted.

| Arm | MPJPE | PA |
|---|---|---|
| A — official GT labels, 69 takes | 58.1 | 26.6 |
| B-69 — our labels, 69 takes | 65.6 | 38.2 |
| **B — our labels, 150 takes** | **42.4** | **32.6** |
| C — fully self-built, 123 takes *(exploratory)* | 116.9 | 91.6 |
| Official MotionBERT, zero-shot | 94.3 | 66.8 |

Derived, controlled comparisons:

| Quantity | Value |
|---|---|
| Label-layer cost (A → B-69) | **+7.5 mm (+13%)** |
| Data-volume gain (B-69 → B, 69 → 150 takes) | **−23.2 mm (−35%)** |
| B vs A, despite 13% label cost | **−27%** |
| Self-built upstream cost (B → C) | +74.5 mm (2.8×) |

### 4.2 Absolute positioning (λ metric, median)

| | |
|---|---|
| geometric prior | **996 mm** |
| root-regression head (full data) | **9 mm** |

### 4.3 Cross-domain

| | H4D (in-domain) | Panoptic (independent mocap) |
|---|---|---|
| Official MotionBERT zero-shot | 94.3 / PA 66.8 | 114.9 / PA 101.2 |
| Combat3D fine-tuned | — / **PA 32.6** | — / **PA 74.0 (−27%)** |

### 4.4 Same-protocol baselines (fine-tuned from public weights)

| Method | Zero-shot (15 takes) | Fine-tuned (val) | **Fine-tuned, 9 held-out** (same protocol as the main table) |
|---|---|---|---|
| VideoPose3D (detectron) | 276.6 / 174.2 | **106.1** | **188.3 / 138.7** |
| VideoPose3D (cpn) | — | 115.9 | **186.9 / 137.4** |
| MixSTE | 154.8 / 112.2 | *(ONNX only — omitted)* | — |
| **Combat3D-Mono** | **15.8 / 10.4** | **18.4** | **42.4 / 32.6** |

**Comparable reading (identical held-out protocol): VideoPose3D 188.3 vs ours 42.4 — 4.4×.**
(The val-only reading is 106.1 vs 18.4 = 5.8×; val is the model-selection set, so quote the
held-out column.)

### 4.5 Diagnostic and scene quantities

| Quantity | Value |
|---|---|
| Our labels re-projection residual | **6.0 px** |
| Official GT after alignment | **16.5 px** |
| Official GT read without alignment | 746 px *(wrong)* |
| Label quality (`em_off` vs official GT, similar-transform aligned) | **≈ 21 mm** |
| Self-built 2D vs official 2D | **14.9 px** |
| Self-built 2D chain re-projection / official 2D chain | **31 px / 6.9 px** |
| True-value frame-to-frame displacement | H4D **13.8 mm/frame** vs Panoptic **1.4 mm/frame** |
| Person-selection accuracy, before/after position prior | **72.0% → 95.7%** |
| End-to-end error, before/after position prior | **383.2 → 39.1 mm** |
| Root-regression head, 828 npz → 1800 npz | 104 mm → **9 mm** (2.4× data → 11.6× gain) |
| Triangulation, dirty upstream: ours vs naive | **20.47 vs 36.75 (−44%)** |
| Frame filtering ablation | A 56.7→72.2, B **18.4→56.4**, C 151.7→255.5 |

---

## 5. Three rules for any cross-method comparison

1. **Fix "root-relative or not" and the unit before comparing.** Getting this wrong once
   produced a 200× error.
2. **The triangulation side is in COLMAP units** and must be multiplied by that take's
   scale factor. Panoptic is in **centimetres** (`ms = 0.01`).
3. **Evaluate against ground truth, never against your own labels.** Self-certification
   is how the 15-take figure survived as long as it did.

## 6. Metric conventions worth restating

* PA (Procrustes-aligned) uses a per-frame optimal rotation + scale + translation → this
  is the **shape** lower bound.
* End-to-end uses a **single** `(s, R)` per `(take, view)`, root already zeroed, and
  **excludes absolute position**. It is therefore irrelevant to tactical-layer analysis —
  which is exactly why the absolute layer is reported separately (`LOCALIZATION` section in `REPRODUCE.md`).
* SA-Metric: report shape and alignment **separately**. Reporting only end-to-end makes a
  model that learned the motion very well look worthless.
* Every metric is computed in **millimetres** after applying the take's scale.

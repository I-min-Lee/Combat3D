# Pitfalls, ranked by the time they cost

Fifteen engineering traps from building this pipeline end-to-end, plus the four that came
from the ablation work. Most of them fail **silently** — they produce a plausible number
that is simply wrong.

---

## Tier 1 — silent, and they corrupt your results

### 1. `pip` pulled in a CUDA-13 package and broke every convolution

**Symptom:** every conv layer raises `CUDNN_STATUS_NOT_INITIALIZED`.

**Cause:** `pip install local-attention` dragged in `nvidia-cudnn-cu13`, which forced
torch's cuDNN to 9.24 against a cu121 build.

**Fix:**
```bash
pip install --force-reinstall --no-deps nvidia-cudnn-cu12==9.1.0.70
```

**Rule: do not install or upgrade packages in this environment.**

### 2. `calibrate()` zeroed the wrong axis → end-to-end error inflated 6×

`out - out[:, 0:1, :]` slices the **time** axis, not the joint axis, on a `(B,T,17,3)`
tensor. The global scale `s` comes out 0.55–0.75× of truth and every mm metric inherits
it.

**Fix:** `out[:, :, 0:1, :]`. End-to-end goes 190.3 → **27.5 mm**. See `METRICS.md`.

### 3. Official GT is in a different world frame → 746 px re-projection

**Fix:** per-take Umeyama, **inverse direction**. `umeyama(a=em_off, b=official)` gives
`official ≈ sR·a + t`; you need `a_new = Rᵀ(official − t)/s`. Reversed, 746 px becomes
3357 px. Correct: **16.5 px**.

### 4. Official GT is COCO17; our npz is H36M17

Silently mismatched joints. Use the composite `IDX17_FOR_13 → F13_TO_COCO` mapping, and
compare on the **13 joints** both layouts share (the official COCO17 eyes/ears have no
counterpart).

### 5. Unmapped joints left with `valid=True`

The model is then trained to predict `(0,0,0)` for joints that have no label.
**Every unmapped joint must set `valid=False`.**

### 6. Missing pelvis joint (`joint0`) in the A/B data

The calibration scale comes out at 219 instead of 405 and the loss explodes.
**Fix:** synthesise `joint0` from the midpoint of the two hips and set `valid=True`.

### 7. Raw-level leakage

Test subsequences get trained on. **Exclude by original sequence**, not by take:
`06_sword3`, `16_mma5`, `05_sword2`. The six `001…006_sword3` held-out takes overlap
training sources.

### 8. Cross-arm validation numbers are not comparable

`kb_train.evaluate()` reads validation GT from *that arm's own* npz. Comparing across arms
gives "our labels win by 10%"; the comparable protocol gives the **opposite** result.
**Fix:** fix the ground-truth source across arms.

---

## Tier 2 — loud failures, obscure messages

### 9. `--match` must be an integer

`invalid int value` — strip any suffix from the sequence directory name.

### 10. `assemble_h4d.py` takes `--raw`, not `--vp` / `--det`

The error message does not say so.

### 11. Concurrent ViTPose / detection runs fail silently

Two jobs sharing a GPU cause silent failures and contention. **Run serially**, and do not
use a GPU someone else has claimed.

### 12. `render3d` `--bounds` / `--floor` are in *the dataset's* units

Pass mm where the dataset is metres and the figure collapses to a single point.
**H4D is in metres.**

### 13. No `ffmpeg` in the container

`mp4` output is empty. Encode with `cv2.VideoWriter(*'mp4v')`.

---

## Tier 3 — reproducibility hazards

### 14. `min_delta=1.0` means the best checkpoint never lands on disk

Saving requires `cur < best - min_delta`. We watched a 17.6 mm epoch be discarded while
18.4 mm stayed on disk. Lower `--min-delta` if the last millimetre matters.

### 15. `/workshop` exists only inside the container

Path tests from the WSL side (`-f`, `-d`) are always false. **Do path checks inside the
container.**

### 16. Inlining commands through `wsl.exe` eats `$VAR` and `/path`

This failed more than six times. **Always write commands to a file and execute the file.**

### 17. Editing a local `.py` without re-pushing

The server runs from `/tmp/scr`. Re-push after every edit or you will debug the previous
revision.

---

## From the ablation work

### 18. A switch that "does nothing" may just be tested in the wrong regime

`tri_ablate.py`'s `AB_NO_ENUM=1` shows **zero** difference on clean upstream input — the
enumeration searches downward from the full view set, and when the full set is already
optimal, enumerating changes nothing. We concluded the switch was broken. **To verify a
switch took effect, test it on dirty upstream.**

### 19. Frame-level quality filtering is a net loss at this data scale

Enabling the built-in 10 px filter made all three arms worse
(A 56.7→72.2, B 18.4→56.4, C 151.7→255.5). **The marginal value of data volume exceeds
that of label cleanliness here.**

### 20. Enabling quality filtering crashes on H4D

Removing `--no-quality` makes `kb_common.load_quality()` look for armored stick-fighting's
`_segqa_summary.txt`, which does not exist on H4D →
`FileNotFoundError`. **Workaround:** `touch $B/_segqa_summary.txt` (an empty file leaves
report-level quality empty while keeping frame-level QA active).

### 21. A "target" written as `(G @ Rᵀ)/s` destroys the shape

`R` differs per `(take, view)`, and the model only sees 2D — it cannot know which camera
it is. Loss falls monotonically while validation worsens; PA degrades 38.1 → 50.0.
**Force `KB_FORCE_R_I=1`.**

---

## Anti-patterns in the modelling itself

These were all tried and all **lost** to a simple rule. They are recorded so nobody
re-runs them.

| Approach | Outcome |
|---|---|
| Complex 3D association (v3 / v4 / v5) for person selection | **Lost** to "top-2 by area + position prior" |
| Temporal smoothing for low-frequency mis-association | Does not help; **larger savgol windows are worse** |
| Cross-view-consistency person selection (our own) | **0/3** — lost every time; per-frame, no tracker |
| Scale-convention "fixes" by re-calibrating | Change the evaluation convention and make arms incomparable |

---

## Two project-level rules

* **Never let triangulation into an inference path.** It is legal as a training label or
  an offline ground truth only. Legal inference inputs are exactly: one view's 2D +
  camera intrinsics/extrinsics + model weights.
* **Back up before changing anything, and never sweep a directory.** Batch edits operate
  on an explicit file list; dry-run first; if an unrecognised path appears in the output,
  stop.

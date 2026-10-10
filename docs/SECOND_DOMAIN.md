# Second-domain evidence: the same back-end on a different sport

The design principle in the README —

> **the detection-box and identity layers are scene-specific; the triangulation layer and
> everything below it are universal**

— is stated as a claim about *layers*, not about a dataset. A single-domain result cannot
支持 it. This document records the second domain we ran, and states honestly what it does
and does not establish.

> **Status.** The armored stick-fighting material was produced on a separate machine. A **runnable sample is
> now released** as `Combat3D-armored stick-fighting-1.1-sample.zip` on the Releases page: 1000 frames of
> real footage with 3D labels, raw and identity-corrected 2D, calibration, the domain
> weights, and — the point of the exercise — **both scene-specific script families**
> (detection and identity) alongside the universal ones.
> Nothing in this document is used by the Harmony4D numbers.

---

## 1. The claim, stated so it can be falsified

If the principle is right, then moving to a new domain should look like this:

* the two front-end layers have to be **re-implemented**, and the new implementation
  should look **different** from the old one — not a parameter tweak;
* the back-end layers should run **unchanged**, with only their input paths repointed.

If instead the front-end port needed nothing but a config change, the layer would not
really be scene-specific. If the back-end needed modification, the "universal" half is
misnamed.

## 2. What the two domains look like

| | Harmony4D (main results) | Armored stick-fighting (second domain) |
|---|---|---|
| Sport | wrestling / jiu-jitsu / MMA / sword | armored stick-fighting |
| Cameras | 6 views, 3840×2160, fisheye | 5 views, **two groups**: 960×720@200 fps and 1920×1440@25 fps |
| Boxes + identity | shipped by the dataset | **must be built**: RT-DETR + position prior, then jersey-colour identity calibration |
| 2D | dataset's own `poses2d` (SMPL-fit projections) | ViTPose on our own boxes |
| Existing ground truth | none (that is the point) | none; triangulation used as the reference |
| Native fps | 20 | 200 / 25 |

Two different sports, two different camera systems, two different frame rates, two
different resolutions — and, critically, a **different box/identity source**.

## 3. Layer-by-layer: what changed and what did not

| Layer | Harmony4D | Armored stick-fighting | Verdict |
|---|---|---|---|
| Calibration | `h4d_calib.py` (COLMAP SfM) | own calibration (`calib_idfix`, `calib_B_1920x1440`) | **re-implemented** |
| Frames | `h4d_frames.py` | own frame indexing + `--fps-src` handling | **re-implemented** |
| Detection boxes | dataset's own | **RT-DETR + position prior** (`border-margin`) | **re-implemented, differently** |
| Identity | dataset's own | **jersey-colour calibration** → a per-view piecewise-constant flip curve + colour matching on single-person frames | **re-implemented, differently** |
| 2D pose | ViTPose / dataset's `poses2d` | ViTPose | reuse |
| **Triangulation** | `tri_h4d.py` | triangulation output used as the training label, **unchanged** | **reused** |
| **Monocular lifter** | fine-tuned MotionBERT | fine-tuned MotionBERT, same recipe | **reused** |
| **Root head** | `rh_h4d_v04_w0_full.pt` | `roothead_v1.pt`, same architecture and training script | **reused** |
| **Render** | `render3d_5000f13.py` | `render3d_5000f13.py`, recorded as used **unchanged** | **reused** |

The two front-end cells are the point. The Harmony4D identity layer is "take the
dataset's person ids"; the armored stick-fighting identity layer is "build a per-view flip curve from
jersey colour, then match colour on single-person frames". Those are not the same
algorithm with a different constant — they are different algorithms, which is what
"scene-specific" predicts.

Meanwhile the back end — triangulation, lifter, root head, render — did not change.

## 4. What this does NOT establish

State these plainly; a reviewer will find them otherwise.

1. **The metrics are not comparable across the two domains.** Harmony4D is reported as
   root-relative MPJPE / PA in mm against official held-out GT. Armored stick-fighting is reported as
   **absolute root-position error** (median 98–181 mm across three takes) against a
   **triangulation-derived** reference. Different quantity, different reference.
2. **The armored stick-fighting reference is our own triangulation**, not an independent ground truth. By
   a single-domain rule, which is weak evidence — it can show that the monocular path
   agrees with the multi-view path, not that either is correct.
3. **The armored stick-fighting root head error (≈86 mm median depth, validation) is ~10× the Harmony4D
   head (9 mm).** Different camera geometry, different validation take, different scale of
   scene. Do not present these side by side as if they were the same measurement.
4. **Single sport, single institution.** "Two domains" means two, not many.
5. The armored stick-fighting code is a **divergent branch**, not this repository. Treat the second-domain table above as
   documentation of what was done.

## 5. The released sample

`Combat3D-armored stick-fighting-1.1-sample.zip` (Release asset) contains take 1.1, frames 000000–000999:

| Asset | What |
|---|---|
| `data/1.1/final13/pid{0,1}/keypoints3d/` | the 3D labels — 13 joints, metres, calibration frame, Y down |
| `data/1.1/vitpose/{1,3,4,7,11}/` | **raw** ViTPose 2D — `personID` flips arbitrarily per (frame, view) |
| `data/1.1/annots/{1,3,4,7,11}/` | the same 2D after identity correction |
| `data/1.1/video_view*.mp4` | matching video segments, for overlay |
| `calib/{intri.yml,extri.yml}` | the 960×720 group's intrinsics and extrinsics |
| `scripts/scene_specific/` | RT-DETR detection + the jersey-colour identity family |
| `scripts/universal/` | triangulation, SMPL fit, lifter, root head, render |
| `scripts/drivers/` | `w_run_take7.sh` — the single-take 15-step chain |
| `weights/` | `kb_full_v3.pt` + its `(s, R)` calibration, `roothead_v1.pt` |

The raw-vs-corrected 2D pair is the reproducible demonstration: run the identity layer and
the same 1000 frames become usable; skip it and residuals are 120–660 px.

## 6. Still to do before this is citable as evidence

1. **Re-express the armored stick-fighting numbers in a metric comparable to Harmony4D**, or keep them
   explicitly labelled as a different quantity. A root-position error table is honest; a
   merged MPJPE table would not be.
2. **Add a "re-implemented / reused" column to the second-domain table above.** That table
   *is* the validation of the principle. Without it, the principle is an assertion
   supported by one dataset.
3. Note in the limitations that the armored stick-fighting reference is triangulation-derived, so it
   supports the *agreement* claim and not the *correctness* claim.
4. **Settle the sample's terms of use** — see the sample's own README §5. It is footage of
   real people from a private dataset; confirm that releasing a video sample is consistent
   with any ethics approval or consent agreement before publishing.

## 6. Related traps from the armored stick-fighting run worth carrying over

These were hit while porting, and generalise:

* **Person ids from a generic 2D model are not global ids.** ViTPose's `personID` must be
  mapped through a per-(frame, view) flip decision; skip it and you get 120–660 px
  residuals.
* **Median filtering creates stutter.** A 5-frame median on a continuous trajectory makes
  plateau-then-jump artefacts; 9.3% of frames read as completely frozen. Use
  despike → zero-phase savgol instead. Diagnostic: the "root frozen %" column; >1% is bad.
* **Predict depth along the ray, not the 3D root.** Regressing the 3D root welds the
  pixel→world mapping to the training camera; regressing `λ` along the pelvis ray keeps
  X/Y determined by the extrinsics, so a camera change needs only new extrinsics.
* **Zero-initialise the head's output layer**, so training starts from the geometric prior
  and the correction shrinks toward zero when the camera is off-nominal.
* **A geometric ground-plane prior is easy to get wrong**: the plane normal is the
  **second column** of the world→camera rotation (`R[:, 1]`), not the second row. Taking
  the row inflates the prior error from 1514 mm to 3315 mm.
* **Do not compare "speed" across native frame rates.** Unequal smoothing windows make it
  look like a 1100 vs 740 mm/s difference when the trajectories differ by 62 mm.

# Reproduction guide — Combat3D

Everything needed to go from Harmony4D raw data to every number in the paper, **in
execution order**. Each section names the wrapper in `scripts/` and the underlying
script it runs.

> **Read [`METRICS.md`](METRICS.md) before quoting any number.** Several older figures
> that circulated in early drafts are **retracted** (they were produced by a silent
> axis bug). This guide only uses valid ones.

---

## 0. Conventions used below

```bash
B=/workshop/Lym/combat3d                     # artifact root (set COMBAT3D_ROOT to override)
PY=$B/envs/miniconda3/envs/pose312/bin/python
V=01,03,04,07,09,14                          # the six Harmony4D views used throughout
RAW=$B/data/harmony4d/raw                    # <-- your unpacked Harmony4D sequences
TAKE=016_mma4                                # <-- one sequence
NF=<last frame index of that take>
ID=<Harmony4D integer sequence id>           # e.g. 15 for 016_mma4 — see §1.1
```

Two rules that will save you hours:

1. **Write every command to a file; never inline it.** Piping through `wsl.exe` eats
   `$VAR` and `/path`. This alone cost more than six failed runs during development.
2. **After editing a local `.py`, re-push it.** The server executes from `/tmp/scr`.

`scripts/run.sh` and `scripts/pull.sh` implement push-and-run / pull on top of
`sshd + docker exec`. Their password is read from the `SSH_PASS` environment variable —
set it before use, and never commit it.

---

## 1. Label production (per take) — `Combat3D-Label`

This is the zero-manual-annotation pipeline. Layers are ordered so that everything
scene-specific is at the front and everything universal is at the back.

### 1.1 Calibration — `scripts/01_calib.sh` → `code/adapters/harmony4d/h4d_calib.py`

```bash
$PY code/adapters/harmony4d/h4d_calib.py \
    --seq-root $RAW/$TAKE --out $B/calib_gt_$TAKE --views $V
```

Produces `extri.yml`, `intri.yml`, `h4d_meta.json`. **The world frame written here is the
COLMAP frame**, which is why all downstream 3D (and `gt3d_colmap`) lives in COLMAP units.

### 1.2 Frame chain — `scripts/02_frames.sh` → `h4d_frames.py`

```bash
$PY code/adapters/harmony4d/h4d_frames.py \
    --seq-root $RAW/$TAKE --frames-root $B/frames \
    --match $ID --seg 1 --views $V
```

* `--match` must be an **integer** sequence id (strip any suffix, or you get
  `invalid int value`).
* `ID` is the numeric prefix of the take directory name: `016_mma4` → `16`.

### 1.3 Detection boxes + identity — pick **one** of 1.3a / 1.3b

This layer is **scene-specific**. If your scene already ships boxes and identities, use
them — that is the cheapest and best-configured option (1.3a). Only build your own when
it does not (1.3b), and expect to tune it.

**1.3a Official boxes + identity — `scripts/03a_boxes_official.sh` → `h4d_boxes.py`**

```bash
$PY code/adapters/harmony4d/h4d_boxes.py \
    --seq-root $RAW/$TAKE --out $B/det_gt2_$TAKE --views $V --tag $TAKE
```

This is the route used for arms **A** and **B** in the paper.

**1.3b Fully self-built detection + identity — `scripts/03b_boxes_self.sh` → `det_self_final.py`**

```bash
$PY code/adapters/harmony4d/pipeline/det_self_final.py \
    --frames-root $B/frames/$ID/1 --out $B/det_self2_$TAKE --views $V --tag $TAKE \
    --start 1 --end $NF --border-margin 0.15 --ref-view 04 \
    --calib $B/calib_gt_$TAKE --device 0
```

`--border-margin` is the **position prior**: reject candidate boxes whose centre lies
outside the frame border. In isolation it moves person-selection accuracy from
72.0% → 95.7% and end-to-end error from 383.2 → 39.1 mm. It is the single highest-leverage
scene-specific change we made.

### 1.4 2D pose (ViTPose) — `scripts/04_pose2d.sh` → `vp_h4d.py`

```bash
VP_ROOT=$B VP_MATCH=$ID VP_SEG=1 \
$PY code/adapters/harmony4d/pipeline/vp_h4d.py \
    --det $B/det_self2_$TAKE --out $B/vp_self2_$TAKE \
    --views 01 03 04 07 09 14 --start 1 --end $NF
```

**Run detectors and ViTPose serially, never concurrently** — two parallel jobs sharing a
GPU fail silently.

### 1.5 Assemble — `scripts/05_assemble.sh` → `assemble_h4d.py`

```bash
$PY code/adapters/harmony4d/pipeline/assemble_h4d.py \
    --raw $B/vp_self2_$TAKE --out $B/asm_self2_$TAKE \
    --views $V --start 1 --end $NF
```

> The parameter is `--raw`, **not** `--vp` / `--det`. The error message is opaque.

### 1.6 Triangulation — `scripts/06_triangulate.sh` → `tri_h4d.py`

```bash
AB_CAL=$B/calib_gt_$TAKE AB_VIEWS=$V AB_ANNOTS=$B/asm_self2_$TAKE/annots \
AB_W_IMG=3840 AB_H_IMG=2160 AB_DET_DIR=$B/det_self2_$TAKE \
$PY code/adapters/harmony4d/pipeline/tri_h4d.py \
    --out $B/em_self2_$TAKE --start 1 --end $((NF+1)) \
    --min-conf 0.3 --lams 1.0 --order 2
```

This is the **universal** layer. Its robustification (subset enumeration over views) is a
noise-suppression device, not a universal improvement — see finding **F4** and
`scripts/20_ablation_tri.sh`.

**Triangulation is a label-production / offline-GT device only.** It must never appear in
an inference path. The legal inference inputs are exactly: *2D in one view + camera
intrinsics/extrinsics + model weights*.

### 1.7 SMPL fitting — `scripts/07_fit_smpl.sh` → `fit_h4d.py` *(optional)*

Only needed for mesh-format output and PVE-style comparisons; the reported MPJPE/PA
numbers come from the 13-joint lifter output and do not require this step.

```bash
FIT_K=0.024 $PY code/adapters/harmony4d/pipeline/fit_h4d.py <pid> \
    --start 1 --end $((NF+1)) \
    --annots $B/asm_self2_$TAKE/annots --k3d $B/em_self2_$TAKE/lam1.0 \
    --calib $B/calib_gt_$TAKE --frames $B/frames/$ID/4 --out $B/emfit_h4d
```

---

## 2. Official ground truth — `scripts/08_gt_extract.sh`, `09_gt_align.sh`

Evaluation **must** be against the official Harmony4D GT, never against our own labels
(self-certification). The `gt_*` directory we need is the one **we** extract from raw —
not the same thing as our pipeline output.

```bash
# 08 — extract official GT
$PY code/adapters/harmony4d/h4d_gt.py --seq-root $RAW/$TAKE --out $B/gt_$TAKE --views $V

# 09 — move the official 3D into our calibration world frame
$PY scripts/tools/ai2_fixalign.py
```

### The alignment direction matters (this is a real trap)

The official 3D and our calibration extrinsics are **not in the same world frame**. Read
directly, the official GT re-projects at **746 px**; ours re-projects at **6.0 px**.

The fix is a per-take Umeyama solve (our triangulation → official 3D). **The direction is
easy to invert:**

* `umeyama(a=em_off, b=official)` returns `official ≈ sR·a + t`
* what you want is the **inverse**: `a_new = Rᵀ(official − t)/s`

Inverting it the wrong way turns 746 px into **3357 px**. Correctly applied: **16.5 px**.
`ai2_fixalign.py` implements the inverse and writes `gt_<take>_aligned/`.

---

## 3. Training npz — `scripts/10…12`

Three arms, three builders. All produce `{tag}_v{view}_p{pid}.npz` with
`k2d (T,17,3)`, `k3d (T,17,4)`, `valid (T,)`, `meta`. **`k3d` is H36M17; the official GT
is COCO17** — the mapping is applied at evaluation time, never by reordering silently.

| Script | Arm | 2D source | 3D label source | Dataset | Count |
|---|---|---|---|---|---|
| `10_npz_armB.sh` → `monocular/mb_npz.py` | **B** | `asm_off_*` (official) | `em_off_*` (ours) | `data_h4d_offtri` | 1800 npz / 150 takes |
| `11_npz_armA.sh` → `scripts/tools/z31_mkab.py` | **A** | copied from B | `gt_<take>_aligned` (official) | `data_h4d_gtalign` | 828 npz / 69 takes |
| `12_npz_armC.sh` → `scripts/tools/ae1_mknpz.py` | **C** | `asm_self2_*` | `em_self2_*` | `data_h4d_selfv2` | 1476 npz / 123 takes |

`z31_mkab.py` replaces **only** `k3d` in the arm-B npz, leaving `k2d` / `valid` / `meta`
byte-identical. That is what makes A vs B-69 a clean controlled comparison: the two arms
differ *only* in where the 3D label came from.

**Leakage rule.** Test subsequences must be excluded **by original sequence**, not by
take: `06_sword3`, `16_mma5`, `05_sword2`. The six `001…006_sword3` test takes overlap
training-source sequences and are removed from every reported held-out number.

---

## 4. Training — `scripts/13…16`

### 4.1 Main arm (B) — `scripts/13_train_main.sh`

```bash
cd $B/mb/kb
KB_FORCE_R_I=1 CUDA_VISIBLE_DEVICES=<free-gpu> $PY -u -W ignore kb_train.py \
  --data $B/mb/data_h4d_offtri --mode full --init official --no-quality \
  --val-takes train01_hugging --clip-len 121 --stride 40 \
  --epochs 300 --clips-per-epoch 1500 --batch 4 --eval-every 10 \
  --patience 12 --min-delta 1.0 --lr 2e-5 --out $B/mb/ckpt/Combat3D_FULL.pt
```

Three things about this command are load-bearing:

* **`KB_FORCE_R_I=1` is mandatory.** The training target must be parameterised as a
  *camera-frame* pose. If you write it as `(G @ Rᵀ)/s` with a per-`(take, view)` fixed
  rotation `R`, the model only sees 2D and cannot know which camera it is — loss falls
  monotonically while validation worsens and the shape is destroyed (PA 38.1 → 50.0).
  With `KB_FORCE_R_I=1` the loss starts at 0.378 instead of 0.587 and validation improves
  throughout.
* **`--no-quality`** disables frame-level quality filtering. Leaving it off is a net
  loss at this data scale (**F3**). If you do enable it, note that `load_quality()` will
  look for armored stick-fighting's `_segqa_summary.txt` and raise `FileNotFoundError` on H4D —
  `touch $B/_segqa_summary.txt` is the workaround.
* **Checkpointing uses `cur < best - min_delta` with `min_delta=1.0`** — an improvement
  below 1 mm is **not saved**. We observed a 17.6 mm epoch discarded with 18.4 mm left on
  disk. Lower `--min-delta` if you care about the last millimetre.

### 4.2 Root-regression head — `scripts/14_train_roothead.sh` → `monocular/roothead_h4d_train.py`

```bash
$PY monocular/roothead_h4d_train.py --out $B/mb/ckpt/rh_h4d_v04_w0_full.pt
```

Trains on all 150 takes / 128,104 frames. Data volume is what matters here: 828 npz
(52,520 frames) gives 104 mm, the full set gives **9 mm** — a 2.4× data increase buying an
11.6× improvement. It attaches *after* any root-relative lifter.

### 4.3 Baselines, same protocol — `scripts/15_train_baselines.sh` → `scripts/tools/z03_ft.py`

```bash
FT_INIT=<path to public pretrained weights> \
$PY scripts/tools/z03_ft.py --arch vp3d  --epochs 40 --out $B/mb/ckpt/base_vp3d_ft120.pt
FT_INIT=<path to public pretrained weights> \
$PY scripts/tools/z03_ft.py --arch mixste --epochs 40 --out $B/mb/ckpt/base_mixste_ft.pt
```

**`FT_INIT` is the whole point.** It warm-starts from each method's public pre-trained
weights, on the same H4D data and the same `(take, view)` alignment. Without it the
comparison degenerates into "large-scale pre-training vs from-scratch" and is not fair —
that variant was run and **discarded** (120.3 / 149.1 mm; the from-scratch MixSTE run,
338.5 mm, is discarded too).

**Check which one you produced:** a warm-started run starts near `loss ≈ 0.097`, a
from-scratch run near `loss ≈ 0.72`. The released checkpoints are
`base_vp3d_ftpub.pt` (106.1) and `base_vp3d_ftpub_cpn.pt` (115.9).

MixSTE is omitted from the table: only ONNX weights are public, no PyTorch checkpoint.

### 4.4 Frame-filtering ablation — `scripts/16_ablation_qa.sh` → `scripts/tools/at5_retrain.sh`

Retrains all three arms with a uniform 10 px frame-level filter. All three get worse:

| Arm | No filter | With filter | Frames dropped |
|---|---|---|---|
| A (official GT) | 56.7 | **72.2** | 49% |
| B (our labels) | **18.4** | **56.4** | 35% |
| C (self-built) | 151.7 | **255.5** | 62% |

---

## 5. Evaluation — `scripts/17…19`

### 5.1 Held-out vs official GT — `scripts/17_eval_heldout.sh` → `scripts/tools/z25_gtEval.py`

```bash
$PY scripts/tools/z25_gtEval.py $B/mb/ckpt/Combat3D_FULL.pt
```

Reports against **official** GT, over the 13 joints both layouts share
(H36M17 → final13 → COCO17 composite mapping). Prints both the full 15 takes and the
9 takes with the leaked `sword3` set removed.

> ⚠️ **This is not the script the paper's table comes from.**
> `z25_gtEval.py` pools every `(take, view, pid)` into one median and reports
> **81.3 mm** for arm B. `ah1_abc.py` (§5.2) takes the per-take median first and then the
> median across takes, and reports **42.4 mm** — the paper's number. Same weights, same
> ground truth, same metric; only the pooling differs. **Quote the `ah1_abc.py` number.**

### 5.2 All arms at once — `scripts/18_eval_abc.sh` → `scripts/tools/ah1_abc.py` ★

**This is the authoritative evaluator and the source of the paper's main table.**

```bash
$PY scripts/tools/ah1_abc.py
```

Verified on the released checkpoints (2026-10-05):

| model | subset | n | MPJPE | PA |
|---|---|---|---|---|
| `A_officialGT` | 9 held-out | 9 | 58.1 | 26.6 |
| `B69_ourLabel` | 9 held-out | 9 | 65.6 | 38.2 |
| **`B_ourLabel`** | 9 held-out | 9 | **42.4** | **32.6** |
| `C_selfBuilt_v1` | 9 held-out | 9 | 115.1 | 92.8 |
| `C_selfBuilt_v2` | 9 held-out | 9 | 116.9 | 91.6 |

If your run does not reproduce these five rows, something upstream is off — check the
ground truth path and `h4d_metric_scale.json` before touching the model.

> **Cross-arm validation numbers are NOT comparable.** `kb_train.evaluate()` reads the
> validation GT from *that arm's own npz*. Arm A validates against official labels, arm B
> against ours. Comparing them yields "our labels are better by 10%" (val A 56.7 vs
> B-69 50.9); the comparable held-out protocol gives the **opposite** answer
> (58.1 vs 65.6). **Always fix the ground-truth source across arms.**

### 5.3 Per-take breakdown — `scripts/19_eval_pertake.sh` → `scripts/tools/eval_pertake.py`

Reports three quantities per take: PA shape residual (lower bound on shape error),
post-alignment end-to-end using that take's own clips, and the calibrated `R` — a large
`R` means the model's output frame is far from the camera frame.

---

## 6. Ablations

### 6.1 Triangulation layer scope — `scripts/20_ablation_tri.sh` → `scripts/tools/aw2_naive.sh`

Compares our `tri_h4d.py` against a naive all-views-equal-weight baseline
(`tri_ablate.py` with `AB_NO_ENUM=1`), on the **same 2D input**:

| Upstream 2D | Ours | Naive | |
|---|---|---|---|
| clean (official SMPL-fit projections) | **identical** | baseline | difference < 1e-6 |
| dirty (self-built detector 2D) | **20.47** | **36.75** | **−44%** |

**Why they are identical on clean input:** subset enumeration searches downward from the
full view set. When the full set is already optimal, enumerating changes nothing.

> **Pitfall:** the `AB_NO_ENUM=1` switch shows no difference on clean upstream. We
> initially concluded the switch was broken. **To verify a switch took effect, you must
> test on dirty upstream.**

### 6.2 Official 2D are SMPL-fit projections, not detections

Measured: official `poses2d` confidence is identically **1.000** per joint per frame with
17/17 usable joints; our ViTPose gives 0.58–0.89 with 15.5/17 usable. Any self-built vs
official 2D comparison is unfair by construction. This is why arm C is marked
**exploratory** and is not a primary result.

---

## 7. Visualization — `scripts/21…23`

```bash
# 21 — run the fine-tuned lifter over a take, with absolute root anchoring
#      (bone-length 850 mm + lowest-ankle-on-ground depth solve)
$PY monocular/mb_to_final13.py --take $TAKE --view 04 --tag idfix \
    --annots $B/easymocap_idfix/<mx>_<sx>/annots --calib $B/calib_idfix \
    --ckpt $B/mb/ckpt/Combat3D_FULL.pt --out $B/final13_mono_$TAKE --stride 8

# 22 — 3D nodes re-projected onto the original video (fisheye, no mesh)
$PY scripts/tools/v26_node_overlay.py --take $TAKE --view 04 --match $ID \
    --f13 $B/final13_mono_$TAKE --out $B/nodes_$TAKE

# 23 — 3D node render in world space
$PY code/stage8_render/render3d_5000f13.py --smpl $B/final13_mono_$TAKE \
    --out $B/render3d_mono_$TAKE --start 0 --end <N> --fps 20 \
    --floor=-1.187,1.643,-1.520,1.100
```

* `--bounds` / `--floor` take numbers in **the dataset's own coordinate system**. H4D is
  in **metres**; passing mm collapses the figure to a single point.
* The container has **no ffmpeg** — encode with `cv2.VideoWriter(*'mp4v')`.
* MotionBERT is trained in a 25 fps domain (the original 200 fps data was subsampled by
  stride 8), so feed 2D at stride 8 and expect a 25 fps output.

---

## 8. Figures and complexity table — `scripts/24…25`

```bash
$PY scripts/tools/z36_complexity.py > $B/complexity.txt   # motion speed / out-of-frame rate / hip distance
$PY scripts/tools/bc2_figs2.py                            # fig7 / fig8 / fig9
```

The complexity table is what justifies the domain claim quantitatively: true-value
frame-to-frame displacement is **13.8 mm/frame** on H4D versus **1.4 mm/frame** on
Panoptic — a **10×** difference.

---

## 9. Cross-domain: CMU Panoptic — `code/adapters/panoptic/`

Panoptic provides independent optical mocap and needs no login. Sequence
`160906_ian2`, frames 4256–4855, 5 views.

```bash
bash code/adapters/panoptic/run_p2.sh
```

Differences from the Harmony4D adapter, all of which will bite you:

* Panoptic uses **pinhole + 5-parameter distortion**, not fisheye.
* Units are **centimetres** (`ms = 0.01`), and the triangulation side is in COLMAP units
  and needs that per-take scale factor.
* COCO19 layout is derived from the official `body_edges`; `bodies[].id` in `{0,1}` is a
  stable ground-truth identity.

Result: `MPJPE 31.8 mm` for the self-built pipeline vs `30.8 mm` with oracle boxes, and
95.4% person-selection accuracy.

---

## 10. Verifying your run

| Check | Expected |
|---|---|
| Our labels re-project | **≈ 6.0 px** |
| Official GT after `ai2_fixalign.py` | **≈ 16.5 px** |
| Official GT read directly (no alignment) | 746 px — *this is the wrong answer* |
| Arm-B val median | ≈ 18.4 mm (single val take, optimistic) |
| Arm-B held-out vs official GT, 9 takes | **42.4 / PA 32.6 mm** |
| Label-layer cost, A → B-69 | **+7.5 mm (+13%)** |

`weights/MANIFEST_release_v2.md5` holds the MD5 of every file in the original release bundle, and
`weights/README.md` holds the checkpoint MD5s. Verify before trusting a download.

---

## 11. External assets you must supply yourself

These are **third-party** and are deliberately not vendored. The pipeline expects them
under `$COMBAT3D_ROOT`:

| Path | What | Where from |
|---|---|---|
| `$B/port/autodl-tmp/vitpose_ft.pth` | ViTPose-B checkpoint (~360 MB) | ViTPose release |
| `$B/port/smpl/` | SMPL body model (`smpl.pkl`, `J_regressor_body25.npy`) | SMPL (MPI) — **non-commercial research licence** |
| `$B/port/EasyMocap-master/` | EasyMocap core (`read_camera`, `fit`) | EasyMocap upstream |
| `$B/port/emcore/` | EasyMocap core modules used by stage5/6 | as above |
| `$B/mb/MotionBERT/` | MotionBERT repo | [Walter0807/MotionBERT](https://github.com/Walter0807/MotionBERT) |
| `$B/baselines/videopose3d/` | VideoPose3D repo (for `z03_ft.py --arch vp3d`) | [facebookresearch/VideoPose3D](https://github.com/facebookresearch/VideoPose3D) |
| `$B/baselines/mixste/` | MixSTE repo (for `--arch mixste`) | MixSTE upstream |
| RT-DETR weights | only for the self-built detector (arm C) | ultralytics |
| Harmony4D raw zips | the dataset | Harmony4D release. `scripts/tools/k9_dl_rest.py` downloads the trailing sequences via `hf-mirror` |

### The one environment file you must copy

`code/env/sitecustomize.py` must be placed in the target environment's `site-packages/`.
Python imports it automatically at startup. It fixes three things that otherwise break
EasyMocap's fit stage on Python 3.12 / numpy 2:

1. `chumpy` calls `inspect.getargspec`, removed in Python 3.11
2. `chumpy` imports `numpy.bool` / `numpy.int`, removed in numpy 2
3. EasyMocap dumps numpy scalars and arrays straight into `json.dump` → `TypeError`

Without it the SMPL fit stage fails; stages 1–6 and all evaluation are unaffected.

### Directory convention

```
$COMBAT3D_ROOT/
  envs/miniconda3/envs/pose312/   main environment (see ENV_pose312.md)
  code/                           the pipeline (this repo's code/ tree)
  monocular/                      the lifter layer (this repo's monocular/)
  scripts/tools/                  the driver scripts
  mb/kb/                          kb_train.py and friends (this repo's monocular/kb/)
  mb/data_h4d_*/                  training npz
  mb/ckpt/                        checkpoints  <-- weights/ckpt goes here
  port/                           third-party model assets (table above)
  data/harmony4d/{zips,raw}/      the dataset
  frames/  calib_gt_*/  det_*/  vp_*/  asm_*/  em_*/  gt_*/   per-stage artifacts
  h4d_metric_scale.json           <-- this repo's configs/
```

`scripts/00_env.sh` prints this layout check for you.

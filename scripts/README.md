# Call order

Run the numbered wrappers **in this order**. Each is a thin, readable wrapper around the
underlying script (named in the second column); the full explanation of every step,
argument and trap is in [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

```bash
export COMBAT3D_ROOT=/path/to/your/artifacts     # default: /workshop/Lym/combat3d
export RAW=$COMBAT3D_ROOT/data/harmony4d/raw     # your unpacked Harmony4D sequences

bash 00_env.sh                                   # ← always run this first
bash 01_calib.sh          016_mma4
bash 02_frames.sh         016_mma4 16
bash 03a_boxes_official.sh 016_mma4              # or 03b, not both
bash 04_pose2d.sh         016_mma4 16 <last-frame>
bash 05_assemble.sh       016_mma4 <last-frame>
bash 06_triangulate.sh    016_mma4 <last-frame>
```

## Stage 1 — label production (the zero-annotation pipeline), per take

| # | Wrapper | Underlying script | Reads | Writes |
|---|---|---|---|---|
| 00 | `00_env.sh` | *(inline check)* | `weights/ckpt/` | report |
| 01 | `01_calib.sh` | `code/adapters/harmony4d/h4d_calib.py` | `raw/<take>` | `calib_gt_<take>` |
| 02 | `02_frames.sh` | `h4d_frames.py` | `raw/<take>` | `frames/<ID>/1` |
| 03a | `03a_boxes_official.sh` | `h4d_boxes.py` | `raw/<take>` | `det_gt2_<take>` |
| 03b | `03b_boxes_self.sh` | `pipeline/det_self_final.py` | `frames/<ID>/1`, calib | `det_self2_<take>` |
| 04 | `04_pose2d.sh` | `pipeline/vp_h4d.py` | `det_*_<take>` | `vp_*_<take>` |
| 05 | `05_assemble.sh` | `pipeline/assemble_h4d.py` | `vp_*_<take>` | `asm_*_<take>` |
| 06 | `06_triangulate.sh` | `pipeline/tri_h4d.py` | `asm_*`, calib, `det_*` | `em_*_<take>` |
| 07 | `07_fit_smpl.sh` | `pipeline/fit_h4d.py` | `asm_*`, `em_*`, calib, frames | `emfit_h4d` |

* **03a vs 03b** — 03a uses the dataset's own boxes and identity; 03b builds them from
  scratch. This layer is **scene-specific**; prefer 03a whenever the dataset has them.
* **07 is optional** — the reported MPJPE/PA numbers do not use the SMPL fit.

## Stage 2 — ground truth

| # | Wrapper | Underlying script | Reads | Writes |
|---|---|---|---|---|
| 08 | `08_gt_extract.sh` | `h4d_gt.py` | `raw/<take>` | `gt_<take>` |
| 09 | `09_gt_align.sh` | `scripts/tools/ai2_fixalign.py` | `gt_*`, `em_off_*` | `gt_<take>_aligned` |

## Stage 3 — training data

| # | Wrapper | Underlying script | Arm | Output | Size |
|---|---|---|---|---|---|
| 10 | `10_npz_armB.sh` | `monocular/mb_npz.py` | **B** | `mb/data_h4d_offtri` | 1800 npz / 150 takes |
| 11 | `11_npz_armA.sh` | `scripts/tools/z31_mkab.py` | **A** | `mb/data_h4d_gtalign` | 828 npz / 69 takes |
| 12 | `12_npz_armC.sh` | `scripts/tools/ae1_mknpz.py` | **C** | `mb/data_h4d_selfnpz` | 1476 npz / 123 takes |

`11` rewrites **only** `k3d` in the arm-B npz, so A vs B-69 isolates the label source.

## Stage 4 — training

| # | Wrapper | Underlying script | Output |
|---|---|---|---|
| 13 | `13_train_main.sh` | `monocular/kb/kb_train.py` | `mb/ckpt/Combat3D_FULL.pt` |
| 14 | `14_train_roothead.sh` | `monocular/roothead_h4d_train.py` | `mb/ckpt/rh_h4d_v04_w0_full.pt` |
| 15 | `15_train_baselines.sh` | `scripts/tools/z03_ft.py` | `mb/ckpt/base_*.pt` |
| 16 | `16_ablation_qa.sh` | `scripts/tools/at5_retrain.sh` | `mb/ckpt/Combat3D_*_qa.pt` |

`13` requires `KB_FORCE_R_I=1` — it is set inside the wrapper, do not remove it.
`15` requires `FT_INIT_VP3D` (and optionally `FT_INIT_MIXSTE`) to point at the public
pre-trained checkpoints; without warm-starting the comparison is not fair.

## Stage 5 — evaluation

| # | Wrapper | Underlying script | Output |
|---|---|---|---|
| 17 | `17_eval_heldout.sh` | `scripts/tools/z25_gtEval.py` | 15-take and **9-take** held-out, vs official GT |
| 18 | `18_eval_abc.sh` | `scripts/tools/ah1_abc.py` | the paper's main table |
| 19 | `19_eval_pertake.sh` | `scripts/tools/eval_pertake.py` | per-take PA / end-to-end / R |

## Stage 6 — ablations

| # | Wrapper | Underlying script | Question |
|---|---|---|---|
| 20 | `20_ablation_tri.sh` | `scripts/tools/aw2_naive.sh` | does the triangulation layer actually help? |

## Stage 7 — visualization

| # | Wrapper | Underlying script | Output |
|---|---|---|---|
| 21 | `21_to_final13.sh` | `monocular/mb_to_final13.py` | `final13_mono_<take>` |
| 22 | `22_viz_overlay.sh` | `scripts/tools/v26_node_overlay.py` | `nodes_<take>` (mp4) |
| 23 | `23_render3d.sh` | `code/stage8_render/render3d_5000f13.py` | `render3d_mono_<take>` |

## Stage 8 — figures

| # | Wrapper | Underlying script | Output |
|---|---|---|---|
| 24 | `24_complexity.sh` | `scripts/tools/z36_complexity.py` | complexity table |
| 25 | `25_figs.sh` | `scripts/tools/bc2_figs2.py` | `figs/fig7..fig9` |

---

## `scripts/tools/` — verbatim driver scripts

These are the scripts **exactly as they ran on the server**, with only credentials
removed. The numbered wrappers call them; read them when a wrapper hides a detail.

| File | Used by | Purpose |
|---|---|---|
| `run.sh` / `pull.sh` | all | push-and-run / pull over ssh + `docker exec` (credentials from env) |
| `z25_gtEval.py` | 17 | held-out evaluation vs official GT |
| `ah1_abc.py` | 18 | all arms, one metric, one alignment |
| `eval_pertake.py` | 19 | per-take breakdown |
| `z31_mkab.py` | 11 | arm A npz (official GT labels) |
| `ae1_mknpz.py` | 12 | arm C npz (fully self-built) |
| `ai2_fixalign.py` | 09 | **inverse** Umeyama world-frame alignment |
| `z03_ft.py` | 15 | baseline fine-tuning, with `FT_INIT` warm start |
| `at5_retrain.sh` | 16 | frame-filtering ablation |
| `aw2_naive.sh` | 20 | naive triangulation comparison |
| `z36_complexity.py` | 24 | scene complexity table |
| `bc2_figs2.py` | 25 | fig7 / fig8 / fig9 |
| `mb_npz2.py` | *(alt)* | npz builder, alternate variant |
| `det_self_xview.py` | *(expl.)* | cross-view detection variant |
| `viz_mono.py`, `viz_mono3d.py` | *(viz)* | monocular overlays |
| `zi_shape2.py` | *(exp.)* | zero-shot vs fine-tuned shape comparison |
| `v26_node_overlay.py` | 22 | node overlay onto the source video |
| `k9_dl_rest.py` | — | downloads the remaining Harmony4D sequences |

## Working history

The ad-hoc scripts from the working session are not part of this release. Everything needed
to reproduce the paper is in `code/`, `monocular/` and the numbered wrappers above. If you need the raw session history, contact the authors.

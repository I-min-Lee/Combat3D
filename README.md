# Combat3D

### Monocular 3D human motion for mocap-free close-combat scenes: zero-annotation label production, domain adaptation, and capability boundaries

### —— with two quantities systematically hidden by evaluation protocols (absolute localisation / temporal resolution)

**中文说明见 [README_CN.md](README_CN.md)。**

---

## What this is

**Some domains cannot capture 3D ground truth at all.** In close combat (wrestling / jiu-jitsu /
MMA / armoured stick-fighting) markers are occluded by one's own body and by the opponent, and get
knocked off — **no in-domain dataset with mocap truth exists**. We surveyed the public datasets
layer by layer and **none satisfies all five requirements** (multi-view / intrinsics+extrinsics /
ground truth / two-person close contact / heavy occlusion).

This repository takes the **existing** route — *multi-view pseudo-labels → monocular lifting* — and
**adapts it to this domain**, then measures what the adaptation costs and where it breaks.

**We claim no architectural or modular novelty.** The mechanisms we use are prior work and are cited
as such (`RootNet` for root-position regression, `DeciWatch` (ECCV 2022) for sparse→dense recovery,
`MBTI` (ICCV 2025) for the frame-rate axis and rate conditioning, `MAMMA` (CVPR 2026) for markerless
multi-view pipelines as ground truth). **What is non-trivial is the adaptation** — every item below
is an answer to "why does the existing method not work here?":

| Obstacle specific to this domain | Our adaptation | Effect |
|---|---|---|
| No official 2D; **fisheye**, close-range multi-camera | OPENCV_FISHEYE triangulation + per-joint **view-subset enumeration** + **box-contact re-weighting** | misusing the pinhole model degrades reprojection **by an order of magnitude** |
| Contact makes the two boxes overlap heavily → pure geometry **always swaps them** | **position prior** + jersey-colour calibration | person selection **72% → 95.7%**; end-to-end **10×** |
| ★ Multi-rig ⇒ the target contains a **different fixed rotation `R` per (take, view)** | ★ the model **cannot know which camera it belongs to** → shape destroyed; **set `R = I`**, target = pure camera-frame pose | wrong target → PA **38.1 → 50.0 mm** |
| **Root-relative metrics discard absolute localisation**, yet tactical analysis / replay / inter-person distance **all need it** | **root-regression head** | geometric prior **996 mm → 9 mm** |
| Deployment frame rate is uncontrolled, and this domain has **no ground truth to supervise it** | **rate conditioning** (one extra input channel) | curve steepness **+89% → +14%** |

## Results

| Quantity | Value |
|---|---|
| Pose **shape** (PA-MPJPE), in-domain | **32.6 mm** |
| **held-out** vs official GT (9 unseen scenes) | **42.4 / 32.6 mm** |
| Same protocol, fine-tuned **VideoPose3D** | **188.3 / 138.7 mm** → **4.4×** |
| **Zero-shot** on independent true mocap — CMU Panoptic | **120.9 / 74.0 mm** |
| **Zero-shot** on independent true mocap — MPI-INF-3DHP (6 sequences, 2840 frames) | **133.5 / 86.5 mm** |
| Inter-person hip distance error | **< 1%** |
| Zero-annotation cost at the **label** layer | **+13%** |

**Two independent true-mocap datasets agree in magnitude** (PA 74.0 vs 86.5), which rules out
"you happened to pick a matching dataset". **We do not claim that monocular matches multi-view** —
`MAMMA` (CVPR 2026) notes the inherent difficulty of monocular inference under close interaction.

## The frame-rate axis (one of the adaptations, not the headline)

**The problem was already posed and formalised by `MBTI` (ICCV 2025) — EMDB-FPS benchmark, MCF
metric. We claim no novelty for it.** We measure it in a **mocap-free** domain.

On **52 unseen scenes of the public Harmony4D dataset**, with **the same model, the same content and
the same ground truth**, changing only the input sampling rate gives:

| test frame rate | 20 (train) | 10 | 5 | 4 |
|---|---|---|---|---|
| end-to-end MPJPE | **21.7 mm** | 29.1 (+33.8%) | 41.0 (+88.6%) | 44.9 (+106.5%) |

Monotonic and dose–response; unseen-scene readings match the all-scene ones (ruling out
memorisation). **Both directions hold on public data** — training on the 5 fps copy only and testing
back at 10 / 20 fps degrades by **+30.3% / +44.9%**.

**Three countermeasures, compared on the same protocol:**

| test fps | base | ① frame-rate aug. | ② time-norm. window | **★③ Δt conditioning** |
|---|---|---|---|---|
| 20 | **21.7** | 41.9 | 21.7 | **21.9** |
| 10 | 29.1 | 40.6 | 30.2 | **22.7** |
| 5 | 41.0 | 34.7 | 41.7 | **24.9** |
| steepness | **+89%** | +0% (origin +93%) | worse | **+14%** |

* **★③ Δt conditioning is the solution** — one extra input channel `log2(fps/f0)` plus a widened
  `joints_embed` `(512,3) → (512,4)`: **the native rate does not degrade**, and the 10 / 5 fps errors
  drop by **22% / 39%**.
* **① frame-rate augmentation** flattens the curve but **sacrifices the origin (+93%)** — it is a
  countermeasure for an unknown/varied deployment rate, not a universal improvement.
* **② the inference-time time-normalised window gives no gain** on either dataset —
  the countermeasure lives in **training and architecture, not in inference**.

> ⚠️ **Withdrawn readings.** An earlier revision reported a "200 fps trained" long-weapon arm and an
> "upward" check of **+24.1% / +63.8%**. Auditing the data showed that dataset's two resolution
> groups have **different native frame rates** (960×720 → 200 fps, 1920×1440 → 25 fps) and that the
> arm was built from the 25 fps group while the meta recorded the *target* value. **Those readings
> are withdrawn**; the public-data two-directional result above is unaffected.

A cross-dataset picture (Harmony4D / long-weapon / Panoptic), a mechanism explained by an
**approximately constant absolute penalty (~+19 mm at 5 fps)**, and **a hypothesis we refuted
ourselves** are in [`monocular/temporal/fps_axis/`](monocular/temporal/fps_axis/README.md).

---

## The carrier domain (where the measurements come from)

Two athletes in continuous body contact — wrestling / jiu-jitsu / MMA / armored stick-fighting — is
a setting where **motion capture physically cannot produce ground truth**: reflective markers are
occluded by the opponent and by the subject's own body, and body contact knocks them off. There is no
cost argument here; it is a physical constraint.

The repository therefore also contains a *zero-manual-annotation* multi-view label-production
framework and the monocular lifting recipe built on it, evaluated on **Harmony4D** and
cross-validated on **CMU Panoptic** (independent optical mocap).

> **Scope note.** The label pipeline itself is assembled from existing components (multi-view
> triangulation, SMPL fitting, an off-the-shelf monocular lifter) and we claim **no architectural
> novelty** for it. It contributes a *layer-wise error budget*, seven ablation-backed rules, and the
> code + weights + exact commands to reproduce every number. **The new claim is the frame-rate axis.**

---

## Demo

Short segments from the **armored stick-fighting** domain — our own dataset, and the
second domain in which the layer contract was validated.

<p align="center">
  <img src="media/demo_raw.gif" width="410" alt="Raw footage from one camera view">
  <img src="media/demo_smpl.gif" width="410" alt="SMPL fitted to the triangulated labels">
</p>

**Left — raw footage** from a single camera view.
**Right — SMPL fitted to the triangulated labels**, overlaid on the source view.

Still frames reconstructed in world space (pose nodes only; the weapon is deliberately not
drawn, as it is outside what this pipeline reconstructs):

<p align="center">
  <img src="media/render_grid_1.1.png" width="410" alt="Rendered 13-joint skeleton, take 1.1">
  <img src="media/render_grid_0.1.png" width="410" alt="Rendered 13-joint skeleton, take 0.1">
</p>

Full-length clips: [`media/demo_raw.mp4`](media/demo_raw.mp4) · [`media/demo_smpl.mp4`](media/demo_smpl.mp4)

> **Note on the footage.** These are real people from a private dataset. See the terms in
> `Combat3D-kendo-0.1-sample/README.md` §5 — research reproduction only; do not
> redistribute; do not attempt to identify the individuals.

---

## Results at a glance

All numbers are **MPJPE / PA-MPJPE in mm**, root-relative, evaluated on **9 genuinely
held-out Harmony4D takes against the official Harmony4D GT** (six `sword3` takes that
overlap the training source sequences are removed as leakage).

| Arm | Upstream (box / identity / 2D) | 3D labels | Train takes | MPJPE / PA |
|---|---|---|---|---|
| **A** | official | official GT | 69 | 58.1 / 26.6 |
| **B-69** | official | ours (triangulated) | 69 | 65.6 / 38.2 |
| **B** | official | ours (triangulated) | **150** | **42.4 / 32.6** |
| **C** | fully self-built | ours (triangulated) | 123 | 116.9 / 91.6 *(exploratory, see note)* |
| Official MotionBERT, zero-shot | — | — | — | 94.3 / 66.8 |

Decomposition this yields:

1. **Label layer cost = +7.5 mm (+13%)** — A → B-69, same 69 takes, same 828 npz.
2. **More data beats cleaner labels** — the full 150-take arm (42.4) overtakes the
   official-GT arm trained on 69 takes (58.1) by 27%.
3. **Self-built upstream cost = +74.5 mm (2.8×)** — but see the caveat below; this is
   not a fair comparison and arm C is **not** a primary result.

**Caveat that decides the framing of arm C.** The official Harmony4D `poses2d` are
*SMPL-fit projections*, not detections: their per-joint confidence is identically
`1.000` with 17/17 usable joints, whereas our ViTPose output has confidence 0.58–0.89
and 15.5/17 usable joints. Any "self-built 2D vs official 2D" comparison is therefore
unfair by construction — the official 2D "already knows what the answer should look
like". Arm C is retained as an **exploratory upper bound on the cost**, not as a
like-for-like baseline.

### Absolute positioning (a layer the literature discards by definition)

Shape metrics are root-relative and therefore *cannot* see absolute placement. We treat
absolute positioning as an orthogonal layer and attach a root-regression head:

| Absolute root-position error (median) | |
|---|---|
| Geometric prior (depth from a fixed bone-length assumption) | 996 mm |
| **Combat3D root-regression head** | **9 mm** |

The head bolts onto *any* root-relative lifter and is what turns a floating skeleton
into a usable world-space trajectory (inter-athlete distance, tactical analysis,
re-projection).

### Cross-domain transfer to independent optical mocap

CMU Panoptic, PA-MPJPE, same architecture, zero-shot vs fine-tuned:

| | Harmony4D (in-domain) | CMU Panoptic (cross-domain) |
|---|---|---|
| Official MotionBERT, zero-shot | PA 66.8 | PA 101.2 |
| **Combat3D (fine-tuned)** | **PA 32.6** | **PA 74.0 (−27%)** |

### Same-protocol baseline comparison

Every baseline is fine-tuned **from its public pre-trained weights**, on the same H4D
training set with the same 2D input and the same `(take, view)` alignment procedure.

| Method | Zero-shot (15 test takes) | Fine-tuned from public weights (val) | **Fine-tuned, 9 held-out** (same protocol as the main table) |
|---|---|---|---|
| VideoPose3D (detectron weights) | 276.6 / 174.2 | **106.1** | **188.3 / 138.7** |
| VideoPose3D (cpn weights) | — | 115.9 | **186.9 / 137.4** |
| MixSTE | 154.8 / 112.2 | *(ONNX only, no PyTorch weights — omitted)* | — |
| **Combat3D-Mono (ours)** | **15.8 / 10.4** | **18.4** | **42.4 / 32.6** |

**Comparable reading (identical held-out protocol): VideoPose3D 188.3 vs ours 42.4 — 4.4×.**
(The val-only reading is 106.1 vs 18.4 = 5.8×; val is the model-selection set, so quote the
held-out column.) Every "ours" number in this repository uses the **same** checkpoint,
`Combat3D_FULL.pt`.

> A from-scratch training variant was run and then **discarded**: it measures data
> efficiency, not architecture, and comparing a heavily pre-trained model against a
> from-scratch baseline is not a fair protocol.

---

## Seven ablation-backed findings

Each is verifiable from the artifacts in this repository, and (to our knowledge) each was
previously unreported.

| # | Finding | Measured | Why it is counter-intuitive |
|---|---|---|---|
| **F1** | Zero-annotation cost **at the label layer** is small | **+7.5 mm (+13%)**, controlled: 69 takes / 828 npz on both sides | The default assumption is that without annotation you cannot learn |
| **F2** | The marginal value of **data volume** exceeds that of **label precision** | 150-take arm at **42.4** beats the official-label arm at 69 takes by **27%** | It not only recovers the 13% gap, it overtakes it |
| **F3** | **Frame-level quality filtering is a net loss** at this data scale | All three arms get worse: A 56.7→72.2, B **18.4→56.4**, C 151.7→255.5 | "Clean your data" is treated as universally good |
| **F4** | The triangulation layer's robustification **only helps when upstream is dirty** | clean upstream: **identical** to the naive baseline (< 1e-6); dirty upstream: median **36.7→20.5 (−44%)** | It is not "always better" — it is a *noise-suppression device* |
| **F5** | A wrong per-`(take, view)` **scale convention** inflates end-to-end error **6×**, silently | **190.3 → 27.5 mm** after the fix; "residual rotation" 30° → 3.7° | It raises no exception and produces plausible-looking output |
| **F6** | Cross-domain gain **transfers to independent optical mocap** | same architecture: PA **101.2 → 74.0 (−27%)** | Rules out the gain being self-justification against our own labels |
| **F7** | The root-regression head turns absolute placement from **110×** worse into usable | **996 → 9 mm** | Almost every monocular paper discards this layer by definition of its metric |

**F1–F3 answer "what does it cost"; F4–F5 answer "where does the error live"; F6–F7
answer "is it real and is it useful".**

### Design principle that follows

Splitting label production into five layers, our measurements support one operational rule:

> **The detection-box layer and the identity layer are *scene-specific* — adapt them per
> scene (and if the scene already gives you boxes and identities, use those instead).
> The triangulation layer and everything below it are *universal*.**

Supporting evidence: a **single** change at the identity layer — rejecting candidate
boxes whose centre falls outside the frame border — moves person-selection accuracy from
**72.0% to 95.7%**, and end-to-end error from **383.2 mm to 39.1 mm** (a 10× gap). That
is direct evidence these layers are determined by scene conditions, and that designing
them as pluggable is the right call. On our own armored stick-fighting data the same layer needed
jersey-colour cues plus a global consistency curve — further evidence that the layer is
scene-specific by nature.

---

## Repository layout

The layout follows the system's layers — **label production** (where the scene-specific layers
live) → **monocular inference** → **evaluation and diagnostics**.

```
Combat3D/
├── code/
│   ├── label_pipeline/            # ① LABEL PRODUCTION (multi-view → 3D labels)
│   │   ├── adapters/harmony4d/    #   · dataset adapter: calib / frames / boxes / gt
│   │   │   └── pipeline/          #       detect → 2D → assemble → triangulate → fit
│   │   ├── adapters/panoptic/     #   · second-dataset adapter (cross-domain transfer)
│   │   ├── stages/                #   · the layer-wise stages, in execution order
│   │   │   ├── 1_detect/  2_pose2d/  3_identity/  4_assemble/
│   │   │   └── 5_triangulate/  6_fit/  7_temporal/  8_render/
│   │   └── env/sitecustomize.py   #   · py3.12 / numpy2 compatibility shim (required)
│   ├── eval/                      #   metrics, audits, stratification
│   └── viz/                       #   re-projection overlays, 3D renders
├── monocular/                     # ② MONOCULAR INFERENCE
│   ├── kb/                        #   · training toolkit (kb_train.py, kb_common.py, builders, QA)
│   ├── lifter/                    #   · Harmony4D artifacts → npz → fine-tuned lifter → 13-joint output
│   ├── localization/              #   · absolute root-position regression head
│   └── temporal/                  #   ★ frame-rate axis: effect, three countermeasures, pitfalls
│       └── fps_axis/              #       fps_axis.py — build / train (Δt conditioning) / eval ladder
├── scripts/                       # ③ DRIVERS
│   ├── README.md                  #   ★ call order table
│   ├── 00_env.sh … 23_render3d.sh #   stage wrappers, numbered in execution order
│   └── tools/                     #   verbatim driver scripts as run on the server
├── docs/                          # ④ DOCUMENTATION
│   ├── REPRODUCE.md               #   ★ full reproduction guide, in call order
│   ├── PITFALLS.md                #   ★ silent-failure list, ranked by cost
│   ├── METRICS.md                 #   ★ which numbers are valid and which are retracted
│   ├── ENV_pose312.md             #   exact environment recipe + install traps
│   ├── SECOND_DOMAIN.md           #   ★ the armored stick-fighting second-domain evidence
│   └── H4D_adapter_README.md
├── configs/h4d_metric_scale.json
├── weights/README.md              # where to put the downloaded checkpoints (+ md5 manifest)
├── data/DATASETS.md               # every npz dataset produced, with counts
├── figs/                          # fig1…fig10 (framework, pipeline, error budget, complexity, …)
├── media/                         # demo clips and rendered stills
│   ├── demo_raw.mp4 / .gif        #   input footage, as captured
│   ├── demo_smpl.mp4 / .gif       #   reconstructed motion
│   └── render_grid_*.png          #   multi-view render grid
└── (working session history is not part of this release)
```
```

## Quick start

```bash
git clone https://github.com/I-min-Lee/Combat3D.git && cd Combat3D

# 1. weights (1.2 GB) are NOT in the repo — download from GitHub Releases, then:
bash scripts/00_env.sh                     # checks env + places ckpt/ where code expects
python scripts/tools/z25_gtEval.py ckpt/Combat3D_FULL.pt    # reproduce the main number

# 2. full pipeline from Harmony4D raw data:
#    see docs/REPRODUCE.md — scripts/01…25 run in numbered order

# 3. second domain (armored stick-fighting): grab Combat3D-armored stick-fighting-1.1-sample.zip from Releases
#    — 1000 frames of real footage with 3D labels, 2D, calibration, weights and
#      the scene-specific detection + identity scripts. See docs/SECOND_DOMAIN.md
```

Environment: Python 3.12, **torch 2.4.1+cu121**, mmcv 2.2.0 (official prebuilt) — see
[`docs/ENV_pose312.md`](docs/ENV_pose312.md). Two hard rules:

* **Do not install or upgrade packages.** A single `pip install local-attention` pulled
  in `nvidia-cudnn-cu13` and silently broke every convolution layer. Recovery:
  `pip install --force-reinstall --no-deps nvidia-cudnn-cu12==9.1.0.70`
* **Do not modify triangulation for inference.** Triangulation is a *training-label /
  offline-ground-truth* device only. See [`docs/METRICS.md`](docs/METRICS.md).

## Checkpoints

See [`weights/README.md`](weights/README.md) for the full table and MD5s. Primary:

| File | Role |
|---|---|
| `Combat3D_FULL.pt` | **arm B — the main result** |
| `Combat3D_GTlabel2.pt` | arm A (official GT labels) |
| `Combat3D_OurLabel69.pt` | arm B-69 (controlled comparison) |
| `Combat3D_SelfLabel_v2.pt` | arm C (exploratory) |
| `rh_h4d_v04_w0_full.pt` | root-regression head (9 mm) |
| ★ `h4d_dt.pt` | **Δt conditioning (4-channel input) — the frame-rate solution** |
| ★ `h4d_aug.pt` | frame-rate augmentation (3-channel; flattens but raises the origin) |
| `base_vp3d_ft120.pt`, `base_mixste_ft.pt` | fine-tuned baselines |

`h4d_dt.pt` loads into the standard `DSTformer` **with `dim_in=4`** and `joints_embed`
widened from `(512,3)` to `(512,4)` — see `monocular/temporal/fps_axis/fps_axis.py:build_dt_model`.
The 4th input channel is a constant per clip: `log2(fps / 20)`.

## What this work does *not* claim

* **No new architecture.** The pipeline is assembled from existing components.
* **"Training without mocap" is not a novel idea.** Multi-view pseudo-labels for
  monocular lifting is an established route (≥ 2020, e.g. Iqbal & Kautz CVPR 2020,
  MetaPose CVPR 2022, and arXiv:2504.12699 which we cite and compare against).
* **Not a CCF-A system paper.** This is a measurement + protocol contribution.
* **Weights are derivatives.** Fine-tuned from MotionBERT / VideoPose3D / MixSTE
  checkpoints; the upstream licences still apply.


**Frame-rate specific disclaimers (2026-10-09).** We do **not** claim:

- the multi-view pseudo-label route — **Suzuki et al., CVPRW CVSports 2024** do
  multi-view pseudo-labels → monocular fine-tuning in a sports domain, a construction
  identical to ours;
- camera-agnostic lifting — **RUMPL** (ray representation, deliberately camera-invariant),
  **Ray3D**, **EPOCH**, **CameraPose**;
- monocular multi-person identity — **PHALP**, **DETRAM**, **RAM**, **TesseTrack**,
  **Zanfir et al.**, **Opti-Pose3D**, and CVPR 2025 close-interaction reconstruction;
- the discovery that 2D metrics do not predict 3D quality — **2TRAX3** (MDPI Sensors,
  kickboxing) already states it;
- absolute/global root localization — **RootNet**, **PoseAnchor** (ICCV 2025).

**What we do claim on this axis**: that *no prior work has examined cross-frame-rate
transfer*, that the degradation is monotonic and large on public data, that it is an
approximately constant absolute penalty rather than a motion-speed effect (a hypothesis we
refuted ourselves), and that **Δt conditioning** removes it without degrading the native rate.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE). Upstream model weights retain their own licences.

---

# 中文说明

**Combat3D** 面向**高遮挡、强对抗的贴身格斗场景**（摔跤 / 柔术 / MMA / 盔甲对抗棍棒格斗）的单目 3D
人体姿态估计。这类场景是**动捕物理上无法覆盖**的——贴身时反光 marker 被对手和自身身体
遮挡，身体接触还会把 marker 碰掉。这不是成本问题，是物理约束。

本仓库是论文的**全部可复现产物**：代码、权重、配置、以及能逐条重跑出论文每个数字的命令。

**这不是"新网络"型贡献，是"测量 + 协议"型贡献。** 管线由现成组件组装，我们不主张架构新颖性。
我们提供的是：零人工标注这条路线的一份**逐层误差预算**、七条**有消融支撑的结论**、
以及一条**经过测量支持的设计原则**（检测框层/身份层场景相关、三角化层及以下通用）。

**主结果**（9 条真 held-out，对官方 Harmony4D GT，MPJPE / PA，mm）：

| 臂 | 上游 | 3D 标签 | 训练量 | MPJPE / PA |
|---|---|---|---|---|
| A | 官方 | 官方 GT | 69 take | 58.1 / 26.6 |
| B-69 | 官方 | 自建三角化 | 69 take | 65.6 / 38.2 |
| **B** | 官方 | 自建三角化 | **150 take** | **42.4 / 32.6** |
| C | 全自建 | 自建三角化 | 123 take | 116.9 / 91.6（**探索性，非主结果**）|
| 官方 MotionBERT 零样本 | — | — | — | 94.3 / 66.8 |

**七条结论**：① 零标注在标签层的代价仅 +13%（受控对照）；② 数据量的边际收益 > 标签精度
（150 take 反超 27%）；③ 帧级质量筛选在小数据规模下是净损失（三臂全部变差）；④ 三角化层的
鲁棒化只在上游脏时起作用（干净上游与朴素做法完全相同）；⑤ 逐 (take,view) 尺度约定写错会让
端到端虚高 6 倍且完全静默；⑥ 跨域增益可迁移到独立真动捕（同架构 −24%）；⑦ 根回归头把绝对
定位从 996 mm 压到 9 mm（110×）。

**先从 [`docs/REPRODUCE.md`](docs/REPRODUCE.md) 读起**——它是按调用顺序写的完整复现指南。
写论文或引用数字前**必读 [`docs/METRICS.md`](docs/METRICS.md)**（里面列了已作废的数字口径）。

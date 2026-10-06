# Checkpoints

The checkpoints are **not** stored in this repository — a single `.pt` is ~170 MB, above
GitHub's 100 MB per-file limit. Download the release archive and unpack it:

```bash
# from the Releases page: Combat3D-weights-20261005.zip
unzip Combat3D-weights-20261005.zip
export COMBAT3D_ROOT=/path/to/your/artifacts
mkdir -p $COMBAT3D_ROOT/mb/ckpt
cp Combat3D-weights-20261005/ckpt/* $COMBAT3D_ROOT/mb/ckpt/
```

---

## Checkpoints shipped in the release

| File | MD5 (first 12) | Role | Val |
|---|---|---|---|
| `Combat3D_FULL.pt` | `78fa1d25855c` | **Arm B — the main result** (our triangulated labels, 150 takes) | 18.4 mm |
| `Combat3D_GTlabel2.pt` | `1c3eb465406c` | Arm A — official GT labels, 69 takes | — |
| `Combat3D_OurLabel69.pt` | `b027f1fb323b` | Arm B-69 — the controlled comparison against A | — |
| `Combat3D_SelfLabel_v2.pt` | `79d7edc969f3` | Arm C — fully self-built *(exploratory)* | — |
| `Combat3D_SelfLabel.pt` | `1132d4908d09` | Arm C, first version — **needed by `ah1_abc.py`** | — |
| `kb_h4d_ri_lr2e4.pt` | `9dbdfa825115` | `KB_FORCE_R_I=1` ablation | — |
| `H4D_BEST_79.2mm.pt` | `98db4e058063` | Single-view arm; also the frozen lifter used by the armored stick-fighting branch | 78.1 mm |
| `kb_full_v3.pt` | `0962ad8131c0` | Armored stick-fighting-domain lifter — zero-shot reference **and** the frozen model of the second-domain run | 52 mm |
| `base_vp3d_ftpub.pt` | `e1fb209b7384` | **VideoPose3D, fine-tuned from the public detectron weights → 106.1 mm** | 106.1 mm |
| `base_vp3d_ftpub_cpn.pt` | `58751e597beb` | **VideoPose3D, fine-tuned from the public cpn weights → 115.9 mm** | 115.9 mm |
| `rh_h4d_v04_w0_full.pt` | `4d9bdb99240e` | Root-regression head (absolute positioning, 9 mm) | — |

### Which VideoPose3D checkpoint is the paper's?

**`base_vp3d_ftpub.pt` and `base_vp3d_ftpub_cpn.pt`.** Both were trained with `z03_ft.py`
**warm-started from the public pre-trained weights** (`FT_INIT`), which is the protocol the
paper's table reports (106.1 / 115.9 mm).

Two other VideoPose3D checkpoints exist on the training machine and are **deliberately not
shipped**:

| File | What it is | Why it is excluded |
|---|---|---|
| `base_vp3d_ft120.pt` | trained **from scratch**, 120 epochs → 120.3 mm | from-scratch training measures data efficiency, not architecture. Comparing a pre-trained model against it is not a fair protocol; the paper explicitly discards it. |
| `base_vp3d_ft.pt` | trained from scratch, 40 epochs → 149.1 mm | same reason |
| `base_mixste_ft.pt` | trained from scratch, 60 epochs → **338.5 mm** | same reason, and the result is under-trained rather than representative. MixSTE is omitted from the table entirely: only ONNX weights are public, no PyTorch checkpoint, so the same-protocol warm start is impossible. |

You can tell the two families apart by their initial loss: a warm-started run starts near
`loss ≈ 0.097`, a from-scratch run starts near `loss ≈ 0.72`. If you retrain and see the
second number, `FT_INIT` did not take effect.

### Calibration sidecars

Every `.pt` ships with a matching `_calib.json` holding the per-`(take, view)` `(s, R)`
fitted at evaluation time. **They must travel together** — evaluating a checkpoint under a
different calibration file changes the reported number. The archive also contains the full
set of ablation calibration JSONs (`kb_h4d_*_calib.json`).

## Starting weights you must obtain yourself

`z03_ft.py` warm-starts from the upstream public checkpoints. These are third-party assets
and are **not redistributed here**:

| Needed for | Asset | Where from |
|---|---|---|
| `z03_ft.py --arch vp3d` (detectron) | `vp3d_h36m_detectron.bin` | facebookresearch/VideoPose3D |
| `z03_ft.py --arch vp3d` (cpn) | `vp3d_h36m_cpn.bin` | facebookresearch/VideoPose3D |
| reference / comparison | `mb_h36m_ft.bin` | Walter0807/MotionBERT |
| MixSTE (ONNX only) | `mixste_3d_243.onnx` | MixSTE upstream |

Pass them via `FT_INIT_VP3D` / `FT_INIT_MIXSTE` (see `scripts/15_train_baselines.sh`).

## Verifying a download

```bash
cd weights/ckpt && md5sum *.pt
```

Compare against `MANIFEST_release_v2.md5` (in this directory), which holds the MD5 of every
file in the original release bundle.

## A note on what these weights are

All eleven are **genuinely fine-tuned**, not renamed copies of an upstream checkpoint:
against the official MotionBERT `best_epoch.bin`, every tensor differs
(mean |Δ| between 2e-3 and 1e-2, **0% of tensors byte-identical**). The verification script
is a few lines long and fully reproducible — compare
`state_dict` keys after stripping the `module.` prefix, which is the only structural
difference between the upstream and our checkpoints.

## Licensing of derived weights

These are fine-tunes of third-party models. The upstream licences still apply:

| Base | Upstream licence |
|---|---|
| MotionBERT | MIT (https://github.com/Walter0807/MotionBERT) |
| VideoPose3D | MIT (https://github.com/facebookresearch/VideoPose3D) |
| MixSTE | see upstream repository |
| ViTPose | Apache-2.0 (https://github.com/ViTAE-Transformer/ViTPose) |

This repository's MIT licence covers **our** code and documentation, not the upstream
model weights.

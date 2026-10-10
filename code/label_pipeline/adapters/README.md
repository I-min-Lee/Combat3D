# Adapters: which layers are scene-specific and which are universal

**Read this before deleting anything.** This directory is the *evidence* for the
design principle, not just a convenience wrapper around the data.

The pipeline is split into five layers. Our measurements say exactly two of them are
determined by the scene:

```
  ┌──────────────────────────────────────────────────────────┐
  │  SCENE-SPECIFIC  —  re-implement per domain              │
  │    detection-box layer      adapters/<dataset>/h4d_boxes.py, pipeline/det_*.py
  │    identity layer           (same files; different evidence per scene)
  └──────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────┐
  │  UNIVERSAL  —  reuse unchanged across domains            │
  │    2D pose extraction       pipeline/vp_h4d.py (ViTPose)
  │    triangulation            pipeline/tri_h4d.py
  │    SMPL fit                 pipeline/fit_h4d.py
  │    monocular lifter         ../monocular/
  └──────────────────────────────────────────────────────────┘
```

## Why this matters — the ablation behind the boundary

**A single change at the identity layer** — rejecting candidate boxes whose centre falls
outside the frame border — moves person-selection accuracy from **72.0% → 95.7%** and
end-to-end error from **383.2 mm → 39.1 mm**. A 10× effect from one scene-level rule is
the evidence that this layer is *not* generic.

Conversely, the triangulation layer's robustification (subset enumeration over views)
produces a **bit-identical** result to the naive baseline when upstream is clean
(difference < 1e-6), and a **44%** median-error reduction when upstream is dirty. It is a
noise-suppression device, not a scene adaptation — which is why it belongs to the
universal half.

## So: should the scene-specific layers be shipped at all?

**Yes — keep them, and label them.** Hiding them would be worse, for three reasons:

1. **They are the evidence for the design principle.** The claim "these two layers are
   scene-specific" is only defensible if you can point at a scene-specific implementation
   and the controlled ablation that shows the impact. Delete the files and the claim loses
   its support — you would be removing your own contribution.
2. **Arm C is not reproducible without them.** The fully-self-built arm explicitly needs
   `pipeline/det_self_final.py` together with the position prior.
3. **A repository with holes is more misleading than one with clearly-scoped parts.** A
   reader who finds `det_self_final.py` labelled *"Harmony4D-specific implementation of a
   scene-adaptive layer — expect to rewrite this for your scene"* learns the right lesson.
   A reader who finds no detection code at all learns nothing and may assume the layer is
   generic.

What would be misleading is shipping them **without** that label. That is why:

* every scene-specific file carries a header comment saying so;
* [`../../docs/SECOND_DOMAIN.md`](../../docs/SECOND_DOMAIN.md) shows the *same* two layers
  having to be **re-implemented differently** for the armored stick-fighting domain, while the universal
  layers ran unchanged;
* the README's design-principle section states the boundary explicitly.

## Layout

```
adapters/
├── harmony4d/          Harmony4D adapter — the domain used for the main results
│   ├── h4d_calib.py        ★ scene-specific: reads this dataset's COLMAP SfM
│   ├── h4d_frames.py       ★ scene-specific: this dataset's frame indexing
│   ├── h4d_boxes.py        ★ scene-specific: uses the dataset's own boxes+identity
│   ├── h4d_gt.py           ground truth extraction (evaluation only)
│   └── pipeline/
│       ├── det_self_final.py   ★ scene-specific: self-built detector + position prior
│       ├── det_ours_v4/v5.py   ★ scene-specific: earlier association attempts (lost)
│       ├── id_vote.py, pid_stabilize.py   ★ scene-specific identity stabilisation
│       ├── vp_h4d.py           universal: ViTPose over the given boxes
│       ├── assemble_h4d.py     universal: package 2D into the triangulation input
│       ├── tri_h4d.py          universal: robust triangulation
│       ├── tri_ablate.py       universal: triangulation with the robustness switch
│       └── fit_h4d.py          universal: SMPL fit
└── panoptic/           CMU Panoptic adapter — the cross-domain check
    ├── p2_calib.py         scene-specific: pinhole + 5-parameter distortion, units = cm
    ├── p2_gt.py            ground truth (independent optical mocap)
    ├── p2_boxes.py         scene-specific: Panoptic's body boxes
    ├── p2_frames.py        scene-specific
    └── run_p2.sh           the whole Panoptic run
```

Note what the Panoptic column demonstrates: the adapter had to change
(**fisheye → pinhole+distortion**, **metres → centimetres**, a different box source),
while `tri_h4d.py` and the monocular lifter were used as-is. Two datasets, two
scene-adaptive layers, one unchanged back-end.

## Adding your own domain

1. Copy `adapters/panoptic/` as a template — it is the smaller and cleaner of the two.
2. Implement calibration, frames, and boxes/identity for your scene. Expect this to be the
   entire cost of the port.
3. Point the universal scripts at your artifacts via their environment variables
   (`AB_CAL`, `AB_VIEWS`, `AB_ANNOTS`, `AB_DET_DIR`, `AB_W_IMG`, `AB_H_IMG`).
4. If you already have boxes and identities for your scene, use them: that is the
   cheapest and best-configured option, and it is what we do for the Harmony4D arms
   A and B (`h4d_boxes.py`).

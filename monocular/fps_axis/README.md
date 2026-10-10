# The frame-rate axis of monocular 3D lifting

> **Temporal receptive fields are specified in *frames* but mean *seconds*.**

Every published temporal monocular 3D lifter fixes its temporal window in **frames**:

| Method | How the temporal window is stated |
|---|---|
| VideoPose3D / dilated TCN | receptive field of **243 frames** |
| LAMP-Net | **"4 seconds"**, parenthetically converted to **120 frames** at 30 Hz |
| MotionBERT | `maxlen = 243` **frames** |

The frames-to-seconds conversion **is the frame rate**, and the frame rate is not controlled at
deployment: broadcast feeds run at 25 / 30 / 50 / 60 fps, phones reach 240 fps. Within the scope of
our search, **no work has examined cross-frame-rate transfer** — there is neither a formalisation of
frame-rate invariance nor a controlled "train at rate A, test at rate B" experiment.

---

## 1. The effect (Harmony4D, 52 unseen scenes)

The **same model, the same content, the same ground truth** — only the input sampling rate changes
(down-sampled along time, model window unchanged):

| Test frame rate | end-to-end MPJPE | relative |
|---|---|---|
| **20** (training rate) | **21.7 mm** | — |
| 10 | 29.1 mm | **+33.8%** |
| 5 | 41.0 mm | **+88.6%** |
| 4 | 44.9 mm | **+106.5%** |

Monotonic and dose–response. On **unseen scenes** the reading matches the full-data one
(+65.4% / +116.5%), which rules out memorisation.

We complete the **upward** direction with a self-collected **200 fps** long-weapon capture: a model
trained at 200 fps degrades **+24.1%** at 25 fps and **+63.8%** at 5 fps.

---

## 2. Three countermeasures

Training set = the same scenes at stride 1 / 2 / 4 (20 / 10 / 5 fps) merged.
Validation set = **only the 20 fps copy** (so that the protocol stays comparable).

| Test fps | base (fixed 20) | ① frame-rate aug. | ② time-norm. window | **★③ Δt conditioning** |
|---|---|---|---|---|
| **20** | **21.7** | 41.9 | 21.7 | **21.9** |
| 10 | 29.1 | 40.6 | 30.2 | **22.7** |
| 5 | 41.0 | **34.7** | 41.7 | **24.9** |
| curve steepness | **+89%** | +0% (origin +93%) | slightly worse | **+14%** |

* **★③ Δt conditioning — the solution.** Add one input channel `log2(fps / f0)` and widen
  `joints_embed` from `(512,3)` to `(512,4)`. The native rate does not degrade, and the 10 / 5 fps
  errors drop by **22% / 39%**.
* **① frame-rate augmentation — effective but costly.** It flattens the curve by sacrificing the
  origin (21.7 → 41.9, **93% worse**); it only overtakes the baseline at the lowest rate.
  **Use it when the deployment rate is unknown or varied — it is not a universal improvement.**
* **② time-normalised window — no gain.** Fixing the window in *seconds* and scaling the frame count
  by the rate helps neither dataset. At 5 fps the required window is 6 frames — too short to carry
  temporal context. **The countermeasure is in training and architecture, not in inference.**

### ★ Two-domain comparison — ③ is the only universal answer

③ was also run on a **self-collected long-weapon 200 fps** capture (same four-rate merge + dt
channel; held-out test scene `f21`, best validation 53.0 mm):

| Domain | Countermeasure | Curve steepness (5 fps vs native) | Origin |
|---|---|---|---|
| **Harmony4D** (public) | base | **+89%** | — |
| | ① frame-rate augmentation | +0% | **+93% (origin sacrificed)** |
| | ② time-normalised window | slightly worse | none |
| | **★③ Δt conditioning** | **+14%** | **intact** |
| **Long-weapon** (200 fps) | **★③ Δt conditioning** | **+13.5%** | **intact** |

Full ladder (long-weapon): `200→62.3 / 100→58.9 / 50→56.5 / 25→60.0 / 5→70.7 mm`.

> **③ lands at +13–14% steepness on both domains** — a **two-domain-validated** countermeasure.
> ① flattens only on long-weapon data; ② does nothing on either. **Δt conditioning is the only one
> that holds across domains.**

---

## 3. A picture across three datasets, and a hypothesis we refuted ourselves

| Dataset | Frame-rate range | 20→5 fps degradation | absolute increment at 5 fps |
|---|---|---|---|
| **Harmony4D** (public, combat) | 20 → 2 | **+89%** | **+20 mm** |
| **Self-collected long-weapon** (private) | 200 → 5 | +64% | — |
| **Panoptic** (public, domed, slow) | 30 → 8 | **+1% (flat)** | buried in a 172 mm domain gap |

Our original hypothesis was *"the faster the motion, the larger the degradation"*. **The data refutes
it**: stratifying the six Harmony4D activity classes by median inter-frame displacement gives
`Pearson r = −0.125`, `Spearman ρ = −0.300 (p = 0.62)` — no correlation (the slowest class,
ballroom, degrades the most).

In **absolute** terms the regularity is cleaner: every class gains **+15 to +25 mm at 5 fps**,
independent of speed. Hence:

> **The frame-rate mismatch costs an approximately *constant absolute* error (~+20 mm at 5 fps).
> Its *relative* impact depends on how good the model already is** — a good model (21 mm) shows
> +89%, a poor one (172 mm, Panoptic) shows +12% and the penalty is swamped by the domain gap.

---

## 4. Reproduce

```bash
# 1) build 20 / 10 / 5 fps copies of an npz dataset (no network change)
python fps_axis.py build --data <npz_dir> --out-root <out> --strides 1 2 4

# 2) Δt conditioning: the only change is one input channel + one column in joints_embed
python fps_axis.py train --data <d20>,<d10>,<d5> --fps-of-dir 20,10,5 \
       --dt-cond --val-takes <held_out_takes> --out h4d_dt.pt

# 3) the ladder (+ the inference-time control in the second column)
python fps_axis.py eval  --ckpt h4d_dt.pt --data <d20> --dt-cond \
       --train-fps 20 --rates 20 10 5 4 2
```

### Pitfalls we hit (worth knowing before you reproduce)

1. **Validation set must not mix frame rates.** If the training set merges several rates and the
   validation set does too, the readings are not comparable. Take the validation set from **one**
   rate copy.
2. **Calibrate on `train + val`.** `kb_train.calibrate` stores an alignment per calibration key; if
   you pass only the training sequences, `evaluate()` raises `KeyError` on the validation takes.
3. **A 4-channel model needs a 4-channel calibration model.** `calibrate()` runs the model on the
   input; if the model is 3-channel and the input is 4-channel you get
   `mat1 and mat2 shapes cannot be multiplied`.
4. **The model output is in native units, the target is in mm.** Multiply the output by the
   calibrated scale `s` before comparing; otherwise the "error" you measure is just the magnitude of
   the ground truth (we lost an hour to this).
5. **Keep `clip_len` identical across arms.** Comparing a 121-frame evaluation against a 243-frame
   one is not a comparison.

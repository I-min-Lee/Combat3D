#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fps_axis.py -- the frame-rate axis of monocular 3D lifting.

Every published temporal monocular 3D lifter specifies its temporal receptive field in
*FRAMES* (VideoPose3D / dilated TCN: 243 frames; LAMP-Net: "4 seconds", parenthetically
120 frames at 30 Hz; MotionBERT: ``maxlen = 243``).  The frame-to-second conversion IS
the frame rate -- and the frame rate is not controlled at deployment (broadcast runs at
25/30/50/60 fps, phones reach 240 fps).

This script provides the three operations behind our measurements:

    build   -- build the same scenes at several frame rates (stride 1/2/4 ...),
               so a training set can *mix* frame rates (= frame-rate augmentation).
    train   -- fine-tune a lifter, optionally with **dt conditioning** (an extra input
               channel log2(fps/f0); the ratio is what the network needs, not fps).
    eval    -- the cross-frame-rate ladder: evaluate a checkpoint at several test rates.
               Also supports the inference-time "time-normalised window" control, where
               the window is fixed in *seconds* (frame count scales with the rate).

------------------------------------------------------------------------------------
KEY NUMBERS (Harmony4D, 52 unseen scenes; validation taken from the 20 fps copy only)
------------------------------------------------------------------------------------
    test fps        base(20)     aug(1x)    win(2x)    dt(3x)
    20                21.7        41.9       21.7       21.9
    10                29.1        40.6       30.2       22.7
     5                41.0        34.7       41.7       24.9
    steepness          +89%        +0%      worse       +14%
    -> 3 (dt conditioning) is the solution: native rate intact, low-rate error -22%/-39%.
    -> 1 (frame-rate augmentation) flattens the curve but sacrifices the origin (+93%).
    -> 2 (time-normalised window, inference-time) gives no gain on either dataset.

The countermeasure lives in TRAINING and in the ARCHITECTURE, not in inference.

------------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------------
    # 1) build 20/10/5 fps copies of an existing npz dataset
    python fps_axis.py build  --data <npz_dir> --out-root <dir> --strides 1 2 4

    # 2) train with dt conditioning (dim_in 3 -> 4)
    python fps_axis.py train  --data <dir_or_csv_of_dirs> --out <ckpt.pt> \
        --fps-of-dir 20,10,5 --dt-cond --epochs 120

    # 3) evaluate the ladder
    python fps_axis.py eval   --ckpt <ckpt.pt> --data <npz_dir> --dt-cond \
        --train-fps 20 --rates 20 10 5 4 2

Notes
-----
* ``--dt-cond`` widens ``joints_embed`` from (dim_feat, 3) to (dim_feat, 4), copying the
  first three columns; this is the *only* network change.
* The dt channel is a constant per clip:  ``log2(fps / train_fps)``.
* The time-normalised-window control scales the clip length to
  ``round(clip_len * fps / train_fps)``, clamped to a usable minimum.
* This file deliberately mirrors the scripts used to produce the paper numbers; it is
  not a framework.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------- helpers
def _load_kt(kb_dir):
    """Import the project's kb_train / kb_common (carried over from the release)."""
    sys.path.insert(0, kb_dir)
    import kb_train as kt          # noqa: E402
    import kb_common as K          # noqa: E402
    return kt, K


def cmd_build(a):
    """Copy a (N, J, C) npz dataset into several frame-rate variants by striding."""
    os.makedirs(a.out_root, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.data, '*.npz')))
    if not files:
        raise SystemExit('no npz found under %s' % a.data)
    print('[build] %d npz, strides=%s' % (len(files), a.strides))
    for s in a.strides:
        d = os.path.join(a.out_root, 'fps_stride%d' % s)
        os.makedirs(d, exist_ok=True)
        for f in files:
            z = np.load(f, allow_pickle=True)
            out = {k: z[k][::s] for k in ('k2d', 'k3d', 'valid') if k in z.files}
            if 'meta' in z.files:
                out['meta'] = z['meta']
            np.savez_compressed(os.path.join(d, os.path.basename(f)), **out)
        n = len(glob.glob(os.path.join(d, '*.npz')))
        print('  stride %d -> %s  (%d files)' % (s, d, n))
    print('[build] done. Merge the directories you want into one training set.')


# ------------------------------------------------------------------------ dt conditioning
def norm2d4(k4):
    """norm2d for 4 input channels: channels 0..2 as usual, channel 3 passes through.

    The bounding box and the root are computed from the *body* joints only (the same
    convention as the 3-channel version), so adding the dt channel does not disturb
    the 2D normalisation.
    """
    A = k4.astype(np.float32).copy()
    xy, conf = A[:, :, :2], A[:, :, 2]
    for t in range(A.shape[0]):
        v = conf[t] > 0.05
        if v.sum() >= 4:
            p = xy[t][v]
            mid = (p.min(0) + p.max(0)) / 2
            span = max(float((p.max(0) - p.min(0)).max()), 1e-3)
            A[t, :, :2] = 2.0 * (xy[t] - mid) / span
        else:
            A[t, :, :2] = 0
    A[:, :, :2] -= A[:, 0:1, :2]
    return A


def build_dt_model(ckpt, K, dim_feat=512, depth=5, num_joints=17, maxlen=243):
    """Load a 3-channel checkpoint into a 4-channel model (joints_embed widened)."""
    import torch
    sys.path.insert(0, K.MB_ROOT)
    from lib.model.DSTformer import DSTformer
    m = DSTformer(dim_in=4, dim_out=3, dim_feat=dim_feat, dim_rep=512, depth=depth,
                  num_heads=8, mlp_ratio=2, maxlen=maxlen, num_joints=num_joints,
                  att_fuse=True)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('model_pos', sd)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    w = sd.get('joints_embed.weight')
    if w is not None and w.shape[1] == 3:
        new = w.new_zeros(w.shape[0], 4)
        new[:, :3] = w
        sd['joints_embed.weight'] = new
        print('[dt] joints_embed %s -> %s' % (tuple(w.shape), tuple(new.shape)))
    missing = m.load_state_dict(sd, strict=False)
    print('[dt] missing=%d unexpected=%d' % (len(missing.missing_keys),
                                             len(missing.unexpected_keys)))
    return m


def cmd_train(a):
    kt, K = _load_kt(a.kb)
    import torch
    dirs = a.data.split(',')
    fps_of_dir = [float(x) for x in a.fps_of_dir.split(',')]
    assert len(dirs) == len(fps_of_dir), '--fps-of-dir must match --data'
    if a.f0 is None:
        a.f0 = max(fps_of_dir)

    tr, va = [], []
    for d, fps in zip(dirs, fps_of_dir):
        seqs = kt.load_all(d, pids=tuple(a.pids))
        dt = np.float32(np.log2(fps / a.f0))
        for s in seqs:
            if a.dt_cond:
                T = len(s['k2d'])
                ch = np.full((T, s['k2d'].shape[1], 1), dt, np.float32)
                s['k2d'] = np.concatenate([s['k2d'], ch], axis=2)
            (va if s['take'] in a.val_takes else tr).append(s)
    print('[data] train=%d val=%d   dt values=%s' %
          (len(tr), len(va), sorted({float(np.median(s['k2d'][:, 0, 3]))
                                     for s in tr}) if a.dt_cond else 'n/a'))

    if a.dt_cond:
        kt.norm2d = norm2d4
        model = build_dt_model(a.init_ckpt or K.CK_FT, K,
                               num_joints=a.num_joints, maxlen=a.maxlen)
    else:
        model = kt.build_official()
    model = model.to(kt.DEV)

    # calibration must see BOTH splits, otherwise evaluate() KeyErrors on val takes
    cal = (build_dt_model(a.init_ckpt or K.CK_FT, K, num_joints=a.num_joints,
                          maxlen=a.maxlen) if a.dt_cond else kt.build_official())
    cal = cal.to(kt.DEV).eval()
    kt.calibrate(cal, tr + va, a.clip_len)
    del cal
    for k in list(kt.CAL):
        kt.CAL[k] = (kt.CAL[k][0], np.eye(3))
    print('[calib] global s=%.1f' % float(np.median([kt.CAL[k][0] for k in kt.CAL])))

    for p in model.parameters():
        p.requires_grad_(True)
    clips, cw = kt.make_clips(tr, a.clip_len, stride=a.stride, min_ok=0.5)
    print('[clips] %d' % len(clips))
    kt.train(model, tr, clips, a.epochs, a.lr, a.batch, a.clips_per_epoch, 0.0,
             log=lambda s, **kw: print(s, flush=True), eval_every=a.eval_every,
             val_seqs=va, clip_len=a.clip_len, ckpt=a.out, clip_w=cw, mode='full',
             patience=a.patience, min_delta=0.5, max_hours=a.max_hours)
    print('[done] ->', a.out)


# ------------------------------------------------------------------------------ ladder
def cmd_eval(a):
    kt, K = _load_kt(a.kb)
    import torch
    if a.dt_cond:
        kt.norm2d = norm2d4
        model = build_dt_model(a.ckpt, K, num_joints=a.num_joints, maxlen=a.maxlen)
    else:
        model = kt.build_official()
        kt.load_ckpt_weights(model, a.ckpt, 'full')
    model = model.to(kt.DEV).eval()

    # ---- data + one-time calibration (see the pitfalls section of the operation log)
    seqs = kt.load_all(a.data, pids=tuple(a.pids))
    cal = (build_dt_model(a.ckpt, K, num_joints=a.num_joints, maxlen=a.maxlen)
           if a.dt_cond else kt.build_official())
    cal = cal.to(kt.DEV).eval()
    kt.calibrate(cal, seqs, a.clip_len)
    del cal
    for k in list(kt.CAL):
        kt.CAL[k] = (kt.CAL[k][0], np.eye(3))
    print('[calib] global s=%.1f' % float(np.median([kt.CAL[k][0] for k in kt.CAL])))

    def downsample(seq, k):
        if k == 1:
            return seq
        o = dict(seq)
        for key in ('k2d', 'k3d', 'valid'):
            o[key] = seq[key][::k]
        return o

    def run(clip):
        pool = []
        with torch.no_grad():
            for s in sub:
                n = len(s['k2d'])
                if n < clip:
                    continue
                k2 = s['k2d']
                if a.dt_cond:
                    k2 = k2.copy()
                    k2[:, :, 3] = np.float32(np.log2(test_fps / a.train_fps))
                x = norm2d4(k2) if a.dt_cond else kt.norm2d(k2)
                G = kt.gt_camera_mm(s, kt.get_cal(s))
                sc = kt.CAL[kt.cal_key(s)][0]
                starts = np.linspace(0, n - clip, min(a.max_clips, n - clip + 1)).astype(int)
                X = torch.from_numpy(np.stack([x[t:t + clip] for t in starts])).float().to(kt.DEV)
                o = model(X).cpu().numpy() * sc
                for i, t in enumerate(starts):
                    g, V = G[t:t + clip], s['valid'][t:t + clip]
                    e = np.linalg.norm(o[i] - g, axis=2)
                    if V.sum():
                        pool.append(e[V])
        return float(np.median(np.concatenate(pool))) if pool else None

    print('\n%-10s %14s %14s' % ('test fps', 'fixed frames', 'time-norm window'))
    base = None
    for test_fps, k in zip(a.rates, a.strides_for_rates):
        sub = [downsample(s, k) for s in seqs]
        r_fixed = run(a.clip_len)
        win = int(round(a.clip_len * test_fps / a.train_fps))
        win = max(a.min_window, min(a.clip_len, win))
        r_win = run(win)
        if base is None and r_fixed:
            base = r_fixed
        f = lambda r: ('%.1f mm' % r) if r else '--'
        rel = ('%+.1f%%' % (100 * (r_fixed / base - 1))) if (base and r_fixed) else ''
        print('%-10d %14s %14s   %s  (window %d frames)' % (test_fps, f(r_fixed), f(r_win), rel, win))
    print('\n(column 1 = the model as trained; column 2 = the inference-time control;'
          '\n the second is reported as a NEGATIVE result in the paper.)')


# --------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='build frame-rate variants by striding')
    b.add_argument('--data', required=True)
    b.add_argument('--out-root', required=True)
    b.add_argument('--strides', type=int, nargs='+', default=[1, 2, 4])
    b.set_defaults(func=cmd_build)

    common = dict(kb=dict(default='../kb',
                          help='directory holding kb_train.py / kb_common.py'),
                  data=dict(required=True),
                  pids=dict(type=int, nargs='+', default=[0, 1]),
                  clip_len=dict(type=int, default=121),
                  maxlen=dict(type=int, default=243),
                  num_joints=dict(type=int, default=17),
                  max_clips=dict(type=int, default=6),
                  dt_cond=dict(action='store_true'))

    t = sub.add_parser('train')
    t.add_argument('--data', required=True, help='comma-separated npz dirs (one per rate)')
    t.add_argument('--fps-of-dir', required=True, help='comma-separated, e.g. 20,10,5')
    t.add_argument('--f0', type=float, default=None, help='reference fps (default: max)')
    t.add_argument('--val-takes', nargs='*', default=[])
    t.add_argument('--out', required=True)
    t.add_argument('--init-ckpt', default=None)
    t.add_argument('--epochs', type=int, default=120)
    t.add_argument('--lr', type=float, default=2e-5)
    t.add_argument('--batch', type=int, default=4)
    t.add_argument('--clips-per-epoch', type=int, default=300)
    t.add_argument('--stride', type=int, default=8)
    t.add_argument('--eval-every', type=int, default=10)
    t.add_argument('--patience', type=int, default=10)
    t.add_argument('--max-hours', type=float, default=2.5)
    for k, v in common.items():
        t.add_argument('--%s' % k, **v)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser('eval')
    e.add_argument('--ckpt', required=True)
    e.add_argument('--train-fps', type=float, required=True,
                   help='the frame rate the checkpoint was trained at')
    e.add_argument('--rates', type=int, nargs='+', default=[20, 10, 5, 4, 2])
    e.add_argument('--strides-for-rates', type=int, nargs='+', default=None)
    e.add_argument('--min-window', type=int, default=24)
    for k, v in common.items():
        e.add_argument('--%s' % k, **v)
    e.set_defaults(func=cmd_eval)

    a = ap.parse_args()
    if a.cmd == 'eval' and a.strides_for_rates is None:
        # default: stride = round(train_fps / rate)
        a.strides_for_rates = [max(1, int(round(a.train_fps / r))) for r in a.rates]
    a.func(a)


if __name__ == '__main__':
    main()

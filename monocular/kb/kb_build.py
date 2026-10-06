#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kb_build —— 把 final13(3D 标签) + vitpose(2D 输入) 打成 MotionBERT 训练用的 npz

★ 身份配对（本脚本的关键）：`vitpose_rtd5000of/` 是**身份修正前的原始 2D**，
  它的 personID 是每视角独立编号，与 3D 的全局 pid 并不一致。
  官方的逐帧映射在 `rtd2/idflip_color_<tag>.json`（键=帧号, 值={视角: flip}）：
        personID_原始 = pid XOR flip
  实测验证（把 3D 投到该视角比 2D 中位像素）：
        f12 v1 flip=0 -> 9.5px(id) vs 220px(swap)    ✓
        f12 v4 flip=1 -> 6.1px(swap) vs 120px(id)    ✓
        f01 v1 flip=1 -> 9.6px(swap) vs 325px(id)    ✓
        f01 v4 flip=0 -> 8.9px(id)  vs 478px(swap)   ✓

每个 (take, view, pid) 一个 npz:
    k2d   (N,17,3) float32   像素(x,y) + conf     —— MB 输入
    k3d   (N,17,4) float32   世界米(x,y,z) + conf —— 标签原始值
    valid (N,17)   bool      该关节有 3D 测量证据 (conf>0)
    meta  json               take/view/pid/fps0/fps/group/calib/stride/n

时间基准统一到 25fps：200fps 组按 stride=8 抽取（视频本就是 8 倍过采样），
25fps 组原样保留 —— 两组时序语义一致，可混训。

用法:
    python kb_build.py --workers 48                    # 全量
    python kb_build.py --takes f12 f01 --audit         # 冒烟 + 重投影审计
"""
import os, sys, json, glob, argparse, traceback
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_common as K

TARGET_FPS = 25
IDFLIP_DIR = f'{K.ROOT}/rtd2'


def _read_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def load_flip(tag):
    """-> dict[frame_int][view_str] = 0/1；缺文件返回 None"""
    p = f'{IDFLIP_DIR}/idflip_color_{tag}.json'
    if not os.path.exists(p):
        return None
    d = _read_json(p)
    return {int(k): v for k, v in d.items()}


def _load_2d_seq(vp_dir, view, pid, n_frames, stride, flip):
    """vitpose -> (N,17,3) H36M17 像素+conf；用 flip 把 personID 映射到全局 pid。"""
    out = np.zeros((n_frames, 17, 3), dtype=np.float32)
    files = sorted(glob.glob(f'{vp_dir}/{view}/*.json'))
    if not files:
        return out
    nflip_missing = 0
    for p in files:
        fr = int(os.path.basename(p)[:6])
        if fr >= n_frames:
            break
        if flip is None:
            want = pid
        else:
            f = flip.get(fr)
            if f is None:
                nflip_missing += 1
                continue
            want = pid ^ int(f.get(view, 0))
        try:
            j = _read_json(p)
        except Exception:
            continue
        ann = None
        for a in j.get('annots', []):
            if int(a.get('personID', -1)) == want:
                ann = a
                break
        if ann is None:
            continue
        kp = np.asarray(ann['keypoints'], dtype=np.float32)      # (17,3) COCO
        if kp.shape[0] < 17:
            continue
        out[fr] = K.map_to_h36m(kp[None], K.H36M_FROM_COCO)[0]
    return out[::stride]


def _load_3d_seq(pid_dir, n_frames, stride):
    """final13/<pid>/keypoints3d/*.json -> (N,17,4) 世界米+conf"""
    out = np.zeros((n_frames, 17, 4), dtype=np.float32)
    files = sorted(glob.glob(f'{pid_dir}/keypoints3d/*.json'))
    if not files:
        return None
    for p in files:
        fr = int(os.path.basename(p)[:6])
        if fr >= n_frames:
            break
        try:
            j = _read_json(p)
        except Exception:
            continue
        if not j:
            continue
        kp = np.asarray(j[0]['keypoints3d'], dtype=np.float32)   # (13,4)
        if kp.shape[0] != 13:
            continue
        out[fr] = K.map_to_h36m(kp[None], K.H36M_FROM_13)[0]
    return out[::stride]


def _audit_reproj(k3d, k2d, cal, view, nmax=300):
    """3D 投到该视角 vs 配对后的 2D：返回中位像素、样本数、2D 命中率"""
    idx = [i for i in range(len(k3d)) if (k3d[i, :, 3] > 0).sum() >= 6]
    if not idx:
        return None, 0, 0.0
    idx = idx[::max(1, len(idx) // nmax)][:nmax]
    errs, hit, tot = [], 0, 0
    for i in idx:
        uv = cal.project(k3d[i:i + 1, :, :3], view)          # (17,2)
        m = (k3d[i, :, 3] > 0) & (k2d[i, :, 2] > 0.05)
        both = (k3d[i, :, 3] > 0)
        tot += int(both.sum())
        hit += int(m.sum())
        if m.sum() < 4:
            continue
        errs.append(float(np.median(np.linalg.norm(uv[m] - k2d[i, m, :2], axis=1))))
    return (float(np.median(errs)) if errs else None, len(errs),
            (hit / tot) if tot else 0.0)


def build_one(tag, group, fps, view, pid, outroot, audit=False):
    f13_root, vp_root = K.take_paths(tag)
    pid_dir = f'{f13_root}/pid{pid}'
    if not os.path.isdir(pid_dir):
        return (tag, view, pid, 'no-3d', None)
    if vp_root is None:
        return (tag, view, pid, 'no-2d', None)
    n_files = len(glob.glob(f'{pid_dir}/keypoints3d/*.json'))
    if n_files == 0:
        return (tag, view, pid, 'empty-3d', None)
    stride = max(1, int(round(fps / TARGET_FPS)))
    flip = load_flip(tag)
    k3d = _load_3d_seq(pid_dir, n_files, stride)
    k2d = _load_2d_seq(vp_root, view, pid, n_files, stride, flip)
    if k3d is None:
        return (tag, view, pid, 'load-3d-fail', None)
    used = min(len(k3d), len(k2d))
    k3d, k2d = k3d[:used], k2d[:used]
    valid = (k3d[:, :, 3] > 0)
    meta = dict(take=tag, view=view, pid=pid, fps_src=fps, fps=TARGET_FPS,
                stride=stride, group=group, n=used,
                calib=K.CALIB[group], flip_src=(flip is not None))
    os.makedirs(outroot, exist_ok=True)
    fn = f'{outroot}/{tag}_v{view}_p{pid}.npz'
    np.savez_compressed(fn, k2d=k2d, k3d=k3d, valid=valid,
                        meta=np.array(json.dumps(meta)))
    info = dict(n=used, fn=os.path.basename(fn), has2d_ratio=float((k2d[:, :, 2] > 0.05).mean()))
    if audit:
        cal = K.Calib(K.CALIB[group])
        med, cnt, hit = _audit_reproj(k3d, k2d, cal, view)
        info['audit'] = dict(med_px=med, n=cnt, hit=hit, calib=K.CALIB[group])
    return (tag, view, pid, 'ok', info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=K.OUT)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--takes', nargs='*', default=None)
    ap.add_argument('--views', nargs='*', default=K.VIEWS)
    ap.add_argument('--pids', nargs='*', type=int, default=[0, 1])
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    takes = K.load_takes()
    if a.takes:
        want = set(a.takes)
        takes = [t for t in takes if t[0] in want]
        if not takes:
            print('!! 场次名无效, 有效例:', [t[0] for t in K.load_takes()[:8]])
            return
    print(f'场次 {len(takes)} 个；views={a.views} pids={a.pids}')

    jobs = []
    for tag, n, fps, group in takes:
        for v in a.views:
            for p in a.pids:
                fn = f'{a.out}/{tag}_v{v}_p{p}.npz'
                if os.path.exists(fn) and not a.overwrite:
                    continue
                jobs.append((tag, group, fps, v, p))
    print(f'待建 {len(jobs)} 个 npz')

    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_one, t, g, f, v, p, a.out, a.audit): (t, v, p)
                for (t, g, f, v, p) in jobs}
        done = 0
        for fu in as_completed(futs):
            t, v, p = futs[fu]
            try:
                results.append(fu.result())
            except Exception as e:
                traceback.print_exc()
                results.append((t, v, p, 'EXC', str(e)[:200]))
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f'  {done}/{len(jobs)}', flush=True)

    ok = [r for r in results if r[3] == 'ok']
    bad = [r for r in results if r[3] != 'ok']
    print(f'\n完成 ok={len(ok)} bad={len(bad)}')
    if ok:
        print('总帧数(25fps域) = %d' % sum(r[4]['n'] for r in ok))
        print('2D 命中率 中位 = %.3f' % float(np.median([r[4]['has2d_ratio'] for r in ok])))
    for r in bad[:20]:
        print('   FAIL', r)

    if a.audit:
        print('\n=== 重投影审计（配对后；中位像素，越小越好）===')
        rows = [r for r in ok if r[4].get('audit') and r[4]['audit']['med_px'] is not None]
        for r in sorted(rows, key=lambda x: (x[0], int(x[1]))):
            au = r[4]['audit']
            print('  %-6s v%-3s p%d  med=%7.2fpx n=%3d 2D命中=%.2f' %
                  (r[0], r[1], r[2], au['med_px'], au['n'], au['hit']))
        if rows:
            meds = [r[4]['audit']['med_px'] for r in rows]
            print('  全体中位 = %.2f px  (最大 %.2f)' % (float(np.median(meds)), float(np.max(meds))))

    json.dump([{'take': r[0], 'view': r[1], 'pid': r[2], 'status': r[3],
                'info': r[4] if isinstance(r[4], dict) else str(r[4])} for r in results],
              open(f'{a.out}/_build_report.json', 'w'), ensure_ascii=False, indent=1)
    print('报告 ->', f'{a.out}/_build_report.json')


if __name__ == '__main__':
    main()

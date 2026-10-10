#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  ✅  UNIVERSAL LAYER — REUSE AS-IS ACROSS SCENES
#
#  这是 Combat3D-Label 五层里【通用】的那几层之一（2D 提取 / 三角化 / SMPL 拟合 /
#  单目 lifter）。两个域上的实测：换场景时这几层【一行未改】。
#
#  与它对照的是【场景相关】的检测框层与身份层 —— 那两个必须换算法，
#  文件头带 "SCENE-SPECIFIC LAYER" 横幅。层契约见 code/adapters/README.md。
# =============================================================================
"""fit_h4d.py — stage6 SMPL 拟合的 Harmony4D 版（`stage6_fit/sweep_full_v3.py` 的适配副本）

★ 与原版的差异只有 5 处（都用 ★H4D 标出），**拟合逻辑/权重表/EasyMocap 调用一字未改**：
   1. B/MASTER/PYO 换成 222 的路径（EasyMocap 从 westc 拉的 52M 副本）
   2. ANNOT_ROOT / K3D_ROOT 指向我们的产物
   3. subs → 我们的 6 视角；ranges → 我们的帧范围
   4. smpl.pkl / J_regressor_body25.npy 指向 port/smpl/
   5. 新增 `--infer-verts`：同时导出**逐帧顶点**（供算 PVE 与公开榜单对照）

用法（容器内，FIT_K 为 fit 平滑权重 k）：
  FIT_K=0.024 python fit_h4d.py <pid> --start 1 --end 61 \
      --annots /workshop/Lym/combat3d/asm_h4d_full/annots \
      --k3d /workshop/Lym/combat3d/em_h4d_full/lam1.0 \
      --calib /workshop/Lym/combat3d/calib_h4d_016mma4 \
      --frames /workshop/Lym/combat3d/frames/15/4 \
      --out /workshop/Lym/combat3d/emfit_h4d
"""
import os, sys, json, glob, re, subprocess, argparse
import numpy as np

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')
MASTER = os.environ.get('EM_MASTER', B + '/port/EasyMocap-master')      # ★H4D-1
PYO = os.environ.get('EM_PY', B + '/envs/miniconda3/envs/pose312/bin/python')
SMPL = B + '/port/smpl'                                                 # ★H4D-4
SUB_VIEWS = ['01', '03', '04', '07', '09', '14']                        # ★H4D-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pid', type=int)
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=61, help='ranges 上界（不含）')
    ap.add_argument('--annots', default=B + '/asm_h4d_full/annots', help='含 {view}/{fr:06d}.json 的目录')
    ap.add_argument('--k3d', default=B + '/em_h4d_full/lam1.0', help='三角化根（下面有 pid{p}/keypoints3d）')
    ap.add_argument('--calib', default=B + '/calib_h4d_016mma4')
    ap.add_argument('--frames', default=B + '/frames/15/4')
    ap.add_argument('--out', default=B + '/emfit_h4d')
    ap.add_argument('--views', default=','.join(SUB_VIEWS))
    ap.add_argument('--zero-base', action='store_true', help='★H4D-6 建 0-based 输入树（索引 i ↔ Harmony4D 帧 i+1），对齐 EasyMocap 的 ranges 约定')
    a = ap.parse_args()
    pid = a.pid
    k = float(os.environ.get('FIT_K', '0.024'))          # ★H4D-3b：20fps 先用 25fps 标定值
    N = a.end - a.start
    views = [v.zfill(2) for v in a.views.split(',')]

    # ---- exp：从 EasyMocap 官方 mv1p3d.yml 派生，只替换平滑权重（与原版一致） ----
    lines = open(MASTER + '/config/fit/mv1p3d.yml', encoding='utf-8').read().split('\n')
    spec = {'sTh': k, 'sRh': k, 'skpts2': k, 'spose2': k / 100.0, 'spose1': k / 10000.0}
    out, stack, n = [], [], 0
    for ln in lines:
        ind = len(ln) - len(ln.lstrip())
        mb = re.match(r'^\s*([A-Za-z_][A-Za-z_0-9]*):\s*$', ln)
        if mb:
            while stack and stack[-1][0] >= ind:
                stack.pop()
            stack.append((ind, mb.group(1))); out.append(ln); continue
        m2 = re.match(r'^(\s*)weight:\s*([0-9.eE+-]+)\s*$', ln)
        if m2 and stack and stack[-1][1] in spec and abs(k - 1.0) > 1e-9:
            # ★ 必须写成带小数点的定点形式：PyYAML 的 float 正则要求含 '.'，
            #   否则 `2e-07` 会被解析成**字符串** → EasyMocap 报 Type mismatch（踩过）
            out.append('%sweight: %.12f' % (m2.group(1), spec[stack[-1][1]])); n += 1; continue
        out.append(ln)
    d = '%s/pid%d' % (a.out, pid)
    os.makedirs(d, exist_ok=True)
    exp = '%s/exp_k%s.yml' % (d, k)
    open(exp, 'w', encoding='utf-8').write('\n'.join(out))
    print('[k=%s pid%d] exp 写入 (%s)，平滑替换 %d 处' % (k, pid, exp, n), flush=True)

    # ---- per-pid 目录：symlink 成 EasyMocap 期望的结构（与原版一致） ----
    # ★H4D-6：EasyMocap 要求帧号 **0-based 连续**且 `ranges[1] <= 文件数`（他们自己的数据是
    #   000000..014016 + ranges [0,14017,1]）。Harmony4D 的帧是 1-based，因此当 --zero-base 时
    #   先建一套"索引 i ↔ Harmony4D 帧 i+1"的 0-based 输入树（纯 symlink），再喂给 fit。
    src_annots, src_frames, src_k3d = a.annots, a.frames, a.k3d
    if a.zero_base:
        root_in = os.path.join(a.out, '_in')
        nfr = 0
        for v in views:
            for kind, src, ext, sub in (('annots', a.annots, 'json', v),
                                        ('frames', a.frames, 'png', v)):
                od = os.path.join(root_in, kind, v)
                os.makedirs(od, exist_ok=True)
                fs = sorted(glob.glob(os.path.join(src, v, '*' + ext)))
                for f in fs:
                    fi = int(os.path.basename(f).split('.')[0])
                    dstl = os.path.join(od, '%06d.%s' % (fi - 1, ext))     # ★ i ↔ 帧 i+1
                    if not os.path.lexists(dstl):
                        os.symlink(f, dstl)
                if kind == 'annots':
                    nfr = max(nfr, len(fs))
        kd = os.path.join(root_in, 'pid%d' % pid, 'keypoints3d')
        os.makedirs(kd, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(a.k3d, 'pid%d' % pid, 'keypoints3d', '*.json'))):
            fi = int(os.path.basename(f).split('.')[0])
            dstl = os.path.join(kd, '%06d.json' % (fi - 1))
            if not os.path.lexists(dstl):
                os.symlink(f, dstl)
        src_annots = os.path.join(root_in, 'annots')
        src_frames = os.path.join(root_in, 'frames')
        src_k3d = os.path.join(root_in, 'pid%d' % pid, 'keypoints3d')
        r0, r1 = 0, nfr
        print('[k=%s pid%d] ★0-based 输入树已建：索引 0..%d ↔ Harmony4D 帧 1..%d'
              % (k, pid, nfr - 1, nfr), flush=True)
    else:
        r0, r1 = a.start, a.end

    k3d_link = src_k3d if a.zero_base else ('%s/pid%d/keypoints3d' % (a.k3d, pid))
    links = [('annots', src_annots), ('images', src_frames),
             ('intri.yml', os.path.join(a.calib, 'intri.yml')),
             ('extri.yml', os.path.join(a.calib, 'extri.yml')),
             ('keypoints3d', k3d_link)]
    for nm, src in links:
        lnk = os.path.join(d, nm)
        if os.path.lexists(lnk):
            os.remove(lnk)
        os.symlink(src, lnk)
    os.makedirs(d + '/fit', exist_ok=True)

    # ---- data cfg：从原版模板派生，替路径/视角/帧范围 ----
    tpl = open(B + '/code/label_pipeline/stages/6_fit/data_fit_off_p0_14k.yml', encoding='utf-8').read()
    txt = (tpl.replace('/root/autodl-tmp/emfit_rtd14k/pid0', d)
              .replace("subs: ['1','3','4','7','11']",
                       "subs: [%s]" % ','.join("'%s'" % v for v in views))
              .replace('ranges: [0, 14017, 1]', 'ranges: [%d, %d, 1]' % (r0, r1))
              .replace('pid: 0', 'pid: %d' % pid))
    assert 'ranges: [%d, %d, 1]' % (r0, r1) in txt, 'ranges 未替换'
    assert d in txt, 'path 未替换'
    cfg = '%s/data_p%d.yml' % (d, pid)
    open(cfg, 'w', encoding='utf-8').write(txt)
    print('[k=%s pid%d] cfg_data 写入 %s（ranges %d..%d，视角 %s）' % (k, pid, cfg, r0, r1, views), flush=True)

    # ---- 调 EasyMocap fit（与原版同一条命令） ----
    env = dict(os.environ); env['PYTHONPATH'] = MASTER
    cmd = [PYO, 'apps/fit/fit.py', '--cfg_data', cfg, '--cfg_model', 'config/model/smpl.yml',
           '--opt_model', 'args.model_path', SMPL + '/smpl.pkl',
           'args.regressor_path', SMPL + '/J_regressor_body25.npy', '--cfg_exp', exp]
    print('[k=%s pid%d] fit 开始（%d 帧）' % (k, pid, N), flush=True)
    log = '%s/fit_p%d.log' % (a.out, pid)
    with open(log, 'w') as lf:
        r = subprocess.run(cmd, cwd=MASTER, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print('[k=%s pid%d] fit 退出码 %d（日志 %s）' % (k, pid, r.returncode, log), flush=True)
    if r.returncode:
        print(open(log).read()[-2500:])
        sys.exit(1)

    # ---- fit 结果 → b25 关节（与原版一致） ----
    sys.path.insert(0, MASTER)
    import torch
    from easymocap.bodymodel.smpl import SMPLModel
    m = SMPLModel(model_path=SMPL + '/smpl.pkl', regressor_path=SMPL + '/J_regressor_body25.npy',
                  device='cuda', use_pose_blending=True, use_joints=True, NUM_SHAPES=10)
    dst = '%s/keypoints3d_b25' % d; os.makedirs(dst, exist_ok=True)
    vdst = '%s/verts' % d; os.makedirs(vdst, exist_ok=True)          # ★H4D-5：顶点（供 PVE）
    for fp in sorted(glob.glob(d + '/fit/smpl/*.json')):
        j = json.load(open(fp)); an = j['annots'][0] if isinstance(j, dict) and 'annots' in j else j[0]
        p = {q: torch.tensor(an[q], dtype=torch.float32).cuda() for q in ['shapes', 'poses', 'Rh', 'Th']}
        with torch.no_grad():
            kp = m(return_verts=False, **p)
            vv = m(return_verts=True, **p)
        b25 = kp[0].cpu().numpy().astype(np.float64)
        verts = (vv.vertices[0] if hasattr(vv, 'vertices') else vv[0]).cpu().numpy().astype(np.float32)
        bn = os.path.basename(fp)
        json.dump([{'id': pid, 'keypoints3d': [[float(b25[q, 0]), float(b25[q, 1]), float(b25[q, 2]), 1.0]
                                               for q in range(25)]}], open(os.path.join(dst, bn), 'w'))
        np.save(os.path.join(vdst, bn.replace('.json', '.npy')), verts)
    print('[k=%s pid%d] b25 -> %d 帧，顶点 -> %d 帧' % (k, pid, len(os.listdir(dst)), len(os.listdir(vdst))), flush=True)
    print('[k=%s pid%d] ALL_DONE' % (k, pid), flush=True)


if __name__ == '__main__':
    main()

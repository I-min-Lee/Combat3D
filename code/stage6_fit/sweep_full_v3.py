#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sweep_full.py <k> <pid>  — 全量 14017 帧: k3d-Hampel 输入 + fit 平滑权重 x k + extract b25
产物 /root/autodl-tmp/emfit_full_k<k>/pid<pid>/{fit/smpl, keypoints3d_b25}
"""
import os, sys, json, glob, re, subprocess
import numpy as np, torch
B = '/root/autodl-tmp'
MASTER = B + '/emoff/EasyMocap-master'
PYO = B + '/emoff/env/bin/python'
k = float(sys.argv[1]); pid = int(sys.argv[2])
import os as _os
N = int(_os.environ.get('SWEEP_N', '14017'))   # 帧数：B组 0.1 传 2428
ANNOT_ROOT = _os.environ.get('ANNOT_ROOT', B + '/easymocap_data_rtd14k/1_1')
root = _os.environ.get('OUT_ROOT', '%s/emfit_full_k%d' % (B, int(k)))
os.makedirs(root, exist_ok=True)
K3D = _os.environ.get('K3D_ROOT', B + '/em_rtd14k_h11')

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
        out.append('%sweight: %s' % (m2.group(1), spec[stack[-1][1]])); n += 1; continue
    out.append(ln)
exp = '%s/exp_k%d.yml' % (root, int(k))
open(exp, 'w', encoding='utf-8').write('\n'.join(out))
print('[k=%s pid%d] exp 写入, 平滑替换 %d 处' % (k, pid, n), flush=True)

d = '%s/pid%d' % (root, pid)
os.makedirs(d, exist_ok=True)
for nm in ['annots', 'images', 'intri.yml', 'extri.yml']:
    lnk = os.path.join(d, nm)
    if os.path.lexists(lnk): os.remove(lnk)
    os.symlink(os.path.join(ANNOT_ROOT, nm), lnk)
lnk = d + '/keypoints3d'
if os.path.lexists(lnk): os.remove(lnk)
os.symlink('%s/pid%d/keypoints3d' % (K3D, pid), lnk)
os.makedirs(d + '/fit', exist_ok=True)
cfg = '%s/data_p%d.yml' % (d, pid)
txt = open(B + '/rtd3cfg/data_fit_off_p%d_14k.yml' % pid, encoding='utf-8').read()
txt = txt.replace(B + '/emfit_rtd14k/pid%d' % pid, d)
_R0 = 'ranges: [0, 14017, 1]'
assert _R0 in txt, 'ranges 模板不符'
if N != 14017:
    txt = txt.replace(_R0, 'ranges: [0, %d, 1]' % N)
assert ('ranges: [0, %d, 1]' % N) in txt, 'ranges 未落到 N=%d' % N
open(cfg, 'w', encoding='utf-8').write(txt)

env = dict(os.environ); env['PYTHONPATH'] = MASTER
cmd = [PYO, 'apps/fit/fit.py', '--cfg_data', cfg, '--cfg_model', 'config/model/smpl.yml',
       '--opt_model', 'args.model_path', B + '/smpl/smpl.pkl',
       'args.regressor_path', B + '/smpl/J_regressor_body25.npy', '--cfg_exp', exp]
print('[k=%s pid%d] fit 开始 %s' % (k, pid, '全量%d' % N), flush=True)
with open('%s/fit_p%d.log' % (root, pid), 'w') as lf:
    r = subprocess.run(cmd, cwd=MASTER, env=env, stdout=lf, stderr=subprocess.STDOUT)
print('[k=%s pid%d] fit 退出码 %d' % (k, pid, r.returncode), flush=True)
if r.returncode:
    sys.exit(1)
sys.path.insert(0, MASTER)
from easymocap.bodymodel.smpl import SMPLModel
m = SMPLModel(model_path=B + '/smpl/smpl.pkl', regressor_path=B + '/smpl/J_regressor_body25.npy',
              device='cuda', use_pose_blending=True, use_joints=True, NUM_SHAPES=10)
dst = '%s/keypoints3d_b25' % d; os.makedirs(dst, exist_ok=True)
for fp in sorted(glob.glob(d + '/fit/smpl/*.json')):
    j = json.load(open(fp)); a = j['annots'][0] if isinstance(j, dict) and 'annots' in j else j[0]
    p = {q: torch.tensor(a[q], dtype=torch.float32).cuda() for q in ['shapes', 'poses', 'Rh', 'Th']}
    with torch.no_grad(): kp = m(return_verts=False, **p)
    b25 = kp[0].cpu().numpy().astype(np.float64)
    json.dump([{'id': pid, 'keypoints3d': [[float(b25[q,0]),float(b25[q,1]),float(b25[q,2]),1.0] for q in range(25)]}],
              open(os.path.join(dst, os.path.basename(fp)), 'w'))
print('[k=%s pid%d] b25 -> %d' % (k, pid, len(os.listdir(dst))), flush=True)
print('[k=%s pid%d] ALL_DONE' % (k, pid), flush=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kb_test_eval —— 对**测试集**做一次性评测（独立于训练，随时可跑）

用途：
  · 训练被提前停掉时，仍然能拿到测试集成绩
  · 对比不同 checkpoint 的测试集表现

标定 (s,R) 直接从训练产出的 <ckpt>_eval.json 里读（训练时算好的，不要重算，
重算会用到训练集、虽然不影响测试集指标但口径要一致）。

用法:
  python kb_test_eval.py --ckpt /workshop/Lym/combat3d/mb/ckpt/kb_full_v2.pt
  python kb_test_eval.py --ckpt xxx.pt --also-val      # 顺带也评验证集
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_common as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default=K.OUT)
    ap.add_argument('--split-json', default=f'{K.ROOT}/mb/data/_split.json')
    ap.add_argument('--clip-len', type=int, default=243)
    ap.add_argument('--max-clips', type=int, default=40)
    ap.add_argument('--sets', default='test', choices=['test', 'val', 'both'],
                    help='评哪些集合；测试集请只在最终模型上开封')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    import kb_train as kt

    # ---- 划分（先加载，后面标定要用 train 名单）----
    sp = json.load(open(a.split_json))
    test_takes = set(sp['test'])
    print(f'[split] 测试集 {len(test_takes)} 场: {sorted(test_takes)}')

    # ---- 模型 ----
    sd = torch.load(a.ckpt, map_location='cpu')
    mode = 'lora' if any(k.endswith('.A') for k in sd) else 'full'
    ej = a.ckpt.replace('.pt', '_eval.json')
    cj = a.ckpt.replace('.pt', '_calib.json')
    if os.path.exists(ej):
        try:
            mode = json.load(open(ej)).get('mode', mode)
        except Exception:
            pass
    m = kt.build_official()
    if mode == 'lora':
        m = kt.to_lora(m, 8)
    m.load_state_dict(sd, strict=(mode == 'full'))
    m.to(kt.DEV).eval()
    print(f'[ckpt] {a.ckpt}  mode={mode}')

    # ---- 标定：优先读训练产物；没有就在【训练集】上重算（不碰 val/test）----
    CAL = {}
    for cand in (ej, cj):
        if os.path.exists(cand):
            for v, (s, R) in json.load(open(cand)).get('calib', {}).items():
                CAL[v] = (s, np.array(R))
            print(f'[calib] 从 {cand} 读到 {len(CAL)} 个视角标定')
            break
    kt.CAL_OBJ = K.Calib(K.CALIB['960'])      # 仅作几何容器（两组外参相同）
    kt.CAL = CAL
    if not CAL:
        print(f'!! 既无 {ej} 也无 {cj} —— 在【训练集】上重算标定（与训练时口径一致，不碰 val/test）')
        tr_seq = kt.load_all(a.data, set(sp['train']), exclude=kt.load_exclude(a.data))
        print(f'  训练集 {len(tr_seq)} 个序列用于标定 ...')
        kt.calibrate(m, tr_seq, a.clip_len)

    excl = kt.load_exclude(a.data)

    res = {}
    if a.sets in ('test', 'both'):
        te = kt.load_all(a.data, test_takes, exclude=excl)
        nf = sum(len(s['k2d']) for s in te)
        print(f'== 测试集评测（{len(test_takes)} 场 / {len(te)} 序列 / {nf} 帧，全程未参与训练）==')
        res['test'] = kt.evaluate(m, te, a.clip_len, max_clips=a.max_clips)
        n_frames = nf
    if a.sets in ('val', 'both'):
        va = kt.load_all(a.data, set(sp['val']), exclude=excl)
        print(f'== 验证集评测（{len(va)} 序列 / {sum(len(s["k2d"]) for s in va)} 帧）==')
        res['val'] = kt.evaluate(m, va, a.clip_len, max_clips=a.max_clips)
        n_frames = sum(len(s['k2d']) for s in va)

    out = a.out or a.ckpt.replace('.pt', '_TEST.json')
    json.dump(dict(ckpt=a.ckpt, mode=mode, res=res,
                   test_takes=sorted(test_takes),
                   n_frames=n_frames),
              open(out, 'w'), ensure_ascii=False, indent=1)
    print()
    print('========= 结论 (mm) =========')
    for k, v in res.items():
        # 标准 MPJPE = 全部关节观测的均值 = clean/occ 按关节数加权合并
        nc, no = v['n_clean'], v['n_occ']
        mpjpe = ((v['clean_mean_mm'] * nc + v['occ_mean_mm'] * no) / (nc + no)) if (nc + no) else float('nan')
        print(f'--- {k} 集 (关节观测 {v["n_joint_obs"]} 个: 可见 {nc} / 遮挡 {no}) ---')
        print(f'  标准 MPJPE（全部关节 均值）      = {mpjpe:8.1f}')
        print(f'  中位   MPJPE（全部关节 pooled中位）= {v["med_mm"]:8.1f}   ← 与上一轮 156mm 同口径')
        print(f'    其中 可见关节 中位/均值        = {v["clean_mm"]:8.1f} / {v["clean_mean_mm"]:.1f}')
        print(f'    其中 遮挡关节 中位/均值        = {v["occ_mm"]:8.1f} / {v["occ_mean_mm"]:.1f}')
        print(f'    双腕       中位/均值           = {v["wrist_mm"]:8.1f} / {v["wrist_mean_mm"]:.1f}')
        json.dump(dict(mpjpe_std=mpjpe, **v), open(out.replace('.json', '_detail.json'), 'w'),
                  ensure_ascii=False, indent=1)
    print('->', out)


if __name__ == '__main__':
    main()

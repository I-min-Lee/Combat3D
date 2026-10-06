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
"""kb_train —— 用 final13(3D) 当标签、vitpose 2D 当输入，微调 MotionBERT 做单目 2D->3D

配方沿用上一轮定稿 mb_ft4.py（已验证 156mm）:
  ① 逐帧 2D 归一化 x' = 2(x-mid)/span，再逐帧根置零
  ② 标定 pass：官方权重前向训练集，每 (视角) 用 Kabsch 求 (s,R) 把 GT(相机系 mm)
     映到 MB 原生输出系；**没有这步 loss 被常数淹没、看起来"学不动"**
  ③ LoRA(r=8, scale=2) 只训 qkv/proj，A/B 零初始化
  ④ 掩码 MPJPE（只看有测量证据的关节）+ 遮挡链增广
  ⑤ 评测：mm 中位 MPJPE（总体 / 遮挡关节 / 双腕专项）

数据侧与上一轮的区别：序列长 2000~24000 帧 >> maxlen=243，所以按 clip 切窗。

用法:
  # 冒烟（2 场次、20 epoch）
  python kb_train.py --smoke --epochs 20
  # 正式
  python kb_train.py --epochs 150 --clips-per-epoch 2500 --out /workshop/Lym/combat3d/mb/ckpt/kb_lora_v1.pt
"""
import os, sys, json, glob, argparse, time, random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_common as K

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ------------------------------------------------------------------ 数据
def load_exclude(datadir):
    """读 _exclude.json（kb_exclude.py 产出）-> set of (take,view,pid)"""
    p = f'{datadir}/_exclude.json'
    if not os.path.exists(p):
        return set()
    j = json.load(open(p))
    s = {tuple(x) for x in j.get('pairs', [])}
    if s:
        print(f'[exclude] 已按 {p} 排除 {len(s)} 个 (take,view,pid)')
    return s


def load_quality(datadir):
    """读 kb_quality.py 产出的 _quality.json -> {tag: bool 掩码(25fps 帧) 由 ranges 展开}"""
    p = f'{datadir}/_quality.json'
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    out = {}
    for tag, d in j.get('takes', {}).items():
        n = int(d.get('n25', 0))
        m = np.zeros(n, dtype=bool)
        for f0, f1 in d.get('good', []):
            m[int(f0):int(f1)] = True
        out[tag] = m
    print(f'[quality] 载入报告级掩码 {len(out)} 场次（保留 '
          f'{100*sum(int(m.sum()) for m in out.values())/max(1,sum(len(m) for m in out.values())):.1f}% 帧）')
    return out


def load_all(datadir, takes_filter=None, pids=(0, 1), views=None, exclude=None,
             qa_dir=None, qa_thr=10.0, qa_min_n=4, quality=None):
    """返回 list of dict(seq)。seq['ok'](N,) bool = 该帧可用于训练（两级筛选合并）。
       · 报告级: _quality.json 的 5 秒分箱（★/jumps/conf/low）
       · 几何级: data_qa 的逐帧 3D->2D 重投影残差 qa（按 fx 归一化到 960 域）
    """
    seqs = []
    exclude = exclude or set()
    n_qa = n_qb = 0
    for fn in sorted(glob.glob(f'{datadir}/*.npz')):
        base = os.path.basename(fn)[:-4]
        try:
            take, v, p = base.rsplit('_', 2)
            v = v[1:]
            p = int(p[1:])
        except Exception:
            continue
        if (take, v, p) in exclude:
            continue
        if take == '_build_report':
            continue
        if views and v not in views:
            continue
        if p not in pids:
            continue
        if takes_filter and take not in takes_filter:
            continue
        z = np.load(fn, allow_pickle=True)
        meta = json.loads(str(z['meta']))
        k2d, k3d, valid = z['k2d'], z['k3d'], z['valid']
        n = len(k2d)
        ok = np.ones(n, dtype=bool)
        # --- 几何级：逐帧重投影残差 ---
        if qa_dir:
            qp = f'{qa_dir}/{base}.npz'
            if os.path.exists(qp):
                zq = np.load(qp)
                qa, qn = zq['qa'], zq['qa_n']
                m = min(len(qa), n)
                ok[:m] &= (qa[:m] >= 0) & (qa[:m] <= qa_thr) & (qn[:m] >= qa_min_n)
                n_qa += 1
        # --- 报告级：5 秒分箱 ---
        if quality is not None and take in quality:
            qm = quality[take]
            m = min(len(qm), n)
            ok[:m] &= qm[:m]
            n_qb += 1
        seqs.append(dict(take=take, view=v, pid=p, group=meta['group'],
                         fps=meta['fps'], k2d=k2d, k3d=k3d, valid=valid, ok=ok))
    if qa_dir:
        print(f'[qa] 逐帧残差已应用: {n_qa} 个序列 (阈值 {qa_thr}px, 最少 {qa_min_n} 关节)')
    return seqs


def norm2d(k2d):
    """逐帧归一化 + 根置零。(T,17,3) -> (T,17,3)"""
    x = k2d.astype(np.float32).copy()
    T = x.shape[0]
    xy = x[:, :, :2]
    v = x[:, :, 2] > 0.05
    for t in range(T):
        vt = v[t]
        if vt.sum() >= 4:
            p = xy[t][vt]
            mid = (p.min(0) + p.max(0)) / 2
            span = max(float((p.max(0) - p.min(0)).max()), 1e-3)
            x[t, :, :2] = 2.0 * (xy[t] - mid) / span
        else:
            x[t, :, :2] = 0
    x[:, :, :2] -= x[:, 0:1, :2]          # 根置零
    return x


# ★2026-10-04 H4D：COLMAP 是单目 SfM，**每条序列的世界尺度本来就是任意的**，
#   而这里把 COLMAP 单位当成米。实测各场次尺度散布 2.73 倍（train01_hugging 差 2.3 倍），
#   单一全局标定 (s,R) 根本拟合不上 -> 训练目标自相矛盾。
#   X_metric = S @ X_colmap  =>  1 COLMAP 单位 = ||S[:,0]|| 米。
#   有 scale.npy 的用真值；无 scale.npy 的 17 条用「官方2D三角化 + Umeyama 对齐官方3D」估出。
_METRIC_SCALE = {}
try:
    with open(f'{K.ROOT}/h4d_metric_scale.json') as _f:
        _METRIC_SCALE = json.load(_f)
except Exception as _e:
    print(f'[metric-scale] 未加载 ({_e})', flush=True)


def gt_camera_mm(seq, cal):
    """世界米 -> 相机系 mm，逐帧根置零。(T,17,3)"""
    Xw = seq['k3d'].astype(np.float64)[:, :, :3]
    R, T = cal.R[seq['view']], cal.T[seq['view']]
    Xc = (R @ Xw.reshape(-1, 3).T).T.reshape(Xw.shape) + T           # COLMAP 单位
    Xc = Xc * 1000.0
    _ms = _METRIC_SCALE.get(seq.get('take'))                          # ★H4D 尺度归一
    if _ms:
        Xc = Xc * _ms
    Xc -= Xc[:, 0:1, :]
    return Xc.astype(np.float32)


# ------------------------------------------------------------------ 模型
def build_official():
    sys.path.insert(0, K.MB_ROOT)
    from lib.model.DSTformer import DSTformer
    sd = torch.load(K.CK_FT, map_location='cpu', weights_only=False)['model_pos']
    m = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5,
                  num_heads=8, mlp_ratio=2, maxlen=243, num_joints=17, att_fuse=True)
    m.load_state_dict({k[7:]: v for k, v in sd.items()}, strict=True)
    return m


class LoRA(nn.Module):
    def __init__(self, lin, r=8, s=2.0):
        super().__init__()
        self.base = lin
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(nn.init.trunc_normal_(torch.zeros(r, lin.in_features), std=.02))
        self.B = nn.Parameter(torch.zeros(lin.out_features, r))
        self.s = s

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.s


def to_lora(m, r=8):
    for _, mod in list(m.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear) and cn in ('qkv', 'proj'):
                setattr(mod, cn, LoRA(ch, r))
    for n, p in m.named_parameters():
        p.requires_grad_('.A' in n or '.B' in n)
    print('LoRA 可训参数 %.2fM' % (sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6))
    return m


def merge_lora_into(model, lora_path, scale=2.0):
    """把上一轮的 LoRA 增量 (W' = W + s·B@A) 合并进基础权重，
    得到一个"普通"的全量模型 —— 这样就能从上一轮微调结果继续做全量微调。
    LoRA 前向是 y = Wx + s·(x@Aᵀ@Bᵀ)，故 W' = W + s·B@A。
    """
    sd = torch.load(lora_path, map_location='cpu')
    mods = dict(model.named_modules())
    n = 0
    for k, A in list(sd.items()):
        if not k.endswith('.A'):
            continue
        base = k[:-2]
        B = sd.get(base + '.B')
        m = mods.get(base)
        if B is None or m is None or not hasattr(m, 'weight'):
            print('  [merge] 跳过未知模块', base)
            continue
        with torch.no_grad():
            m.weight.add_(scale * (B.float() @ A.float()))
        n += 1
    print(f'[merge] 合并 {n} 个 LoRA 增量 (scale={scale}) <- {os.path.basename(lora_path)}')
    return model


def build_init(init, ckpt_path=None):
    """热启动点：official=官方；lora5/lora4=官方+上一轮LoRA增量；fullft=上一轮全量微调；
    或 ckpt_path 直接给一个全量 .pt。"""
    m = build_official()
    if ckpt_path:
        m.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
        print(f'[init] 从指定权重热启动: {ckpt_path}')
        return m
    prev = f'{K.ROOT}/mb/prev'
    if init in ('lora5', 'lora4'):
        p = f'{prev}/mb_lora_v5.pt' if init == 'lora5' else f'{prev}/mb_lora_v4.pt'
        if not os.path.exists(p):
            print(f'!! 找不到 {p}，回退官方权重'); return m
        merge_lora_into(m, p, 2.0)
    elif init == 'fullft':
        p = f'{prev}/mb_full_ft.pt'
        if not os.path.exists(p):
            print(f'!! 找不到 {p}，回退官方权重'); return m
        m.load_state_dict(torch.load(p, map_location='cpu'), strict=True)
        print('[init] 载入上一轮全量微调权重 mb_full_ft.pt')
    return m


def save_ckpt(model, path, mode):
    sd = model.state_dict()
    if mode == 'lora':
        sd = {k: v for k, v in sd.items() if '.A' in k or '.B' in k}
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(sd, path)


def load_ckpt_weights(model, path, mode='full'):
    sd = torch.load(path, map_location='cpu')
    model.load_state_dict(sd, strict=(mode == 'full'))
    return model


# ------------------------------------------------------------------ 标定 pass
def cal_key(seq):
    """★2026-10-04 标定索引键。

    Harmony4D 是**多场景多机架**：每个场次的 "cam01" 是现场重新摆的另一台物理相机
    （且每条子序列各有自己的 COLMAP 世界系），所以「模型输出系 -> 相机系」的这个
    相似变换必须按 (场次, 视角) 索引。实测按视角名索引时逐场次需要的 R 散布
    13°~78°（中位 29°），单一全局 R 拟合不上 —— 这正是端到端误差 220mm、
    而逐帧最优对齐（PA）只有 38mm 的原因。
    kendo 只有一套固定机架，cal_key 退化成 seq['view']，行为逐位不变。
    """
    if str(seq.get('group', '')).startswith('h4d_'):
        return (seq['take'], seq['view'])
    return seq['view']


def calibrate(model, seqs, clip_len, batch=8):
    model.eval()
    store = {}
    with torch.no_grad():
        for seq in seqs:
            x = norm2d(seq['k2d'])
            if len(x) < clip_len:
                continue
            starts = np.linspace(0, len(x) - clip_len, min(4, len(x) - clip_len + 1)).astype(int)
            st = np.stack([x[s:s + clip_len] for s in starts])
            out = model(torch.from_numpy(st).to(DEV)).cpu().numpy()
            # ★2026-10-04 修 bug：out 是 (B,T,17,3)，原写法 out[:, 0:1, :] 切的是
            #   【时间轴】（等于每帧都减掉第0帧的整副骨架），应为关节轴。
            #   实测该 bug 让拟合出的全局尺度 s 只有真值的 0.55~0.75 倍，
            #   把所有 mm 指标整体放大（15 条 test 端到端 195.5 -> 31.6 mm）。
            out = out - out[:, :, 0:1, :]
            Gfull = gt_camera_mm(seq, get_cal(seq))
            G = np.stack([Gfull[s:s + clip_len] for s in starts])
            V = np.stack([seq['valid'][s:s + clip_len] for s in starts])
            store.setdefault(cal_key(seq), []).append((out[V], G[V]))
    for v, chunks in store.items():
        A_ = np.concatenate([a for a, b in chunks])
        B_ = np.concatenate([b for a, b in chunks])
        A = A_ - A_.mean(0)
        Bm = B_ - B_.mean(0)
        U, D, Vt = np.linalg.svd(Bm.T @ A / len(A))
        S = np.eye(3)
        S[2, 2] = np.sign(np.linalg.det(U @ Vt))
        R = U @ S @ Vt
        s = float(np.trace(np.diag(D) @ S) / max((A ** 2).sum(1).mean(), 1e-9))
        CAL[v] = (s, R)
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        print(f'  calib v{v}: s={s:.0f} mm/unit, R={ang:.1f}°')


CAL = {}

# ★2026-10-04 周期性重标定开关（epoch 数，0=关）。见 main 训练循环里的说明。
RECAL_EVERY = int(os.environ.get('KB_RECAL_EVERY', '0'))
_CAL_PASS_SEQS = []          # main() 里填；train() 周期性重标定要用


def gt_native(G, view):
    """相机系 mm -> MB 原生输出系（root 已 0）"""
    s, R = CAL[view]
    return (G.reshape(-1, 3) @ R.T / s).reshape(G.shape).astype(np.float32)


# ------------------------------------------------------------------ 增广
CHAINS = [[12, 13], [15, 16], [13], [16], [1, 2, 3], [4, 5, 6],
          [1, 2, 3, 4, 5, 6], [9, 10], [7, 8, 9, 10]]


def augment(x, rng):
    x = x.copy()
    if rng.random() < 0.5:
        ch = list(rng.choice(17, 5, replace=False)) if rng.random() < 0.3 \
            else CHAINS[int(rng.integers(len(CHAINS)))]
        for j in ch:
            x[:, j, :2] = 0
            x[:, j, 2] = 0
    return x


# ------------------------------------------------------------------ clip 索引
def split_takes(takes_all, excl, val_frac=0.12, test_frac=0.12, seed=1234, split_json=None):
    """按场次做 stratified(按 fps 分组) 的 train/val/test 三分。
    ★ 测试集从头到尾不参与训练，也不参与收敛判据 —— 只用来出最终成绩。
    返回 (train, val, test)，并把划分写进 split_json 以便复现。"""
    if split_json and os.path.exists(split_json):
        j = json.load(open(split_json))
        print(f'[split] 复用已有划分 {split_json}')
        return set(j['train']), set(j['val']), set(j['test'])

    groups = {}
    for tag, n, fps, grp in K.load_takes():
        if tag in excl:
            continue
        groups.setdefault(grp, []).append(tag)
    rnd = random.Random(seed)
    train, val, test = set(), set(), set()
    for grp, lst in sorted(groups.items()):
        lst = sorted(lst)
        rnd.shuffle(lst)
        nv = max(1, int(round(len(lst) * val_frac)))
        nt = max(1, int(round(len(lst) * test_frac)))
        test |= set(lst[:nt])
        val |= set(lst[nt:nt + nv])
        train |= set(lst[nt + nv:])
    if split_json:
        json.dump(dict(train=sorted(train), val=sorted(val), test=sorted(test),
                       seed=seed, val_frac=val_frac, test_frac=test_frac,
                       groups={k: sorted(v) for k, v in groups.items()}),
                  open(split_json, 'w'), ensure_ascii=False, indent=1)
        print(f'[split] 已写出 {split_json}')
    return train, val, test


def make_clips(seqs, clip_len, stride, take_w=None, min_ok=0.5):
    """返回 (clips, weights)。weights 来自质量报告，越干净的场次权重越高。
    min_ok: clip 内"可用帧"占比低于此值就整条 clip 丢掉（避免用大量废帧凑数）。"""
    idx, w = [], []
    n_skip = 0
    for i, seq in enumerate(seqs):
        n = len(seq['k2d'])
        if n < clip_len:
            continue
        okf = seq.get('ok')
        ww = float((take_w or {}).get(seq['take'], 1.0))
        for st in range(0, n - clip_len + 1, stride):
            if okf is not None and okf[st:st + clip_len].mean() < min_ok:
                n_skip += 1
                continue
            idx.append((i, st))
            w.append(ww)
    if n_skip:
        print(f'[clips] 因可用帧不足丢弃 {n_skip} 条 clip')
    return idx, w


# ------------------------------------------------------------------ 训练
def train(model, seqs, clips, epochs, lr, batch, cpe, bone_w, log=print,
          eval_every=0, val_seqs=None, clip_len=243, ckpt=None, clip_w=None, mode='full',
          patience=5, min_delta=1.0, max_hours=0.0, warmup=0.05):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=.01)
    steps_per_ep = max(1, (min(cpe, len(clips)) + batch - 1) // batch)
    n_warm = max(1, int(warmup * epochs * steps_per_ep))
    n_tot = max(n_warm + 1, epochs * steps_per_ep)

    def lr_at(step):
        if step < n_warm:
            return (step + 1) / n_warm
        prog = (step - n_warm) / max(1, n_tot - n_warm)
        return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))

    gstep = 0
    best, bad, best_ep = float('inf'), 0, 0
    t_start = time.time()
    rng = np.random.default_rng(0)
    prng = random.Random(0)
    # 每条序列预算好归一化 2D
    cache = {}
    bone_pairs = [(1, 2), (2, 3), (4, 5), (5, 6), (11, 12), (12, 13), (14, 15), (15, 16)]
    for ep in range(epochs):
        model.train()
        sel = prng.choices(clips, weights=clip_w, k=min(cpe, len(clips))) if clip_w \
            else prng.sample(clips, min(cpe, len(clips)))
        tot, nb = 0.0, 0
        t0 = time.time()
        for b0 in range(0, len(sel), batch):
            xs, gs, ms = [], [], []
            for (si, st) in sel[b0:b0 + batch]:
                seq = seqs[si]
                if si not in cache:
                    cache[si] = norm2d(seq['k2d'])
                x = cache[si][st:st + clip_len]
                x = augment(x, rng)
                G = gt_camera_mm(seq, get_cal(seq))[st:st + clip_len]
                gt = gt_native(G, cal_key(seq))
                # 只有「标签可信 + 该关节 2D 有信号 + 该帧通过质量筛选」的关节进 loss
                m = seq['valid'][st:st + clip_len] & (x[:, :, 2] > 0.05)
                m &= seq['ok'][st:st + clip_len][:, None]
                xs.append(x); gs.append(gt); ms.append(m)
            X = torch.from_numpy(np.stack(xs)).to(DEV)
            GT = torch.from_numpy(np.stack(gs)).to(DEV)
            M = torch.from_numpy(np.stack(ms)).float().to(DEV)
            out = model(X)
            e = torch.sqrt(((out - GT) ** 2).sum(-1) + 1e-8)
            loss = (e * M).sum() / M.sum().clamp(min=1)
            if bone_w > 0:
                # 骨长一致性：预测骨长 vs 标签骨长（逐帧）。治"四肢被拉长/压扁"
                bl = 0.0
                for a, b in bone_pairs:
                    la = torch.linalg.norm(GT[:, a] - GT[:, b], dim=-1)
                    lo = torch.linalg.norm(out[:, a] - out[:, b], dim=-1)
                    mb = M[:, a] * M[:, b]                 # M 已是 0/1 float
                    bl = bl + ((la - lo).abs() * mb).sum() / mb.sum().clamp(min=1)
                loss = loss + bone_w * bl / len(bone_pairs)
            for g in opt.param_groups:
                g['lr'] = lr * lr_at(gstep)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            gstep += 1
            tot += float(loss); nb += 1
        # ★2026-10-04 周期性重标定：目标里的 (s,R) 是训练前用初始模型拟合一次就冻住的，
        #   而模型输出尺度在训练中会漂（实测 s: 98 -> 279）-> 目标与模型自洽性被破坏，
        #   典型症状 = loss 单调降、验证集不动甚至变差。用当前模型重标定一次即可复位。
        if RECAL_EVERY and (ep + 1) % RECAL_EVERY == 0:
            model.eval()
            calibrate(model, _CAL_PASS_SEQS, clip_len)
            model.train()
            log(f'  [recal] ep{ep+1} 已用当前模型重标定', flush=True)
        cur_lr = opt.param_groups[0]['lr']
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            log(f'  [ep {ep+1}/{epochs}] loss={tot/max(nb,1):.4f}u lr={cur_lr:.2e}  '
                f'{time.time()-t0:.1f}s/epoch  {(time.time()-t_start)/3600:.2f}h', flush=True)
        if eval_every and val_seqs and (ep + 1) % eval_every == 0:
            r = evaluate(model, val_seqs, clip_len, max_clips=8, log=log)
            cur = r['med_mm']                      # 统一口径后的主指标
            if cur < best - min_delta:
                best, bad, best_ep = cur, 0, ep + 1
                if ckpt:
                    save_ckpt(model, ckpt, mode)
                log(f'  ★ 新最优 val_median={cur:.1f}mm (ep {ep+1}) 已存档', flush=True)
            else:
                bad += 1
                log(f'  无改善 {bad}/{patience}  cur={cur:.1f}mm  best={best:.1f}mm@ep{best_ep}', flush=True)
                if bad >= patience:
                    log(f'== 收敛：连续 {patience} 次评测无改善，提前停止于 ep {ep+1}（最优 ep {best_ep}, {best:.1f}mm）==', flush=True)
                    break
            model.train()
        if max_hours > 0 and (time.time() - t_start) / 3600 > max_hours:
            log(f'== 到时间上限 {max_hours}h，停止于 ep {ep+1}（最优 {best:.1f}mm@ep{best_ep}）==', flush=True)
            break
    return model


# ------------------------------------------------------------------ 评测
SKEL = K.SKEL_H36M


def evaluate(model, seqs, clip_len=243, max_clips=40, log=print):
    """返回 mm 口径指标（乘回 s 还原到相机系 mm）。

    ★ 口径统一：所有指标都是**把 (帧, 关节) 全部摊平后取 pooled 中位**，
      所以 overall = clean ∪ occ，三行可以直接对比。
      （旧版是"中位的中位 / 均值的均值"混用，55 / 85 / 312 并列会误导人。）
    另外保留 mean 作为尾部诊断：mean 远大于 median 说明误差右尾重。
    """
    model.eval()
    # (帧,关节) 摊平的误差池
    pool_all, pool_clean, pool_occ, pool_wr = [], [], [], []
    n_obs = 0
    with torch.no_grad():
        for seq in seqs:
            n = len(seq['k2d'])
            if n < clip_len:
                continue
            x = norm2d(seq['k2d'])
            G = gt_camera_mm(seq, get_cal(seq))
            starts = np.linspace(0, n - clip_len, min(max_clips, n - clip_len + 1)).astype(int)
            s = CAL[cal_key(seq)][0]
            for b0 in range(0, len(starts), 8):
                ss = starts[b0:b0 + 8]
                X = torch.from_numpy(np.stack([x[t:t + clip_len] for t in ss])).to(DEV)
                out = model(X).cpu().numpy() * s              # -> 相机系 mm, root 已 0
                for k, t in enumerate(ss):
                    gt = G[t:t + clip_len]
                    e = np.linalg.norm(out[k] - gt, axis=2)    # (T,17)
                    conf2d = seq['k2d'][t:t + clip_len, :, 2]
                    val = seq['valid'][t:t + clip_len]
                    ok = val & (conf2d > 0.05)                 # 标签有证据 且 2D 有信号
                    occ = val & (conf2d <= 0.05)               # 标签有证据 但 2D 被遮
                    if ok.sum():
                        pool_all.append(e[ok]); pool_clean.append(e[ok])
                    if occ.sum():
                        pool_all.append(e[occ]); pool_occ.append(e[occ])
                    for j in (13, 16):
                        mj = ok & (np.arange(17)[None, :] == j)
                        if mj.sum():
                            pool_wr.append(e[mj])
                    n_obs += int(ok.sum()) + int(occ.sum())

    def stat(pool):
        if not pool:
            return float('nan'), float('nan'), 0
        a = np.concatenate(pool)
        return float(np.median(a)), float(np.mean(a)), int(a.size)

    m_all, mean_all, n_all = stat(pool_all)
    m_cl, mean_cl, n_cl = stat(pool_clean)
    m_oc, mean_oc, n_oc = stat(pool_occ)
    m_wr, mean_wr, n_wr = stat(pool_wr)
    res = dict(
        # 主指标：pooled 中位（mm），三行同口径、可直接比
        med_mm=m_all, clean_mm=m_cl, occ_mm=m_oc, wrist_mm=m_wr,
        # 尾部诊断
        mean_mm=mean_all, clean_mean_mm=mean_cl, occ_mean_mm=mean_oc, wrist_mean_mm=mean_wr,
        # 样本量（关节观测数）
        n_joint_obs=n_all, n_clean=n_cl, n_occ=n_oc, n_wrist=n_wr,
    )
    log('  eval: %s' % json.dumps({k: (round(v, 1) if isinstance(v, float) else v)
                                   for k, v in res.items()}, ensure_ascii=False))
    return res


CAL_OBJ = None
# ★2026-10-03 增量：按 group 惰性加载标定。kendo 的 seq['group'] 是 '960'/'1920'（与原行为一致），
#   Harmony4D 是 'h4d_<tag>'（各自有 calib_gt_<tag>）。
_CAL_CACHE = {}


def get_cal(seq):
    g = seq['group']
    if g not in _CAL_CACHE:
        _CAL_CACHE[g] = K.Calib(K.CALIB[g])
    return _CAL_CACHE[g]


def main():
    global CAL_OBJ
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=K.OUT)
    ap.add_argument('--mode', choices=['full', 'lora'], default='full',
                    help='full=全量微调(本轮)；lora=上一轮的参数高效路线')
    ap.add_argument('--init', choices=['official', 'lora5', 'lora4', 'fullft'], default='lora5',
                    help='热启动点：lora5=官方+上一轮LoRA-v5增量(默认,最优)；official=纯官方')
    ap.add_argument('--init-ckpt', default=None,
                    help='从任意全量权重(.pt)热启动（注意：若该权重训练时见过本轮的 val/test 场次 '
                         '会造成泄漏，除非划分完全相同）')
    ap.add_argument('--patience', type=int, default=5,
                    help='连续 N 次评测验证集无改善就停（收敛即停）')
    ap.add_argument('--min-delta', type=float, default=1.0, help='改善阈值(mm)')
    ap.add_argument('--max-hours', type=float, default=0.0, help='时间上限(小时)，0=不限')
    ap.add_argument('--epochs', type=int, default=200,
                    help='epoch 上限（真正停止由 --patience 收敛判据决定）')
    ap.add_argument('--lr', type=float, default=5e-5,
                    help='全量微调建议 2e-5~5e-5；LoRA 用 1e-4')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--clip-len', type=int, default=243)
    ap.add_argument('--stride', type=int, default=81)
    ap.add_argument('--clips-per-epoch', type=int, default=2680)
    ap.add_argument('--bone-w', type=float, default=0.1,
                    help='骨长一致性损失权重（0=关闭）')
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--val-frac', type=float, default=0.12)
    ap.add_argument('--test-frac', type=float, default=0.12,
                    help='独立测试集占比（按场次分层划分；全程不参与训练与收敛判据）')
    ap.add_argument('--split-json', default=f'{K.ROOT}/mb/data/_split.json',
                    help='划分落盘，保证可复现；已存在则直接复用')
    ap.add_argument('--no-split-reuse', action='store_true', help='忽略已有划分、重新划')
    ap.add_argument('--val-takes', nargs='*', default=None)
    ap.add_argument('--no-quality', action='store_true',
                    help='不做任何质量筛选（默认：报告级分箱 + 逐帧重投影残差 两级筛选）')
    ap.add_argument('--qa-dir', default=f'{K.ROOT}/mb/data_qa',
                    help='kb_frameqa.py 产出的逐帧质量目录')
    ap.add_argument('--qa-thr', type=float, default=10.0,
                    help='逐帧 3D->2D 重投影残差上限(归一化到 960 域, px)。默认 10 (~保留 82%%)')
    ap.add_argument('--qa-min-n', type=int, default=4,
                    help='该帧参与比对的关节数下限（太少无法判定 -> 弃用）')
    ap.add_argument('--min-clip-ok', type=float, default=0.5,
                    help='clip 内可用帧占比低于此值就丢掉该 clip')
    ap.add_argument('--q-cv', type=float, default=0.18, help='骨长CV 上限，超过就排除该场次')
    ap.add_argument('--q-problems', type=int, default=3, help='问题箱上限，超过就排除该场次')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--eval-every', type=int, default=5,
                    help='每 N epoch 在验证集上评一次并存最优档（收敛判据也用它）')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=f'{K.ROOT}/mb/ckpt/kb_full_v1.pt')
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    # ★2026-10-03：takes 改为从数据目录推导 —— 原 K.load_takes() 依赖 kendo 的 _segqa_summary.txt
    _takes_from_data = sorted({os.path.basename(fn)[:-4].rsplit('_', 2)[0]
                               for fn in glob.glob(f'{a.data}/*.npz')})
    takes_all = _takes_from_data
    excl = set(load_exclude(a.data))

    # ---- 质量筛选（基于 _segqa 报告）----
    tw = {}
    if not a.no_quality:
        tw, qbad = K.take_weights(a.q_cv, a.q_problems)
        print(f'[quality] 质量合格场次 {len(tw)}；因骨长CV>{a.q_cv} 或 问题箱>{a.q_problems} 排除 {len(qbad)} 场: {sorted(qbad)}')
        excl |= qbad

    test_takes = set()
    if a.smoke:
        all_t = sorted(set(takes_all))
        val_takes = set(all_t[:1])
        train_takes = set(all_t[1:3])
    elif a.val_takes:
        val_takes = set(a.val_takes)
        train_takes = set(takes_all) - val_takes - excl
    else:
        train_takes, val_takes, test_takes = split_takes(
            takes_all, excl, a.val_frac, a.test_frac, a.seed,
            None if a.no_split_reuse else a.split_json)
    print(f'训练场次 {len(train_takes)} / 验证场次 {len(val_takes)} / 测试场次 {len(test_takes)}')
    print(f'  验证集: {sorted(val_takes)}')
    if test_takes:
        print(f'  测试集(全程不参与训练, 收敛判据也不用): {sorted(test_takes)}')

    qual = None if a.no_quality else load_quality(a.data)
    qa_dir = None if a.no_quality else a.qa_dir
    tr = load_all(a.data, train_takes, exclude=excl, qa_dir=qa_dir,
                  qa_thr=a.qa_thr, qa_min_n=a.qa_min_n, quality=qual)
    va = load_all(a.data, val_takes, exclude=excl)     # 验证集不做筛选，指标才是诚实的
    n_tr = sum(len(s["k2d"]) for s in tr)
    n_ok = sum(int(s["ok"].sum()) for s in tr)
    print(f'序列 train={len(tr)} val={len(va)}')
    print(f'训练帧: 总 {n_tr}，通过质量筛选可用 {n_ok} ({100*n_ok/max(1,n_tr):.1f}%)  —— 剔除 {n_tr-n_ok} 帧')
    print(f'验证帧: {sum(len(s["k2d"]) for s in va)}（不筛选）')
    if not tr:
        print('没有训练数据，先跑 kb_build.py'); return

    global CAL_OBJ
    # ★2026-10-03：标定改为按 group 惰性加载（见 get_cal），这里不再写死 kendo 的 '960'

    print(f'== 热启动点: {a.init} {a.init_ckpt or ""} ==')
    base = build_init(a.init, a.init_ckpt).to(DEV)
    print('== 标定 pass ==')
    # ★2026-10-04 H4D：按 (take,view) 分组标定 -> 验证场次也要有自己的键，
    #   否则 evaluate 取不到（kendo 仍只用 tr，行为不变）。
    global _CAL_PASS_SEQS
    _cal_pass_seqs = tr
    if tr and str(tr[0].get('group', '')).startswith('h4d_'):
        _cal_pass_seqs = tr + va
        print(f'  [h4d] 标定场次 = 训练 {len(tr)} + 验证 {len(va)}（按 (take,view) 各自成组）')
    _CAL_PASS_SEQS = _cal_pass_seqs
    calibrate(base, _cal_pass_seqs, a.clip_len)
    # ★2026-10-03 实验B：允许强制覆盖标定。KB_FIX_S=162.7 => CAL=(162.7, I)
    #   依据：逐帧最优 s 实测 140~164，与 kendo 的 162.7 一致 => 输出尺度本来就对，
    #   坏的只是朝向；而全局拟合会把正确的 s 一起带崩。
    if os.environ.get('KB_FIX_S'):
        _sf = float(os.environ['KB_FIX_S'])
        _r0 = os.environ.get('KB_FIX_ROT', '')
        _R = np.eye(3) if _r0 == '' else CAL[list(CAL.keys())[0]][1]
        for _k in list(CAL.keys()):
            CAL[_k] = (_sf, _R if _r0 == '' else CAL[_k][1])
        print(f'[KB_FIX_S] 标定覆盖: s={_sf}  R={"I" if _r0 == "" else "拟合值"}  ', flush=True)
    # ★ 标定一算完就落盘：否则训练被中途 kill 时，<out>_eval.json 永远写不出来，
    #    后续想单独复评就没标定可用了（v2 就中了这个坑）
    try:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({'calib': {str(v): [CAL[v][0], CAL[v][1].tolist()] for v in CAL},
                   'mode': a.mode, 'split_json': a.split_json},
                  open(a.out.replace('.pt', '_calib.json'), 'w'), ensure_ascii=False, indent=1)
        print('  [calib] 已落盘 ->', a.out.replace('.pt', '_calib.json'))
    except Exception as e:
        print('  [calib] 落盘失败:', e)

    # ★2026-10-04 可选：把标定的 R 强制成单位阵（只保留拟合出的 s）。
    #   让训练目标变成"纯相机系姿态"，逼模型自己学出相机朝向约定。
    if os.environ.get('KB_FORCE_R_I'):
        import numpy as _np
        for _k in list(CAL.keys()):
            CAL[_k] = (CAL[_k][0], _np.eye(3))
        print(f'[KB_FORCE_R_I] 已把 {len(CAL)} 个标定的 R 置为单位阵', flush=True)

    if a.mode == 'lora':
        model = to_lora(build_init(a.init, a.init_ckpt).to(DEV), a.lora_r).to(DEV)
    else:
        model = build_init(a.init, a.init_ckpt).to(DEV)        # 全量微调：所有参数可训
        del base
        # ★2026-10-04 可选：冻结骨干，只训尾部 K 个 block + norm/pre_logits/head。
        #   零样本 PA 形状 38.1mm 已优于微调后的 50.0mm -> 形状本来是好的，
        #   只有输出约定要改。必须放开至少一个 attention block（见文件头说明）。
        _tail = int(os.environ.get('KB_UNFREEZE_TAIL', '0'))
        if _tail > 0:
            for _p in model.parameters():
                _p.requires_grad_(False)
            _nb = len(model.blocks_st)
            _keep = []
            for _i in range(max(0, _nb - _tail), _nb):
                _keep += list(model.blocks_st[_i].parameters())
                _keep += list(model.blocks_ts[_i].parameters())
                _keep += list(model.ts_attn[_i].parameters())
            _keep += list(model.norm.parameters())
            _keep += list(model.pre_logits.parameters())
            _keep += list(model.head.parameters())
            for _p in _keep:
                _p.requires_grad_(True)
            print(f'[tail] 冻结骨干，只训最后 {_tail}/{_nb} 个 block + norm/pre_logits/head')
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'全量微调，可训参数 {n_train/1e6:.2f}M (共 {sum(p.numel() for p in model.parameters())/1e6:.2f}M)')

    clips, clip_w = make_clips(tr, a.clip_len, a.stride, tw, min_ok=a.min_clip_ok)
    print(f'训练 clip 数 {len(clips)}  (质量加权: {"开" if tw else "关"})')

    if a.epochs <= 0:
        print('== 只评测基线（epochs=0，不训练）==')
        res = evaluate(model, va, a.clip_len)
        save_ckpt(model, a.out, a.mode)
    else:
        print(f'== 训练 ({a.mode}, 热启动={a.init}, 收敛即停 patience={a.patience}) ==')
        train(model, tr, clips, a.epochs, a.lr, a.batch, a.clips_per_epoch,
              a.bone_w, log=lambda s, **kw: print(s, flush=True),
              eval_every=a.eval_every, val_seqs=va, clip_len=a.clip_len,
              ckpt=a.out, clip_w=clip_w, mode=a.mode,
              patience=a.patience, min_delta=a.min_delta, max_hours=a.max_hours)
        # 训练中已按"最优"存档；这里再评一次仅作记录（best 权重已在 ckpt）
        if a.eval_every > 0 and os.path.exists(a.out):
            load_ckpt_weights(model, a.out, a.mode)
            print('[final] 已回载训练期最优权重')
        else:
            save_ckpt(model, a.out, a.mode)
        print('== 最终评测(验证集) ==')
        res = evaluate(model, va, a.clip_len)

    # ★ 测试集只在最后一刻评一次，全程不参与任何选择
    res_test = None
    if test_takes:
        te = load_all(a.data, test_takes, exclude=excl)        # 测试集同样不筛选
        print(f'== 测试集评测（{len(test_takes)} 场 / {sum(len(s["k2d"]) for s in te)} 帧，全程未参与训练）==')
        res_test = evaluate(model, te, a.clip_len)
    json.dump(dict(res=res, res_test=res_test,
                   calib={str(v): [CAL[v][0], CAL[v][1].tolist()] for v in CAL},
                   args=vars(a), val_takes=sorted(val_takes),
                   test_takes=sorted(test_takes), mode=a.mode),
              open(a.out.replace('.pt', '_eval.json'), 'w'), ensure_ascii=False, indent=1)
    print('权重 ->', a.out)
    print('指标 ->', a.out.replace('.pt', '_eval.json'))


if __name__ == '__main__':
    main()

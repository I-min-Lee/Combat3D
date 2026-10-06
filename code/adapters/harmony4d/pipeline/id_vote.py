#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  ★★★  SCENE-SPECIFIC LAYER — DO NOT REUSE AS-IS  ★★★
#
#  这是 Combat3D-Label 五层里【两个场景相关层】之一（检测框层 / 身份层）。
#  我们的测量证明：这两层换场景必须【换算法】，不是调参数 ——
#
#        changing scene  =>  changing algorithm
#
#  实测：Harmony4D 上我们用数据集自带的框与身份；而在这套代码的第二个域
#  （我们自己采集的盔甲对抗棍棒格斗）上，同一层必须从零重写
#  （RT-DETR + 位置先验；以及基于颜色证据的身份定标）。
#  两份实现都在本仓库里 —— 见 docs/SECOND_DOMAIN.md 的逐层对照表。
#
#  通用的是【三角化层及以下】，它们原样复用。层契约见 code/adapters/README.md。
#
#  ★ 连数据格式都是场景相关的：这个文件读的框/身份格式来自特定数据集，
#    换场景时要连 I/O 一起改，不只是改阈值。
#
#  Reusable? This layer: NO — rewrite it. Triangulation & below: YES.
# =============================================================================
"""id_vote.py — ★**用多视角 3D 一致性定编号**（2026-09-30 新增，治 06/08 的根因）

## 为什么必须这么做（而不是任何单视角的时序方法）
实测 06_sword3：**每个视角各自**在长段（~100~160 帧）里翻编号，且**各视角的段边界对不上**
（view01: A:161 … B:11 A:134 … B:43 B:126；view07: A:160 B:166 … B:80 B:116）。

→ 单视角的时间连续性/速度预测**原理上无法确定绝对身份**：一个视角只有自己的轨迹，
  没有任何东西能告诉它"我这一段的 0 号其实是另一个人"。
  （这就是为什么 `pid_stabilize.py` 在这类场次上 0 次换号 —— 它没坏，是问题不在那儿。）

→ 身份在多相机系统里**只能通过跨视角一致性观测到**：6 个视角各持两个人的 2D，
  正确的编号组合是那个能让 6 路射线在 3D 上自洽的组合。

## 做法
逐帧：枚举 2^n 种"逐视角翻/不翻"（n=视角数，6 视角 → 64 种），
每种组合三角化 13 个身体关节，**打分 = 跨视角重投影残差中位**，取最小者。
再用 **Viterbi** 在帧间做时序平滑（状态=翻转组合，转移代价 = 每换一个视角的罚分），
避免逐帧抖动造成 3D 跳变。

★**规范(gauge)**：所有视角同时翻转在 3D 上完全等价 → 投票只能定到"模一个全局置换"。
  这里把**第一个可用视角钉成"不翻"**来定规范。剩下的那一个整段全局置换是**不可观测的**，
  由 `h4d_metrics.py --match-perm` 吸收（真值框版的身份是数据集白给的，我们不假装知道）。

## 实测（06_sword3，与真值框标注做关键点匹配对照）
定规范后逐帧一致率 **96.8%（213/220）**；出错的帧残差明显更高（35/32/27px vs 正常 10~20px）
= 两人贴身、本来就不判的帧，且下一帧就恢复。

## 用法
```bash
python id_vote.py --calib $B/calib_gt_06_sword3 --annots $B/asm_self_06_sword3/annots \
    --out $B/asm_self_06_sword3_vote/annots --views 01,03,04,07,09,14 --start 1 --end 601 \
    [--smooth 6] [--min-views 3]
```
输出只**置换 personID**，关键点一字不改。之后照常跑 tri_h4d.py。
"""
import os, sys, json, glob, argparse, itertools
import numpy as np
import cv2

JOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]     # COCO17 的 13 身体点
CONF = 0.3


def load_fisheye_flags(intri_path):
    """读 `fisheye_<v>: 1`（read_camera 不读自定义键）。Harmony4D 有、Panoptic 没有（针孔）。"""
    import re
    flags = {}
    if os.path.exists(intri_path):
        for ln in open(intri_path):
            m = re.match(r'\s*fisheye_(\S+)\s*:\s*1', ln)
            if m:
                flags[m.group(1)] = True
    return flags


def load_cams(calib, views):
    sys.path.insert(0, os.environ.get('EMCORE', '/workshop/Lym/combat3d/port/emcore'))
    from easymocap.mytools.camera_utils import read_camera
    intri = os.path.join(calib, 'intri.yml')
    C = read_camera(intri, os.path.join(calib, 'extri.yml'))
    FISH = load_fisheye_flags(intri)
    cam = {}
    for v in views:
        c = C[v]
        K = np.asarray(c['K'], float).reshape(3, 3); K[2, 2] = 1.0
        cam[v] = dict(K=K, D=np.asarray(c['dist'], float).reshape(-1),   # ★保留全部系数
                      fisheye=bool(FISH.get(v, False)),
                      P=K @ np.hstack([np.asarray(c['R'], float).reshape(3, 3),
                                       np.asarray(c['T'], float).reshape(3, 1)]))
    return cam


def make_dlt(cam):
    def und(uv, v):
        """去畸变：★按标定里有没有 `fisheye_` 键自动选模型（与 tri_h4d.py 同一套约定）。
        Harmony4D=鱼眼 4 参数；Panoptic=针孔 5 参数。"""
        uv = np.asarray(uv, float).reshape(-1, 1, 2)
        c = cam[v]
        if c['fisheye']:
            return cv2.fisheye.undistortPoints(uv, c['K'], c['D'][:4].reshape(4, 1),
                                               None, None, c['K']).reshape(-1, 2)
        return cv2.undistortPoints(uv, c['K'], c['D'].reshape(-1, 1),
                                   None, None, c['K']).reshape(-1, 2)

    def dlt(pts, views):
        A = []
        for uv, v in zip(pts, views):
            P = cam[v]['P']
            A.append(uv[0] * P[2] - P[0]); A.append(uv[1] * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.stack(A))
        X = Vt[-1]
        return None if abs(X[3]) < 1e-12 else X[:3] / X[3]

    def reproj(X, v):
        q = cam[v]['P'] @ np.append(np.asarray(X, float), 1.0)
        return None if abs(q[2]) < 1e-9 else np.array([q[0] / q[2], q[1] / q[2]])
    return und, dlt, reproj


def frame_cost(kp2d, views, und, dlt, reproj, flip):
    """一个翻转组合的代价（px）。

    ★★ 打分口径是**这条判据的生死的全部**（2026-09-30 踩过）：
      最初写成"对所有 关节×视角 取中位" —— 结果**翻了 1 个视角只污染 13/78 个数**，
      中位纹丝不动 → 判据对本该判的东西视而不见 → 在**真值框版**（身份本来就是对的）
      输入上它也照样"翻 2.6 个视角/帧"，等于纯加噪。
      现改为**逐关节先取"跨视角最大残差"，再对关节取中位**：
      某个视角贴错人 → 那 13 个关节的 max 全被顶到"两人的像素间距"量级 → 判据立刻分辨。
    """
    per_joint = []
    for j in JOINTS:
        pts, vs, obs = [], [], []
        for v in views:
            a = kp2d[v][flip[v]]
            if a is not None and a[j, 2] > CONF:
                pts.append(und(a[j, :2], v)[0]); vs.append(v); obs.append(a[j, :2])
        if len(vs) < 3:                      # 少于 3 视角三角化不出可信解
            continue
        X = dlt(pts, vs)
        if X is None:
            continue
        e = []
        for v, o in zip(vs, obs):
            q = reproj(X, v)
            if q is not None:
                e.append(float(np.linalg.norm(q - o)))
        if e:
            per_joint.append(max(e))         # ★关键：跨视角取最大，才对"单视角贴错"敏感
    return float(np.median(per_joint)) if len(per_joint) >= 6 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', required=True)
    ap.add_argument('--annots', required=True)
    ap.add_argument('--out', required=True, help='输出 annots 目录（会被创建）')
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--smooth', type=float, default=6.0,
                    help='★Viterbi 转移罚分(px)：每多翻一个视角的代价。0 = 逐帧独立（会抖）')
    ap.add_argument('--min-views', type=int, default=3, help='少于这么多视角可判就跳过该帧')
    ap.add_argument('--flip-bias', type=float, default=1.0,
                    help='★"证据不足就别翻"偏置（无量纲）：翻 k 个视角要多付 k×该值×该帧最优打分。'
                         '0=关（会在本来正确的序列上乱翻）。真实该翻的帧分差达数百~数千 px，'
                         '而这个偏置只有几十 px，所以既能压住噪声、又不挡真修。')
    ap.add_argument('--root-switch', type=float, default=0.3,
                    help='★第二层（根轨迹 Viterbi）换规范的罚分，**无量纲**：'
                         '实际罚分 = 该值 × 两规范根间距的中位数 D（自动定标，跨数据集通用）')
    ap.add_argument('--anchor', default='none',
                    help="★规范锚点。'none'(默认)=放开全部 2^n 状态，由转移罚分保证规范整段为常数；"
                         "给某个视角名=把它钉成不翻（**不建议**：该视角自己漂的话整个系统会跟着漂）")
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    views = [v.zfill(2) for v in a.views.split(',')]
    cam = load_cams(a.calib, views)
    und, dlt, reproj = make_dlt(cam)

    files = sorted(glob.glob(os.path.join(a.annots, views[0], '*.json')))
    frames = []
    for f in files:
        fr = int(os.path.basename(f).split('.')[0])
        if a.start <= fr <= a.end:
            frames.append(fr)
    print('[id_vote] %d 帧 · %d 视角 · smooth=%.1f' % (len(frames), len(views), a.smooth), flush=True)

    # ---- Viterbi 状态空间 ----
    # ★**不要**把一个视角钉死当规范：那个视角自己也在漂（实测 view01 整段翻了好几次），
    #   钉死它会让整个系统跟着它漂 → 3D 身份随时间切换。
    #   放开全部 2^n 个状态，让"转移罚分"去保证**规范在整段上是常数**（换一次规范要翻 n 个视角，代价极高）。
    #   剩下的那个"全局谁当 0 号"是几何上不可观测的（两人互换后残差一样），由 --match-perm 吸收。
    if a.anchor and a.anchor in views:
        free = [v for v in views if v != a.anchor]
        fixed = {a.anchor: 0}
    else:
        free, fixed = list(views), {}
    STATES = list(itertools.product((0, 1), repeat=len(free)))
    IDX = {s: i for i, s in enumerate(STATES)}
    HAM = np.zeros((len(STATES), len(STATES)))
    for i, si in enumerate(STATES):
        for j, sj in enumerate(STATES):
            HAM[i, j] = sum(x != y for x, y in zip(si, sj))
    INF = 1e9

    # ---- 逐帧打分 ----
    per_frame = []            # [(fr, {view: (kp0,kp1)}, emission(list over STATES))]
    for fr in frames:
        kp2d = {}
        for v in views:
            f = os.path.join(a.annots, v, '%06d.json' % fr)
            if not os.path.exists(f):
                continue
            d = {}
            for an in json.load(open(f)).get('annots', []):
                d[int(an['personID'])] = np.array(an['keypoints'], float)
            if 0 in d and 1 in d:
                kp2d[v] = (d[0], d[1])
        uv = [v for v in views if v in kp2d]
        if len(uv) < a.min_views:
            per_frame.append((fr, kp2d, None)); continue
        em = np.full(len(STATES), INF)
        for si, st in enumerate(STATES):
            flip = dict(fixed)
            for v, b in zip(free, st):
                flip[v] = b
            c = frame_cost(kp2d, uv, und, dlt, reproj, flip)
            if c is not None:
                em[si] = c
        # 某帧没有可用状态 → 用"全不翻"兜底
        if np.all(em >= INF):
            em[IDX[tuple([0] * len(free))]] = 0.0
        # ★★"证据不足就别翻"偏置：给每个"翻了 k 个视角"的状态加 k × bias 的代价。
        #   判据在两套等价/近等价配置之间的分差只有十几~几十 px（纯噪声量级），
        #   不加偏置的话，**本来正确的序列也会被翻坏**（实测 01_hugging 的 --pa seq 36.8 → 452.2）。
        #   而真正该翻的帧，分差是几百~几千 px（见 dbg_score：单翻一个视角 Δ=+160~+2100px），
        #   所以用"该帧最优打分"当单位自动定标即可跨分辨率/跨数据集使用。
        m = float(em[em < INF].min()) if np.any(em < INF) else None
        if m is not None:
            for si, st in enumerate(STATES):
                if em[si] < INF:
                    em[si] += a.flip_bias * m * sum(st)
        per_frame.append((fr, kp2d, em))

    # ---- DP ----
    valid = [i for i, (_, _, em) in enumerate(per_frame) if em is not None]
    if not valid:
        print('没有可用帧'); return
    dp = per_frame[valid[0]][2].copy()
    bps = []
    for t in range(1, len(valid)):
        em = per_frame[valid[t]][2]
        tot = dp[:, None] + a.smooth * HAM          # (S,S)
        bp = np.argmin(tot, axis=0)
        dp = em + tot[bp, np.arange(len(STATES))]
        bps.append(bp)
    # 回溯
    path = {}
    s = int(np.argmin(dp))
    path[per_frame[valid[-1]][0]] = STATES[s]
    for k in range(len(valid) - 2, -1, -1):
        s = int(bps[k][s])
        path[per_frame[valid[k]][0]] = STATES[s]

    # ---- 写盘：只改 personID ----
    if not a.dry_run:
        for v in views:
            os.makedirs(os.path.join(a.out, v), exist_ok=True)
    # ★★ 规范规范化（gauge canonicalization）—— 这一步是整条判据能不能用的关键：
    #   "所有视角同时翻" 在几何上**完全等价**（只是把 pid0/pid1 整体换个名字），
    #   所以状态空间里 `000000` 与 `111111` 是同一个解的两种写法。
    #   判据两边的差异**只来自置信度掩码**（person0 与 person1 的逐关节置信度不同）= 纯噪声，
    #   于是 DP 会在两种写法之间来回跳 → 输出的"谁叫 0 号"随时间乱变。
    #   而 `--match-perm` 只能吸收**一个**全局置换，段内翻来覆去是修不回来的。
    #   修法：每帧把翻转向量**规范化到较轻的那个等价写法**（权重 ≤ n/2）。
    #   这是纯改名的操作，**几何完全不动**，但让"谁叫 0 号"在整段上保持连续。
    n_half = len(views) / 2.0
    n_canon = 0
    for fr, kp2d, em in per_frame:
        if em is None or fr not in path:
            continue
        st = path[fr]
        if sum(st) > n_half:
            path[fr] = tuple(0 if b else 1 for b in st)
            n_canon += 1
    if n_canon:
        print('    [gauge] 规范化了 %d 帧（整体换名，几何不变）' % n_canon, flush=True)

    # ---- ★★ 第二层：在 **3D 根轨迹** 上定"每帧的规范"，修**整段身份漂移** ----
    #   投票只修"视角之间的相对分歧"。规范规范化把输出绑在多数视角的原始标签上，
    #   而那些标签本身会漂 → 每段"谁叫 0 号"仍会变，表现为根轨迹上 ~1m 的段级跳变。
    #   但**投票之后 6 视角已经一致**，所以这个跳变是干净的、可检测的：
    #   规范 g 是纯改名（pid0↔pid1），对每帧算两个规范下的根，再用 2 状态 Viterbi
    #   选"根轨迹最连续（加速度最小）"的那条即可。这一层以前做不了，现在可以了。
    def root_of(kp2d, uv, flip, g):
        f = {v: (flip[v] ^ g) for v in views}
        pts, vs = [], []
        for v in uv:
            a = kp2d[v][f[v]]
            if a[11, 2] > CONF and a[12, 2] > CONF:
                pts.append(und(0.5 * (a[11, :2] + a[12, :2]), v)[0]); vs.append(v)
        if len(vs) < 3:
            return None
        return dlt(pts, vs)

    seq = []
    for fr, kp2d, em in per_frame:
        if em is None or fr not in path:
            continue
        uv = [v for v in views if v in kp2d]
        flip = dict(fixed)
        for v, b in zip(free, path[fr]):
            flip[v] = b
        seq.append((fr, root_of(kp2d, uv, flip, 0), root_of(kp2d, uv, flip, 1)))
    # ★罚分必须**按数据自身的尺度定标**：换一次规范 = pid0 从一个人跳到另一个人，
    #   跳变量级就是"两人的根间距"。所以用它的中位数 D 当单位，root_switch 是无量纲比例。
    #   （踩过：最早写成固定 150.0 世界单位 —— 而 06 世界系 1 单位≈1 米、两人只隔 2 米，
    #     等于罚 150 米 → Viterbi 永远不换 → 报"切换 0 次"，其实是被罚分锁死了。）
    _dd = [float(np.linalg.norm(r0 - r1)) for _f, r0, r1 in seq
           if r0 is not None and r1 is not None]
    D = float(np.median(_dd)) if _dd else 1.0
    pen_sw = a.root_switch * D
    print('    [根轨迹] 两规范根间距中位 D=%.3f -> 换规范罚分 %.3f（比例 %.2f）'
          % (D, pen_sw, a.root_switch), flush=True)
    DP = np.zeros((len(seq), 2))
    BP = np.zeros((len(seq), 2), int)
    for i in range(1, len(seq)):
        for g in (0, 1):
            best, bj = None, 0
            for gp in (0, 1):
                # 加速度代价：用各自规范下前两帧的根
                r0, r1 = seq[i][1 + g], seq[i - 1][1 + gp]
                r2 = seq[i - 2][1 + gp] if i >= 2 else None
                if r0 is None or r1 is None:
                    c = 0.0
                elif r2 is None:
                    c = float(np.linalg.norm(r0 - r1))
                else:
                    c = float(np.linalg.norm(r0 - 2 * r1 + r2))
                t = DP[i - 1][gp] + c + (pen_sw if g != gp else 0.0)
                if best is None or t < best:
                    best, bj = t, gp
            DP[i][g], BP[i][g] = best, bj
    g_star = int(np.argmin(DP[-1]))
    gs = [0] * len(seq)
    for i in range(len(seq) - 1, -1, -1):
        gs[i] = g_star
        g_star = int(BP[i][g_star])
    n_sw = sum(1 for i in range(1, len(seq)) if gs[i] != gs[i - 1])
    print('    [根轨迹] 规范切换 %d 次' % n_sw, flush=True)
    for i, (fr, _r0, _r1) in enumerate(seq):
        if gs[i]:
            st = path[fr]
            path[fr] = tuple(1 - b for b in st)      # XOR 全 1 = 整体换名，几何不变

    # ---- ★★ 一票否决门（first do no harm）：投票后跨视角一致性反而变差 → 整段不投 ----
    #   判据：**GT-free**（只比几何自洽性），所以可以合法地当模型选择用。
    #   实测必要性：05_sword2 本来每个视角的编号就一致（原始一致性 36.9px），
    #   投票反而把它改坏（62.3px）→ 端到端 --pa seq 27.2 → 623.3。
    #   而真正该投的场次（06: 279.6→65.8、01: 824.8→135.0）改善都在 70% 以上。
    def _med_score(get_st):
        vals = []
        for fr, kp2d, em in per_frame:
            if em is None:
                continue
            uv = [v for v in views if v in kp2d]
            if len(uv) < a.min_views:
                continue
            c = frame_cost(kp2d, uv, und, dlt, reproj, get_st(fr))
            if c is not None:
                vals.append(c)
        return float(np.median(vals)) if vals else None

    def _flip_of(fr):
        flip = dict(fixed)
        for v, b in zip(free, path.get(fr, tuple([0] * len(free)))):
            flip[v] = b
        return flip

    sc_raw = _med_score(lambda fr: {v: 0 for v in views})
    sc_vot = _med_score(_flip_of)
    if sc_raw is not None and sc_vot is not None and sc_vot >= sc_raw:
        print('    [不投] 投票后一致性 %.1f 并未优于原始 %.1f → 整段保留原始编号（不动）'
              % (sc_vot, sc_raw), flush=True)
        for fr in list(path):
            path[fr] = tuple([0] * len(free))
    elif sc_raw is not None and sc_vot is not None:
        print('    [采用] 一致性 %.1f -> %.1f（改善 %.0f%%）'
              % (sc_raw, sc_vot, 100.0 * (1 - sc_vot / sc_raw)), flush=True)

    n_flip = 0
    n_chg = 0
    vec_hist = {}
    cur_vec = None
    n_seg = 0
    for fr, kp2d, em in per_frame:
        if em is None:
            continue
        st = path.get(fr, tuple([0] * len(free)))
        flip = dict(fixed)
        for v, b in zip(free, st):
            flip[v] = b
        n_flip += sum(flip.values())
        n_chg += 1
        key = tuple(flip.get(v, 0) for v in views)
        vec_hist[key] = vec_hist.get(key, 0) + 1
        if key != cur_vec:
            n_seg += 1
            cur_vec = key
        if a.dry_run:
            continue
        for v in views:
            src = os.path.join(a.annots, v, '%06d.json' % fr)
            dst = os.path.join(a.out, v, '%06d.json' % fr)
            if not os.path.exists(src):
                continue
            rec = json.load(open(src))
            if flip.get(v, 0) and len(rec.get('annots', [])) >= 2:
                for an in rec['annots']:
                    if int(an['personID']) == 0:
                        an['personID'] = 1
                    elif int(an['personID']) == 1:
                        an['personID'] = 0
            json.dump(rec, open(dst, 'w'), ensure_ascii=False)
    print('[%s] %d 帧参与 · 共翻 %d 个 (帧,视角) 单元（平均每帧 %.2f 个视角）%s'
          % ('dry-run' if a.dry_run else '已写 ' + a.out, n_chg, n_flip,
             n_flip / max(n_chg, 1), '' if a.dry_run else '；接下来照常跑 tri_h4d.py'), flush=True)
    # ★诊断：翻转向量是"整段常数偏移"还是"逐帧乱跳"？段数≈1 就是常数偏移（规范问题）。
    top = sorted(vec_hist.items(), key=lambda kv: -kv[1])[:4]
    print('    翻转向量: %d 段 / %d 帧 | 最常出现的前 %d 个（视角顺序 %s）: %s'
          % (n_seg, n_chg, len(top), '/'.join(views),
             '  '.join('%s×%d' % (''.join(map(str, k)), v) for k, v in top)), flush=True)

    # ---- ★自检：选出配置的**跨视角一致性**打分分布（绝对身份不可观测，一致性才可观测）----
    #   判据：拿"投票前(全不翻)"与"投票后"两种配置的分辨力对比 —— 后者的中位应显著更低。
    sc_voted, sc_raw = [], []
    for fr, kp2d, em in per_frame:
        if em is None or fr not in path:
            continue
        uv = [v for v in views if v in kp2d]
        flip = dict(fixed)
        for v, b in zip(free, path[fr]):
            flip[v] = b
        c1 = frame_cost(kp2d, uv, und, dlt, reproj, flip)
        c0 = frame_cost(kp2d, uv, und, dlt, reproj, {v: 0 for v in views})
        if c1 is not None:
            sc_voted.append(c1)
        if c0 is not None:
            sc_raw.append(c0)
    if sc_voted:
        import numpy as _np
        print('    自检·一致性打分(px, 越小越自洽): 投票后 中位 %.1f / P90 %.1f | 投票前 中位 %.1f'
              % (_np.median(sc_voted), _np.percentile(sc_voted, 90),
                 _np.median(sc_raw) if sc_raw else float('nan')), flush=True)


if __name__ == '__main__':
    main()

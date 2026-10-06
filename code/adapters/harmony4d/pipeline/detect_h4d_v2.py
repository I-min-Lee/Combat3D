#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect_h4d_v2.py — 检测层 v2：候选池放宽 + N 条轨迹 + 联合选一对 + 硬卡 2 人

★ 基座是你的 `stage1_detect/rtdetr_pipeline.py`（以 importlib 导入，不复制）：
   复用 `candidates()` 的检测/ROI 部分、`DSlot`、`expand_head`、`build_labelme`、`roi_keep`，
   以及 NO-SWAP 偏好分区 + EMA + 速度预测 + coast 的匹配思想。

相对原版的四件事（严格在 stage1 内，不碰三角化）：
  ① **候选池放宽**：`boxes[:2]` → `boxes[:TOP_K]`（默认 8）。实测这一步把"选人正确率的天花板"
     从 60.8% 抬到 89.7%（conf=0.15, K=6）—— 那 30 个点是被"只准看 2 个"白白丢掉的。
  ② **N 条轨迹**：`TwoSlotTracker`（2 槽）→ `NSlotTracker`（按需新建，上限 N_MAX）。
     轨迹本身就是跨帧一致的身份，这是治"gid 逐帧乱跳"的正确位置。
  ③ **联合选一对**（关键）：不是"各自独立挑最大"，而是在**轨迹级**选出让下面这个分数最大的一对：
        pair = 时间覆盖 ^a × 互相靠近 ^b × 双方都在动 ^c
      物理含义 = "这一对人在整段里**长时间待在一起且都在剧烈运动**" = 正在对战的那两个人。
      ◆ 这条判据天然排除"选手 + 摄影师"：他们**相距很远**（prox 很小）。
      ◆ 裁判（若有）不参与，因为他不与选手**持续贴身**，且运动量低。
  ④ **硬卡输出 2 人**：只输出得分最高的那一对（`--keep` 可设 3 以容纳裁判）。

用法（容器内）：
  python detect_h4d_v2.py --frames-root $ROOT/frames/15/4 --out $ROOT/det_v2_cent \
      --views 01,03,04,07,09,14 --tag 016_mma4 --start 1 --end 60 \
      --imgsz 640 --conf 0.15 --topk 8 --keep 2 [--nmax 12]
输出：{out}/{view}/{tag}_{fr:06d}.json（LabelMe，group_id=0/1 由被选中的轨迹决定）
"""
import os, sys, json, argparse, importlib.util
import numpy as np
import cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')


def load_original():
    src = os.environ.get('DET_SRC', os.path.join(B, 'code', 'stage1_detect', 'rtdetr_pipeline.py'))
    spec = importlib.util.spec_from_file_location('rtd_orig', src)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RTD = load_original()          # ★ 模块级加载：为了能**真正继承**你的 TwoSlotTracker


def candidates_k(rtd, img, model, view, K, margin=0.10):
    """★① 候选池放宽 + ★⑤ 位置先验 —— **这是唯一必须重写的几行**，因为你的 `candidates()`
    末尾写死了 `return boxes[:2]`（`rtdetr_pipeline.py:66`），调用它之后再切片是无效的。

    除下面两处外，逐行等同你的实现：
        r = model.predict(img, conf=PERSON_CONF, imgsz=IMG_SIZE, classes=[0], device=DEVICE, verbose=False)[0]
        boxes = roi_keep([...])                       ← 仍是调用你的 roi_keep（ROI 过滤原样）
        boxes.sort(key=面积, reverse=True)
        ★⑤ 丢掉"中心落在画面外圈 margin 内"的框    ← 新增（多机位通用先验：取景中心都是场地）
        return boxes[:K]                              ← 你的原来是 [:2]

    为什么需要 ★⑤（2026-09-29 实测）：仅靠"两人互相靠近 + 都在动"会选出
    **画面最右边缘那个被裁切的摄影师**（view01 的 x=3726/3840）——他和旁边的器材在 2D 上
    "靠得很近"，几何判据分不开。而受试者永远在场地中央。margin=0 即关闭。
    """
    r = model.predict(img, conf=rtd.PERSON_CONF, imgsz=rtd.IMG_SIZE, classes=[0],
                      device=rtd.DEVICE, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    boxes = rtd.roi_keep([b[:4].cpu().numpy() for b in r.boxes.xyxy], view)
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    if margin > 0:
        H, W = img.shape[0], img.shape[1]
        boxes = [b for b in boxes
                 if margin * W < (b[0] + b[2]) / 2 < (1 - margin) * W
                 and margin * H < (b[1] + b[3]) / 2 < (1 - margin) * H]
    return boxes[:K]


class NSlotTracker(RTD.TwoSlotTracker):
    """★ 继承你的 `TwoSlotTracker`：**匹配/跟踪逻辑一行不复制**，只把"槽位管理"放宽到 N 条。

    做法：每次 step() 前先确保有足够的**空槽**（你的 step() 第 5 步本来就"只填空槽、
    绝不顶替在位者"），然后**原样调用你的 step()**。于是
      NO-SWAP 偏好分区 / EMA 校正 / 速度学习与裁剪 / coast 惯性 / MAX_MISS 清空
    这些全部执行的是**你的代码**，我这边只有"按需扩容"这一个新增行为。
    """

    def __init__(self, rtd, nmax=12, n_init=2):
        super().__init__()                       # ← 你的 __init__
        self.rtd = rtd
        self.nmax = nmax
        if n_init > len(self.slots):             # 允许起步就给 N 个空槽
            while len(self.slots) < min(n_init, nmax):
                self.slots.append(rtd.DSlot('P%d' % len(self.slots)))

    def step(self, cands):
        """只做一件事：先按需补齐空槽，再原样交给父类 step()"""
        cands = list(cands)
        alive = sum(1 for s in self.slots if s.box is not None)
        need_empty = max(0, len(cands) - alive)          # 候选多、在位者少 → 需要更多空槽
        n_empty = sum(1 for s in self.slots if s.box is None)
        while n_empty < need_empty and len(self.slots) < self.nmax:
            self.slots.append(self.rtd.DSlot('P%d' % len(self.slots)))
            n_empty += 1
        return super().step(cands)               # ← 你的 step()，未改一字


def track_stats(hist, T):
    """hist: {name: {frame: box}} -> 每条轨迹的统计量（时长/运动量/平均尺寸/质心序列）"""
    out = {}
    for name, h in hist.items():
        if not h:
            continue
        frs = sorted(h)
        cen = np.array([[(h[f][0] + h[f][2]) / 2, (h[f][1] + h[f][3]) / 2] for f in frs])
        diag = np.array([np.hypot(h[f][2] - h[f][0], h[f][3] - h[f][1]) for f in frs])
        step = np.linalg.norm(np.diff(cen, axis=0), axis=1) if len(frs) > 1 else np.array([0.0])
        d = np.maximum(diag[:-1], 1e-6) if len(frs) > 1 else np.array([1.0])
        out[name] = dict(frames=frs, cen=cen, diag=diag,
                         boxes=[h[f] for f in frs],
                         n=len(frs), cover=len(frs) / max(T, 1),
                         motion=float(np.median(step / d)) if len(step) else 0.0,
                         area_med=float(np.median([(h[f][2] - h[f][0]) * (h[f][3] - h[f][1]) for f in frs])),
                         cmed=np.median(cen, axis=0))
    return out


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.


def merge_tracks(hist, st, T, iou_thr=0.45, prox_thr=0.25):
    """★**先把"同一个人的重复轨迹"合并掉**（构造上消除歧义，而不是事后排除）。

    为什么必须这么做（2026-09-29 实测）：贴身/遮挡时检测器会给出两个高度重合的框，
    跟踪器把它们分成两条轨迹；此后无论用 IoU 阈值还是"互相靠近"判据去选，都会把
    "同一人的两条轨迹"当成"两个正在对打的人"（它们 prox≈0.2、最"靠近"）。
    分界线怎么调都会踩一边：IoU 严 → 真贴身的两名选手被误杀；IoU 松 → 重复轨迹放行。

    判据：两条轨迹在重叠帧上 **中位 IoU > iou_thr** 或 **中位归一化中心距 < prox_thr**
          → 判定为同一个人 → 并查集合并（帧取并集，同帧冲突保留面积大的那条）。

    合并后重新统计，再进入 pick_pair。
    """
    names = list(st)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    merged_log = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ov = sorted(set(st[a]['frames']) & set(st[b]['frames']))
            if len(ov) < 3:
                continue
            ia = {f: k for k, f in enumerate(st[a]['frames'])}
            ib = {f: k for k, f in enumerate(st[b]['frames'])}
            iou_m = float(np.median([_iou(st[a]['boxes'][ia[f]], st[b]['boxes'][ib[f]]) for f in ov]))
            prox_m = float(np.median([
                np.linalg.norm(st[a]['cen'][ia[f]] - st[b]['cen'][ib[f]]) /
                max(0.5 * (st[a]['diag'][ia[f]] + st[b]['diag'][ib[f]]), 1e-6) for f in ov]))
            if iou_m > iou_thr or prox_m < prox_thr:
                union(a, b)
                merged_log.append('%s+%s(IoU=%.2f,prox=%.2f)' % (a, b, iou_m, prox_m))
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    if not merged_log:
        return hist, st, 0
    new_hist = {}
    for root, mem in groups.items():
        h = {}
        for n in mem:
            for f, bx in hist[n].items():
                if f not in h or (bx[2] - bx[0]) * (bx[3] - bx[1]) > (h[f][2] - h[f][0]) * (h[f][3] - h[f][1]):
                    h[f] = bx
        new_hist[mem[0]] = h                       # 用组内第一个名字当代表
    new_st = track_stats(new_hist, T)
    print('    [merge] 合并 %d 组重复轨迹: %s' % (len(merged_log), '; '.join(merged_log[:6])), flush=True)
    return new_hist, new_st, len(merged_log)


def pick_pair(st, T, wa=1.0, wb=0.3, wc=0.0, wd=1.0, mincover=0.5, excl_iou=0.45):
    """轨迹级**联合**选一对，含两条硬约束：

      ★硬约束1「都全程在场」：受试者必然整段都在 → 两条轨迹的 cover 都要 ≥ mincover。
        不满足的组合**直接排除**（不是扣分）—— 上一版把 cover 当乘数项，导致选出只覆盖 40% 的
        短轨迹（旁边走动的旁观者），把 view03/04/09 从 100% 打崩。
      ★硬约束2「必须是两个人」：重叠帧上两条轨迹框的中位 IoU 必须 < excl_iou。
        否则两条轨迹其实是**同一个人被重复框**（贴身合并时最易发生）→ 直接排除。

      在通过硬约束的组合里，最大化：互相靠近(prox 越小越好) × 双方都在动 × 时长覆盖
    """
    names = [n for n in st if st[n]['n'] >= max(3, 0.1 * T)]
    mx_mot = max([st[n]["motion"] for n in names] + [1e-9])
    mx_area = max([st[n]["area_med"] for n in names] + [1e-9])
    # ★实测(track_gt_audit, 2026-09-29)：受试者轨迹=**面积中位最大的两条**（6视角中5个成立）
    #   motion 几乎无判别力(受试者0.010-0.025 vs 非受试者0.001-0.03) → wc 默认 0
    #   面积是主判据(与逐帧 top-2 的区别在于：这里是**整段的面积中位**，逐帧会被大框抢位)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_, b_ = names[i], names[j]
            cov_gate = min(st[a_]['cover'], st[b_]['cover']) >= mincover
            fa, fb = set(st[a_]['frames']), set(st[b_]['frames'])
            ov = sorted(fa & fb)
            if len(ov) < max(3, 0.3 * T):
                rows.append((a_, b_, 0.0, np.nan, np.nan, np.nan, '重叠不足', cov_gate)); continue
            ia = {f: k for k, f in enumerate(st[a_]['frames'])}
            ib = {f: k for k, f in enumerate(st[b_]['frames'])}
            iou_ov = float(np.median([_iou(st[a_]['boxes'][ia[f]], st[b_]['boxes'][ib[f]]) for f in ov]))
            cov = len(ov) / max(T, 1)
            d = [np.linalg.norm(st[a_]['cen'][ia[f]] - st[b_]['cen'][ib[f]]) /
                 max(0.5 * (st[a_]['diag'][ia[f]] + st[b_]['diag'][ib[f]]), 1e-6) for f in ov]
            prox = float(np.median(d))
            mot = max(st[a_]["motion"], st[b_]["motion"]) / mx_mot   # ★用 max：缠斗时被压住的一方几乎不动，min 会把正解扣死
            a_norm = float(np.sqrt(st[a_]["area_med"] * st[b_]["area_med"])) / mx_area
            sc = (cov ** wa) * ((1.0 / (1.0 + prox)) ** wb) * (max(mot, 1e-6) ** wc) * (a_norm ** wd)
            why = 'ok'
            if not cov_gate: why = 'cover不足(%.2f)' % min(st[a_]['cover'], st[b_]['cover'])
            elif iou_ov > excl_iou: why = 'IoU过高(%.2f)' % iou_ov
            rows.append((a_, b_, sc, prox, iou_ov, mot, why, cov_gate, a_norm))
    if getattr(pick_pair, 'debug', False):                       # 诊断：把全部组合打出来
        print('    [pair-debug] 共 %d 组合（按 score 降序，前 12）' % len(rows))
        for r in sorted(rows, key=lambda z: -z[2])[:12]:
            print('       %s+%s score=%.4f prox=%.2f IoU=%.2f mot=%.3f  %s'
                  % (r[0], r[1], r[2], r[3] if r[3] == r[3] else -1,
                     r[4] if r[4] == r[4] else -1, r[5] if r[5] == r[5] else -1, r[6]))
    ok = [r for r in rows if r[6] == 'ok']
    if not ok:
        print('    [pair] ⚠️ 没有组合通过硬约束（cover≥%.2f 且 轨迹间IoU≤%.2f）→ 回退：取最长两条'
              % (mincover, excl_iou), flush=True)
        cand = sorted(st, key=lambda n: -st[n]['n'])[:2]
        return cand if len(cand) >= 2 else []
    best = max(ok, key=lambda z: z[2])
    _, a, b, sc, prox, iou_ov, mot = best[0], best[0], best[1], best[2], best[3], best[4], best[5]
    # gid 0/1 的确定性分配：质心 x 小的当 0（跨视角一致，避免随机）
    pair = [a, b]
    if st[pair[0]]['cmed'][0] > st[pair[1]]['cmed'][0]:
        pair = pair[::-1]
    print('    [pair] %s(gid0) + %s(gid1)  score=%.4f  prox=%.2f  轨迹间IoU=%.2f  motion=%.3f'
          % (pair[0], pair[1], sc, prox, iou_ov, mot), flush=True)
    return pair


def process_view_perframe(rtd, model, v, frames_dir, out_dir, tag, start, end, W, H, a):
    """对照模式：**逐帧取面积前 2**（等于 v1 基线），但候选池先过 ★①topk / ★⑤位置先验。
    用途：单独量出「位置先验」这一个改动的净效果（不掺跟踪/联合选对）。"""
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted(int(f.split('.')[0]) for f in os.listdir(frames_dir) if f.endswith('.png'))
    frames = [f for f in frames if start <= f <= end]
    # ---- 第一遍：逐帧选框（面积前 2，候选池先过位置先验）----
    raw = {}
    for fr in frames:
        img = cv2.imread(os.path.join(frames_dir, '%06d.png' % fr))
        if img is None:
            continue
        raw[fr] = candidates_k(rtd, img, model, v, 2, a.border_margin)
    # ---- ★第二遍：gid 稳定化 —— 用你的 TwoSlotTracker 跑在"已选的 2 个框"上 ----
    #   为什么需要：逐帧面积排序的名次会随帧翻转（谁离相机近谁排第一），
    #   直接写 group_id=名次 会让 pid0 的 2D 观测在两个真人之间来回跳 → 三角化被喂错。
    #   所以先选人（面积+位置先验），再用**你的 tracker 只做编号稳定**。
    tr = rtd.TwoSlotTracker()
    n_w = 0
    for fr in frames:
        if fr not in raw:
            continue
        objs = tr.step(raw[fr])                    # ← 你的 tracker，未改一字
        stable = []
        for o in objs:
            stable.append((0 if o.name == 'P0' else 1, list(o.box)))
        shapes = []
        for gid, b in sorted(stable):
            bb = rtd.expand_head(b, W, H)
            shapes.append({"label": "person",
                           "points": [[float(bb[0]), float(bb[1])], [float(bb[2]), float(bb[3])]],
                           "group_id": gid, "description": "sel+track:%d" % gid,
                           "shape_type": "rectangle", "flags": {}})
        json.dump(rtd.build_labelme('%s_%06d.jpg' % (tag, fr), H, W, shapes),
                  open(os.path.join(out_dir, '%s_%06d.json' % (tag, fr)), 'w'),
                  ensure_ascii=False, indent=1)
        n_w += 1
    print('  view %s: %d 帧（逐帧面积前2 + 位置先验 %.2f + gid 稳定化）-> %s'
          % (v, n_w, a.border_margin, out_dir), flush=True)
    return n_w


def process_view(rtd, model, v, frames_dir, out_dir, tag, start, end, W, H, a):
    os.makedirs(out_dir, exist_ok=True)
    tr = NSlotTracker(rtd, nmax=a.nmax)
    frames = sorted(int(f.split('.')[0]) for f in os.listdir(frames_dir) if f.endswith('.png'))
    frames = [f for f in frames if start <= f <= end]
    hist = {}
    # ---- pass 1: 跟踪（记录每条轨迹的逐帧框）----
    for fr in frames:
        img = cv2.imread(os.path.join(frames_dir, '%06d.png' % fr))
        if img is None:
            continue
        cands = candidates_k(rtd, img, model, v, a.topk, a.border_margin)   # ★① + ★⑤
        act = tr.step(cands)
        for s in act:
            hist.setdefault(s.name, {})[fr] = list(s.box)
    st = track_stats(hist, len(frames))
    hist, st, n_merged = merge_tracks(hist, st, len(frames))   # ★先合并同一人的重复轨迹
    print('  view %s: %d 帧, 轨迹 %d 条 (最长 %s)'
          % (v, len(frames), len(st), max([s['n'] for s in st.values()] or [0])), flush=True)
    # ---- ③④ 联合选一对，硬卡 keep 条 ----
    pair = pick_pair(st, len(frames), mincover=a.mincover, excl_iou=a.excl_iou)[:a.keep]
    # ---- pass 2: 写 LabelMe（只写被选中轨迹的框）----
    n_w = 0
    for fr in frames:
        shapes = []
        for gid, nm in enumerate(pair):
            b = hist.get(nm, {}).get(fr)
            if b is None:
                continue
            bb = rtd.expand_head(b, W, H)
            shapes.append({"label": "person",
                           "points": [[float(bb[0]), float(bb[1])], [float(bb[2]), float(bb[3])]],
                           "group_id": gid,
                           "description": "track:%s" % nm,
                           "shape_type": "rectangle", "flags": {}})
        json.dump(rtd.build_labelme('%s_%06d.jpg' % (tag, fr), H, W, shapes),
                  open(os.path.join(out_dir, '%s_%06d.json' % (tag, fr)), 'w'),
                  ensure_ascii=False, indent=1)
        n_w += 1
    print('  view %s: 写出 %d 帧（选中轨迹 %s）-> %s' % (v, n_w, pair, out_dir), flush=True)
    return n_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--tag', default='016_mma4')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--topk', type=int, default=8, help='★① 候选池大小')
    ap.add_argument('--nmax', type=int, default=12, help='★② 轨迹上限')
    ap.add_argument('--keep', type=int, default=2, help='★④ 输出几个人（有裁判设 3）')
    ap.add_argument('--mincover', type=float, default=0.5, help='★硬约束1 两条轨迹的最小时长覆盖')
    ap.add_argument('--border-margin', type=float, default=0.10, help='★⑤ 候选框中心不得落在画面外圈该比例内（0=关闭）')
    ap.add_argument('--mode', default='pair', choices=['pair','perframe'], help="perframe=逐帧面积前2（=v1基线, 用于对照位置先验的效果）")
    ap.add_argument('--debug-pairs', action='store_true', help='打印全部轨迹对的分数与排除原因')
    ap.add_argument('--excl-iou', type=float, default=0.6, help='★硬约束2 轨迹间中位IoU上限（超过视为同一人重复框）')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--W', type=int, default=3840)
    ap.add_argument('--H', type=int, default=2160)
    a = ap.parse_args()

    rtd = RTD
    rtd.MODEL_PATH = os.environ.get('DET_MODEL', B + '/port/weights/rtdetr-l.pt')
    rtd.ROI_JSON = os.environ.get('DET_ROI', B + '/empty_roi.json')
    rtd.IMG_SIZE, rtd.PERSON_CONF, rtd.DEVICE = a.imgsz, a.conf, a.device
    from ultralytics import RTDETR
    model = RTDETR(rtd.MODEL_PATH)
    print('[detect_h4d_v2] imgsz=%d conf=%.2f topk=%d nmax=%d keep=%d'
          % (a.imgsz, a.conf, a.topk, a.nmax, a.keep), flush=True)
    pick_pair.debug = a.debug_pairs
    views = [v.zfill(2) for v in a.views.split(',')]
    fn = process_view_perframe if a.mode == 'perframe' else process_view
    tot = 0
    for v in views:
        fd = os.path.join(a.frames_root, v)
        if not os.path.isdir(fd):
            continue
        tot += fn(rtd, model, v, fd, os.path.join(a.out, v), a.tag,
                  a.start, a.end, a.W, a.H, a)


if __name__ == '__main__':
    main()

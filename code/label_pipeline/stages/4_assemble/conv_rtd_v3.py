#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""conv_rtd_v3.py — 在 conv_rtd.py 基础上只改"身份层"，其余逐字不动。

相比 conv_rtd.py 的四处改动:
  1. IDFLIP 默认路径 -> rtd2/idflip_color.json (逐视角分段常数, 由 idflip_color.py 生成)
  2. 新增 SINGLE_JSON: {frame: {view: pid}} —— 该(帧,视角)只有一个人(两条 annot 同色
     或中位关节距<8px), 该视角的观测只归这个 pid, 另一个 pid 写全 0 占位(不投票)。
     依据: 2026-09-20 定位, 该情形下 conv_rtd 会把同一个人的观测同时喂给两个 pid。
  3. pass1 框内指派: 框内候选 >=2 且 rtdetr 现场歧义时, 用 red_black_cls 按"该框
     (翻转后)应属的球衣颜色"挑 annot; 分类器不确定时退回原 conf 规则。
  4. 其余(标定/伪三角化/占位/写盘格式)与 conv_rtd.py 完全一致, 便于逐帧对比。

用法: python3 conv_rtd_v3.py --limit -1
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import cv2

BASE = "/root/autodl-tmp"
OUT = os.environ.get("CONV_OUT", os.path.join(BASE, "easymocap_idfix", "1_1"))
VIEWS = ["1", "3", "4", "7", "11"]
FRAMES_DIR = os.environ.get("CONV_FRAMES_DIR",
                            os.path.join(BASE, "frames", "1", "1"))
RAW_DIR = os.environ.get("RAW_DIR", os.path.join(BASE, "vitpose_rtd5000of", "1", "1"))
DETECT_DIR = os.environ.get("DETECT_DIR", BASE + "/rtdetr_v2/输出/1/1/json")
H = int(os.environ.get("CONV_H", "720"))    # 分辨率：B组 1920x1440 传 1440/1920
W = int(os.environ.get("CONV_W", "960"))
IDFLIP = json.load(open(os.environ.get("IDFLIP_JSON",
                                       "/root/autodl-tmp/rtd2/idflip_color.json")))
SINGLE = json.load(open(os.environ.get("SINGLE_JSON",
                                       "/root/autodl-tmp/_single_assign.json")))
USE_COLOR_PICK = int(os.environ.get("USE_COLOR_PICK", "1"))
ASSIGN_MAX_PX = 120.0
BODY_IDX = [5, 6, 11, 12]
CLS_CONF = 0.80
COLOR_PID = {0: "B", 1: "R"}          # pid0 = 黑方, pid1 = 红方

_CLS = None


def cls_model():
    global _CLS
    if _CLS is None:
        from ultralytics import YOLO
        _CLS = YOLO(BASE + "/video_detect_v2/red_black_cls/models/red_black_cls.pt")
    return _CLS


_CLS_CACHE = {}


def annot_color(view, fr, kp):
    """该 annot 裁剪图的球衣颜色 'R'/'B'/None(不确定)"""
    key = (view, fr, round(float(kp[:, 0].sum()), 3))
    if key in _CLS_CACHE:
        return _CLS_CACHE[key]
    m = cls_model()
    names = list(m.names.values())
    ok = kp[:, 2] > 0.3
    out = None
    if ok.sum() >= 4:
        x0, y0 = kp[ok, :2].min(axis=0)
        x1, y1 = kp[ok, :2].max(axis=0)
        dx, dy = (x1 - x0) * 0.15, (y1 - y0) * 0.15
        x1i, y1i = int(max(0, x0 - dx)), int(max(0, y0 - dy))
        x2i, y2i = int(min(W, x1 + dx)), int(min(H, y1 + dy))
        if x2i - x1i >= 6 and y2i - y1i >= 6:
            img = cv2.imread(os.path.join(FRAMES_DIR, view, "%06d.png" % fr))
            if img is not None:
                r = m.predict(img[y1i:y2i, x1i:x2i], imgsz=224, verbose=False)[0]
                p = r.probs.data.cpu().numpy()
                d = {names[i]: float(p[i]) for i in range(len(names))}
                if max(d["red"], d["black"]) >= CLS_CONF:
                    out = "R" if d["red"] > d["black"] else "B"
    _CLS_CACHE[key] = out
    return out


def load_cams():
    sys.path.insert(0, "/root")
    from easymocap.mytools.camera_utils import read_camera
    _cal = os.environ.get("CONV_CALIB_DIR",
                          os.path.join(BASE, "calib_t11_原始数据求解未修改版"))
    cams = read_camera(os.path.join(_cal, "intri.yml"),
                       os.path.join(_cal, "extri.yml"))
    cams.pop("basenames", None)
    return cams


def torso_center(kp):
    pts = [kp[i, :2] for i in BODY_IDX if kp[i, 2] > 0.3]
    if len(pts) < 2:
        return None
    return np.mean(np.array(pts), axis=0)


def dlt_center(obs, Pdict):
    if len(obs) < 2:
        return None
    A = []
    for v, (x, y) in obs.items():
        P = Pdict[v]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    X = Vh[-1]
    if abs(X[3]) < 1e-12:
        return None
    X = X / X[3]
    res = []
    for v, (x, y) in obs.items():
        p = Pdict[v] @ X
        res.append(np.hypot(p[0] / p[2] - x, p[1] / p[2] - y))
    if np.mean(res) > 30.0:
        return None
    return X[:3]


def read_boxes(view, fr):
    cand = glob.glob(os.path.join(DETECT_DIR, view, "*_%06d.json" % fr))
    if not cand:
        return {}
    j = json.load(open(cand[0]))
    boxes = {}
    for s in j.get("shapes", []):
        if s.get("label") != "person":
            continue
        gid = s.get("group_id")
        if gid is None or int(gid) not in (0, 1):
            continue
        gid = int(gid) ^ IDFLIP.get(str(fr), {}).get(view, 0)
        pts = s["points"]
        boxes[gid] = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]
    return boxes


def read_annots(view, fr):
    f = os.path.join(RAW_DIR, view, "%06d.json" % fr)
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    a = d["annots"] if isinstance(d, dict) else d
    out = []
    for x in a:
        kp = np.array(x["keypoints"], dtype=np.float64)
        if kp.shape != (17, 3):
            continue
        out.append(kp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()

    cams = load_cams()
    Pdict = {v: cams[v]["K"] @ np.hstack([cams[v]["R"], cams[v]["T"]]) for v in VIEWS}

    os.makedirs(os.path.join(OUT, "annots"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    _cd = os.environ.get("CONV_CALIB_DIR")
    if _cd:
        _pairs = [("intri.yml", os.path.join(_cd, "intri.yml")),
                  ("extri.yml", os.path.join(_cd, "extri.yml"))]
    else:
        _pairs = [("intri.yml", os.path.join(BASE, "intri.yml")),
                  ("extri.yml", os.path.join(BASE, "extri_refined.yml"))]
    for fn, src in _pairs:
        dst = os.path.join(OUT, fn)
        if not os.path.islink(dst) and not os.path.exists(dst):
            os.symlink(src, dst)
    for v in VIEWS:
        os.makedirs(os.path.join(OUT, "images", v), exist_ok=True)
        os.makedirs(os.path.join(OUT, "annots", v), exist_ok=True)

    frames = sorted(int(os.path.basename(x).split(".")[0])
                    for x in glob.glob(os.path.join(RAW_DIR, "1", "*.json")))
    frames = frames[args.start:]
    if args.limit > 0:
        frames = frames[:args.limit]
    print("converting %d frames: %d..%d step%d" % (
        len(frames), frames[0], frames[-1], args.step), flush=True)

    n_noid = {v: 0 for v in VIEWS}
    n_single = 0
    n_colorpick = 0
    for fi, fr in enumerate(frames):
        if fi % args.step:
            continue
        data_v = {}
        for v in VIEWS:
            ann = read_annots(v, fr)
            if ann is None:
                data_v[v] = None
                continue
            boxes = read_boxes(v, fr)
            data_v[v] = {"ann": ann, "boxes": boxes,
                         "ctr": [torso_center(k) for k in ann]}
            img_src = os.path.join(FRAMES_DIR, v, "%06d.png" % fr)
            img_dst = os.path.join(OUT, "images", v, "%06d.png" % fr)
            if os.path.exists(img_src) and not os.path.exists(img_dst):
                try:
                    os.symlink(img_src, img_dst)
                except FileExistsError:
                    pass

        # ---- pass 1: 有框视角 -> gid 匹配(框内按 conf 降序, 颜色歧义时按球衣色挑) ----
        assign = {v: {} for v in VIEWS}
        pseudo_obs = {0: {}, 1: {}}
        for v in VIEWS:
            dv = data_v[v]
            if dv is None or not dv["boxes"]:
                continue
            for gid, bx in dv["boxes"].items():
                if gid not in (0, 1):
                    continue
                pad = 0.1 * max(bx[2] - bx[0], bx[3] - bx[1])
                inside = []
                for i, k in enumerate(dv["ann"]):
                    c = dv["ctr"][i]
                    if c is None:
                        continue
                    if (bx[0] - pad <= c[0] <= bx[2] + pad and
                            bx[1] - pad <= c[1] <= bx[3] + pad):
                        inside.append((float(k[:, 2].mean()), i))
                inside.sort(reverse=True)
                best = -1
                if USE_COLOR_PICK and len(inside) >= 2:
                    want = COLOR_PID[gid]
                    for _cf, _i in inside:
                        if _i in assign[v]:
                            continue
                        if annot_color(v, fr, dv["ann"][_i]) == want:
                            best = _i
                            n_colorpick += 1
                            break
                if best < 0:
                    for _cf, _i in inside:
                        if _i not in assign[v]:
                            best = _i
                            break
                if best >= 0:
                    assign[v][best] = gid
                    pseudo_obs[gid][v] = tuple(dv["ctr"][best])
        pseudo3d = {gid: dlt_center(obs, Pdict) for gid, obs in pseudo_obs.items()}

        # ---- pass 2: 无框/漏框视角 -> 伪3D投影最近匹配 ----
        for v in VIEWS:
            dv = data_v[v]
            if dv is None:
                continue
            taken = set(assign[v].values())
            for gid, X in pseudo3d.items():
                if X is None or gid in taken:
                    continue
                p = Pdict[v] @ np.append(X, 1.0)
                if abs(p[2]) < 1e-9:
                    continue
                px = np.array([p[0] / p[2], p[1] / p[2]])
                best, bd = -1, 1e18
                for i, c in enumerate(dv["ctr"]):
                    if c is None or i in assign[v]:
                        continue
                    dd = np.linalg.norm(c - px)
                    if dd < bd:
                        bd, best = dd, i
                if best >= 0 and bd <= ASSIGN_MAX_PX:
                    assign[v][best] = gid
            n_noid[v] += sum(1 for i in range(len(dv["ann"])) if i not in assign[v])

        # ---- pass 3: 写 annots json (单人帧只给一个 pid) ----
        sing = SINGLE.get(str(fr), {})
        for v in VIEWS:
            dv = data_v[v]
            real = {0: None, 1: None}
            if dv is not None:
                for i, gid in assign[v].items():
                    if real[gid] is None:
                        real[gid] = dv["ann"][i]
            if v in sing:
                keep = int(sing[v])
                other = 1 - keep
                # 该视角只有一个人: 观测强制归颜色匹配的那个 pid, 另一个写占位
                if real[keep] is None and real[other] is not None:
                    real[keep] = real[other]
                if real[other] is not None:
                    real[other] = None
                    n_single += 1
            annots = []
            for gid in (0, 1):
                if real[gid] is not None:
                    annots.append({"personID": int(gid),
                                   "keypoints": np.round(real[gid], 4).tolist()})
                else:
                    annots.append({"personID": int(gid),
                                   "keypoints": [[0.0, 0.0, 0.0] for _ in range(17)]})
            out = {"height": H, "width": W, "annots": annots}
            json.dump(out, open(os.path.join(OUT, "annots", v, "%06d.json" % fr), "w"))
        if fi % 200 == 0:
            print("  frame %d/%d (%d)" % (fi, len(frames), fr), flush=True)

    print("dropped(spectator/unassigned) per view:", n_noid, flush=True)
    print("single-view 观测取消次数:", n_single, " 颜色挑 annot 次数:", n_colorpick)
    print("done ->", OUT)


if __name__ == "__main__":
    main()

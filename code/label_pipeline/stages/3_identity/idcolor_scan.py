#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""idcolor_scan.py — 2D 层身份证据采集
对每(帧,视角):
  1. 读原始 vitpose 两条 annot, 算中位关节距 d01 (重复检测指标)
  2. 按 conv_rtd pass1 同款规则把 annot 指到 gid 框 (躯干中心落框 + conf 优先)
  3. 用 red_black_cls 判每条 annot 的球衣颜色 (red/black)
  4. 输出 "box gid0 里的人是红还是黑" -> 该视角该帧的翻转证据
输出 /root/autodl-tmp/_idcolor_obs.csv
"""
import json, os, glob, sys
import numpy as np
import cv2
from ultralytics import YOLO

B = "/root/autodl-tmp"
FR = os.environ.get("FRAMES_DIR", B + "/frames/1/1")
RAW = os.environ.get("RAW_DIR", B + "/vitpose_rtd5000of/1/1")
DET = os.environ.get("DETECT_DIR", B + "/rtdetr_v2/输出/1/1/json")
OUT = os.environ.get("OBS_OUT", B + "/_idcolor_obs.csv")
VIEWS = ["1", "3", "4", "7", "11"]
TORSO = [5, 6, 11, 12]
STEP = int(os.environ.get("STEP4", "3"))
N = int(os.environ.get("NFR", "14017"))
FR0 = int(os.environ.get("FR0", "0"))
FR1 = int(os.environ.get("FR1", str(N)))
CLS_CONF = 0.80
DUP_TOL = 8.0

M = YOLO(B + "/video_detect_v2/red_black_cls/models/red_black_cls.pt")
NAMES = list(M.names.values())


def kpbox(kp, pad=0.15):
    ok = kp[:, 2] > 0.3
    if ok.sum() < 4:
        return None
    x0, y0 = kp[ok, :2].min(axis=0)
    x1, y1 = kp[ok, :2].max(axis=0)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return [x0 - dx, y0 - dy, x1 + dx, y1 + dy]


def torso(kp):
    p = [kp[i, :2] for i in TORSO if kp[i, 2] > 0.3]
    return np.mean(p, axis=0) if len(p) >= 2 else None


def classify(img, kp):
    bx = kpbox(kp)
    if bx is None:
        return None
    x1, y1, x2, y2 = [int(round(v)) for v in bx]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    r = M.predict(img[y1:y2, x1:x2], imgsz=224, verbose=False)[0]
    p = r.probs.data.cpu().numpy()
    d = {NAMES[i]: float(p[i]) for i in range(len(NAMES))}
    return d["red"], d["black"]


def read_boxes(v, fr):
    cand = glob.glob(os.path.join(DET, v, "*_%06d.json" % fr))
    if not cand:
        return {}
    out = {}
    for s in json.load(open(cand[0])).get("shapes", []):
        if s.get("label") != "person" or s.get("group_id") is None:
            continue
        p = s["points"]
        out[int(s["group_id"])] = [p[0][0], p[0][1], p[1][0], p[1][1]]
    return out


def read_annots(v, fr):
    f = os.path.join(RAW, v, "%06d.json" % fr)
    if not os.path.exists(f):
        return []
    d = json.load(open(f))
    ks = []
    for a in (d["annots"] if isinstance(d, dict) else d):
        k = np.array(a["keypoints"], float)
        if k.shape == (17, 3):
            ks.append(k)
    return ks


def assign_to_gid(ks, boxes):
    """照抄 conv_rtd pass1: 每个 gid 框取框内 conf 最高的未占用 annot"""
    ctr = [torso(k) for k in ks]
    assign = {}
    for gid, bx in boxes.items():
        pad = 0.1 * max(bx[2] - bx[0], bx[3] - bx[1])
        inside = []
        for i, c in enumerate(ctr):
            if c is None or i in assign:
                continue
            if bx[0] - pad <= c[0] <= bx[2] + pad and bx[1] - pad <= c[1] <= bx[3] + pad:
                inside.append((float(ks[i][:, 2].mean()), i))
        inside.sort(reverse=True)
        if inside:
            assign[inside[0][1]] = gid
    return assign


hdr = "fr,view,n_ann,d01,dup,gid0_idx,gid0_red,gid0_blk,gid1_idx,gid1_red,gid1_blk,flip_obs"
fo = open(OUT, "w")
fo.write(hdr + "\n")
nrow = 0
for fr in range(FR0, FR1, STEP):
    for v in VIEWS:
        ks = read_annots(v, fr)
        if len(ks) < 2:
            continue
        ok = (ks[0][:, 2] > 0.3) & (ks[1][:, 2] > 0.3)
        d01 = float(np.median(np.linalg.norm(ks[0][ok, :2] - ks[1][ok, :2], axis=1))) \
            if ok.sum() >= 5 else -1.0
        dup = 1 if (0 <= d01 < DUP_TOL) else 0
        img = cv2.imread(os.path.join(FR, v, "%06d.png" % fr))
        if img is None:
            continue
        boxes = read_boxes(v, fr)
        asg = assign_to_gid(ks[:2], boxes)
        idx_of = {g: i for i, g in asg.items()}
        i0, i1 = idx_of.get(0), idx_of.get(1)
        c0 = classify(img, ks[i0]) if i0 is not None else None
        c1 = classify(img, ks[i1]) if i1 is not None else None
        # 翻转证据: gid0 框里是红方 -> 需要 flip=1 (才能让 pid0=黑方)
        flip = ""
        if c0:
            r, b = c0
            if max(r, b) >= CLS_CONF:
                flip = "1" if r > b else "0"
        fo.write("%d,%s,%d,%.1f,%d,%s,%s,%s,%s,%s,%s,%s\n" % (
            fr, v, len(ks), d01, dup,
            "" if i0 is None else i0, "" if c0 is None else "%.3f" % c0[0],
            "" if c0 is None else "%.3f" % c0[1],
            "" if i1 is None else i1, "" if c1 is None else "%.3f" % c1[0],
            "" if c1 is None else "%.3f" % c1[1], flip))
        nrow += 1
    if fr % 600 < STEP:
        print("  ...f%d  rows=%d" % (fr, nrow), flush=True)
fo.close()
print("done rows=%d -> %s" % (nrow, OUT))

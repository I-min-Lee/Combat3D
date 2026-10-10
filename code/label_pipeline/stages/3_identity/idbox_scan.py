#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""idbox_scan.py — 框级颜色证据采集（补 annot 级证据在"贴靠/两框互含"时缺失的缺口）
对每(帧,视角): 用 red_black_cls 判 **检测框本身**的裁剪图颜色
输出 _idbox_obs*.csv: fr,view,n_box,w0,h0,red0,blk0,w1,h1,red1,blk1,box_ev
   box_ev: 1 = 框gid0是红方(需翻) ; 0 = 框gid0是黑方 ; 空 = 不决定性
依据: video_detect.py 本来就用这个模型做"贴身红黑互换"判据(CLS_CONF_HIGH=0.80),
      即模型就是按"整框裁剪图"训练的 → 框级是它的原生用法。
"""
import json, os, glob
import numpy as np
import cv2
from ultralytics import YOLO

B = "/root/autodl-tmp"
FR = os.environ.get("FRAMES_DIR", B + "/frames/1/1")
DET = os.environ.get("DETECT_DIR", B + "/rtdetr_v2/输出/1/1/json")
OUT = os.environ.get("OBS_OUT", B + "/_idbox_obs.csv")
VIEWS = ["1", "3", "4", "7", "11"]
STEP = int(os.environ.get("STEP5", "3"))
FR1 = int(os.environ.get("FR1", "14017"))
FR0 = int(os.environ.get("FR0", "0"))
N = int(os.environ.get("NFR", "14017"))
CONF = 0.80
MIN_W, MIN_H = 30, 60

M = YOLO(B + "/video_detect_v2/red_black_cls/models/red_black_cls.pt")
NAMES = list(M.names.values())


def cls(img, bx):
    x1, y1, x2, y2 = [int(round(v)) for v in bx]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < MIN_W or y2 - y1 < MIN_H:
        return None
    r = M.predict(img[y1:y2, x1:x2], imgsz=224, verbose=False)[0]
    p = r.probs.data.cpu().numpy()
    d = {NAMES[i]: float(p[i]) for i in range(len(NAMES))}
    if max(d["red"], d["black"]) < CONF:
        return None
    return ("R", d["red"], d["black"]) if d["red"] > d["black"] else ("B", d["red"], d["black"])


fo = open(OUT, "w")
fo.write("fr,view,w0,h0,c0,r0,b0,w1,h1,c1,r1,b1,box_ev\n")
nrow = 0
for fr in range(FR0, FR1, STEP):
    for v in VIEWS:
        cand = glob.glob(os.path.join(DET, v, "*_%06d.json" % fr))
        if not cand:
            continue
        boxes = {}
        for s in json.load(open(cand[0])).get("shapes", []):
            if s.get("label") == "person" and s.get("group_id") is not None:
                p = s["points"]
                boxes[int(s["group_id"])] = [p[0][0], p[0][1], p[1][0], p[1][1]]
        img = cv2.imread(os.path.join(FR, v, "%06d.png" % fr))
        if img is None:
            continue
        res = {}
        for g in (0, 1):
            bx = boxes.get(g)
            if bx is None:
                continue
            r = cls(img, bx)
            res[g] = (bx, r)
        if not res:
            continue
        ev = ""
        c0 = res.get(0, (None, None))[1]
        c1 = res.get(1, (None, None))[1]
        if c0 and c1 and c0[0] != c1[0]:
            ev = "1" if c0[0] == "R" else "0"
        b0 = res.get(0, ([None] * 4, None))[0]
        b1 = res.get(1, ([None] * 4, None))[0]
        fo.write("%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
            fr, v,
            "" if b0[0] is None else int(b0[2] - b0[0]),
            "" if b0[0] is None else int(b0[3] - b0[1]),
            "" if c0 is None else c0[0],
            "" if c0 is None else "%.3f" % c0[1],
            "" if c0 is None else "%.3f" % c0[2],
            "" if b1[0] is None else int(b1[2] - b1[0]),
            "" if b1[0] is None else int(b1[3] - b1[1]),
            "" if c1 is None else c1[0],
            "" if c1 is None else "%.3f" % c1[1],
            "" if c1 is None else "%.3f" % c1[2], ev))
        nrow += 1
    if fr % 600 < STEP:
        print("  ...f%d rows=%d" % (fr, nrow), flush=True)
fo.close()
print("done rows=%d -> %s" % (nrow, OUT))

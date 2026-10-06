# -*- coding: utf-8 -*-
"""verify_all2.py — 逐场验证标定（RANSAC 三角化 + 内点率）

改进点（vs 第一版）：
  · 三角化用 RANSAC：两两视角组合求交，选重投影一致度最高的解，
    这样个别视角身份配错也能被多数视角投票压掉（v2 的 pid 就是反的）
  · 统计"内点率"：重投影 < 15px 的观测占比 —— 这才是标定可用性的直接指标
    （中位残差会被离群值污染，内点率不会）

判据：内点率高、残差低 = 这套标定对该场可用。
"""
import os, sys, json, glob, itertools
import numpy as np
import cv2

B = "/root/autodl-tmp"
PROBE = B + "/calib_probe"
CAL = B + "/calib_t11_主点修正"
VIEWS5 = ["1", "3", "4", "7", "11"]
VIEWS6 = ["1", "2", "3", "4", "7", "11"]
INL_THR = 15.0
FIT_THR = 25.0

sys.path.insert(0, B + "/emoff/EasyMocap-master")
from easymocap.mytools.camera_utils import read_camera
cams = read_camera(CAL + "/intri.yml", CAL + "/extri.yml")
cams.pop("basenames", None)

K2_768 = np.array([[1114.289627, 0, 455.195792],
                   [0, 1083.338694, 355.955055],
                   [0, 0, 1.0]])

TAKES = sorted([os.path.basename(d.rstrip("/")) for d in glob.glob(PROBE + "/*_*/")],
               key=lambda t: (int(t.split("_")[0].split(".")[0]), int(t.split("_")[1])))


def group_of(take):
    a, b = take.split("_")[0].split(".")
    maj = int(a)
    if 1 <= maj <= 4:
        return "A"
    if maj == 0 or 5 <= maj <= 20:
        return "B"
    return "C"


def cam_of(view, take):
    g = group_of(take)
    cm = cams[view]
    K = np.asarray(cm["K"], float).reshape(3, 3).copy()
    if view == "2":
        if g == "C":
            K = K2_768.copy()
    elif g == "B":
        K[0, 0] *= 2; K[1, 1] *= 2
        K[0, 2] *= 2; K[1, 2] *= 2
    return (K, np.asarray(cm["dist"], float).reshape(-1),
            np.asarray(cm["R"], float).reshape(3, 3),
            np.asarray(cm["T"], float).reshape(3, 1))


def proj(X, cam):
    K, d, R, T = cam
    rv = cv2.Rodrigues(np.ascontiguousarray(R))[0]
    uv, _ = cv2.projectPoints(np.ascontiguousarray(np.asarray(X, float).reshape(-1, 3)),
                              rv, T, K, d)
    return uv.reshape(-1, 2)


def _dlt(obs, cams_used):
    A = []
    for uv, cam in zip(obs, cams_used):
        K, d, R, T = cam
        P = K @ np.hstack([R, T])
        u, v = uv
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    try:
        _, _, Vt = np.linalg.svd(A)
    except Exception:
        return None
    X = Vt[-1]
    if abs(X[3]) < 1e-9:
        return None
    return X[:3] / X[3]


def triangulate(obs, cams_used):
    n = len(obs)
    if n < 2:
        return None
    best, best_inl = None, -1
    for i, j in itertools.combinations(range(n), 2):
        X = _dlt([obs[i], obs[j]], [cams_used[i], cams_used[j]])
        if X is None:
            continue
        inl = 0
        for k in range(n):
            uv = proj(X, cams_used[k])[0]
            if np.linalg.norm(uv - obs[k]) < FIT_THR:
                inl += 1
        if inl > best_inl:
            best_inl, best = inl, X
    if best is None or best_inl < max(2, n // 2):
        return None
    io, ic = [], []
    for k in range(n):
        uv = proj(best, cams_used[k])[0]
        if np.linalg.norm(uv - obs[k]) < FIT_THR:
            io.append(obs[k]); ic.append(cams_used[k])
    if len(io) >= 2:
        X2 = _dlt(io, ic)
        if X2 is not None:
            return X2
    return best


rows = []
for take in TAKES:
    d = os.path.join(PROBE, take)
    n_person = 0
    errs = {v: [] for v in VIEWS6}
    vp_files = {}
    for v in VIEWS6:
        for f in glob.glob(os.path.join(d, "vp", v, "*.json")):
            vp_files.setdefault(os.path.basename(f), {})[v] = f
    for fname, vmap in sorted(vp_files.items()):
        if len(vmap) < 5:
            continue
        per_gid = {}
        for v, fp in vmap.items():
            try:
                a = json.load(open(fp))
            except Exception:
                continue
            for an in a.get("annots", []):
                per_gid.setdefault(int(an["personID"]), {})[v] = np.array(an["keypoints"], float)
        for gid, obsv in per_gid.items():
            v5 = [v for v in VIEWS5 if v in obsv]
            if len(v5) < 4:
                continue
            pts3d, used = [], []
            for j in range(17):
                o, cc = [], []
                for v in v5:
                    S = obsv[v]
                    if S[j, 2] > 0.3:
                        o.append(S[j, :2]); cc.append(cam_of(v, take))
                if len(o) >= 3:
                    X = triangulate(o, cc)
                    if X is not None:
                        pts3d.append(X); used.append(j)
            if len(pts3d) < 5:
                continue
            pts3d = np.asarray(pts3d)
            n_person += 1
            for v in VIEWS6:
                if v not in obsv:
                    continue
                S = obsv[v]
                uv = proj(pts3d, cam_of(v, take))
                for k, j in enumerate(used):
                    if S[j, 2] > 0.3:
                        errs[v].append(float(np.linalg.norm(uv[k] - S[j, :2])))
    row = dict(take=take, group=group_of(take), n=n_person)
    for v in VIEWS6:
        e = np.asarray(errs[v]) if errs[v] else None
        row[v] = float(np.median(e)) if e is not None and len(e) else None
        row[v + "i"] = float((e < INL_THR).mean() * 100) if e is not None and len(e) else None
    rows.append(row)

print("场次      组   样本 |  " + "  ".join("%-13s" % ("v" + v) for v in VIEWS6))
print("                    |  " + "  ".join("%-13s" % "残差/内点%" for _ in VIEWS6))
print("-" * 108)
for r in rows:
    cs = []
    for v in VIEWS6:
        if r[v] is None:
            cs.append("%-13s" % "  --")
        else:
            cs.append("%5.0f/%4.0f%%" % (r[v], r[v + "i"]))
    print("%-9s %-3s %5d |  %s" % (r["take"], r["group"], r["n"], "  ".join(cs)))

print()
print("=== 按组汇总（各场中位数）===")
for g in ["A", "B", "C"]:
    sub = [r for r in rows if r["group"] == g and r["n"] > 0]
    if not sub:
        continue
    names = {"A": "960x544", "B": "1920x1088", "C": "960x768"}
    print("  组%s (%s, %d场):" % (g, names[g], len(sub)))
    for v in VIEWS6:
        vals = [r[v] for r in sub if r[v] is not None]
        inls = [r[v + "i"] for r in sub if r.get(v + "i") is not None]
        if vals:
            print("     v%-3s  残差 %6.0f px   内点率 %5.1f%%" % (
                v, np.median(vals), np.median(inls) if inls else 0))

# -*- coding: utf-8 -*-
"""把三角化产物 final13_<tag>（200fps，14017帧）抽成 25fps 并重编号，
以便与 MotionBERT 版（同为 25fps）逐帧对齐对比。"""
import os, sys, json, glob, shutil

src = sys.argv[1] if len(sys.argv) > 1 else "/workshop/Lym/combat3d/final13_idfix"
dst = sys.argv[2] if len(sys.argv) > 2 else "/workshop/Lym/combat3d/final13_tri25"
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 8

for pid in (0, 1):
    fs = sorted(glob.glob(f"{src}/pid{pid}/keypoints3d/*.json"))
    od = f"{dst}/pid{pid}/keypoints3d"
    os.makedirs(od, exist_ok=True)
    n = 0
    for i in range(0, len(fs), stride):
        shutil.copyfile(fs[i], f"{od}/{n:06d}.json")
        n += 1
    print(f"pid{pid}: {len(fs)} -> {n} 帧  {od}")
print("完成 ->", dst)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mb_fixpaths.py —— 把迁移过来的代码里所有 /workshop/Lym/combat3d 换成新根

★ 为什么不用 mb_portabilize.py 那套"推导式"：它按**带引号的完整字面量**匹配
  '/workshop/Lym/combat3d'，匹配不到 '/workshop/Lym/combat3d/mb/kb'（后面还跟着路径）这种。
  迁移是一次性的，纯子串替换更彻底、无遗漏。

用法: python mb_fixpaths.py <新根> [<扫描目录>]
"""
import os, sys

NEW = sys.argv[1] if len(sys.argv) > 1 else '/workshop/Lym/combat3d'
ROOT = sys.argv[2] if len(sys.argv) > 2 else NEW
OLD = '/workshop/Lym/combat3d'
SKIP_DIRS = {'envs', '.git', '__pycache__', 'node_modules'}

nf = ns = 0
for dirpath, dirnames, files in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for f in files:
        if not f.endswith('.py'):
            continue
        p = os.path.join(dirpath, f)
        try:
            s = open(p, encoding='utf-8').read()
        except Exception as e:
            print(f'  !! 读不了 {p}: {e}')
            continue
        if OLD not in s:
            continue
        c = s.count(OLD)
        open(p, 'w', encoding='utf-8').write(s.replace(OLD, NEW))
        nf += 1
        ns += c
        print(f'{c:3d} 处  {p}')
print(f'--- 共 {nf} 个文件 / {ns} 处替换 ---')

# 残留自检
left = []
for dirpath, dirnames, files in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(dirpath, f)
            try:
                if OLD in open(p, encoding='utf-8').read():
                    left.append(p)
            except Exception:
                pass
print('残留:', left if left else '无 ✓')

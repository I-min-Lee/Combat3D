#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mb_portabilize.py —— 把迁移过来的 MotionBERT 体系里的硬编码 /workshop/Lym/combat3d 换成可推导的根

背景：kb_common.py 里 ROOT/OUT/MB_ROOT/CK_FT/CALIB 全部硬编码 /workshop/Lym/combat3d；
      各顶层脚本的 sys.path.insert 和 B 也是。搬到 222 必须改。

做法：在每份文件**第一个出现 '/workshop/Lym/combat3d' 的模块级语句之前**插入
        _MBROOT = <推导式>
      然后把所有字面量 '/workshop/Lym/combat3d' 替换成 _MBROOT。
      ★ 不写死新路径：推导式按"本文件到根有几级"来，因此同一份代码在
        westc(/workshop/Lym/combat3d) 和 222(/workshop/Lym/combat3d) 都能跑。
      ★ 可用环境变量 MB_ROOT_DIR 覆盖。

用法: python mb_portabilize.py <目标根>      # 如 /workshop/Lym/combat3d
"""
import os, sys

LIT = "'/workshop/Lym/combat3d'"
TARGET = sys.argv[1]

# 顶层脚本：根 = 文件所在目录
DEF_TOP = ("# --- 迁移补丁 2026-10-03：根目录从文件位置推导，不再硬编码 /workshop/Lym/combat3d ---\n"
           "_MBROOT = os.environ.get('MB_ROOT_DIR') or os.path.dirname(os.path.abspath(__file__))\n")
# mb/kb/ 下的文件：根 = 上溯三级
DEF_KB = ("# --- 迁移补丁 2026-10-03：根目录从文件位置推导，不再硬编码 /workshop/Lym/combat3d ---\n"
          "_MBROOT = os.environ.get('MB_ROOT_DIR') or os.path.dirname(\n"
          "    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")


def patched(path, definition):
    src = open(path, encoding='utf-8').read()
    if '_MBROOT' in src:
        return 'already'
    if LIT not in src:
        return 'skip(no-literal)'
    lines = src.split('\n')
    # 找第一个含字面量的【模块级】行（无前导空白）
    at = None
    for i, ln in enumerate(lines):
        if LIT in ln:
            if ln[:1] in (' ', '\t'):
                return f'FAIL(首个出现处有缩进, line {i+1})'
            at = i
            break
    if at is None:
        return 'FAIL(找不到)'
    lines.insert(at, definition.rstrip('\n'))
    out = '\n'.join(lines).replace(LIT, '_MBROOT')
    open(path, 'w', encoding='utf-8').write(out)
    return f'ok(插入在第 {at+1} 行前)'


def main():
    jobs = []
    for f in sorted(os.listdir(TARGET)):
        if f.endswith('.py'):
            jobs.append((os.path.join(TARGET, f), DEF_TOP))
    kbdir = os.path.join(TARGET, 'mb', 'kb')
    if os.path.isdir(kbdir):
        for f in sorted(os.listdir(kbdir)):
            if f.endswith('.py'):
                jobs.append((os.path.join(kbdir, f), DEF_KB))
    for p, d in jobs:
        r = patched(p, d)
        print(f'{r:<28} {p}')


if __name__ == '__main__':
    main()

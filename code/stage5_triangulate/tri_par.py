#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tri_par.py <pid> <out_root> <start> <end> <lams> [min_conf]
按 pid 单进程跑时序三角化 —— 只是 import rtd2/tri_temporal_ab.py 并调用它自己的
build_cams / index_det / run_pid，**不修改任何原脚本**。
pid0 与 pid1 完全独立（collect 按 personID 过滤，solve 只在单个关节的时间序列内耦合），
所以两个进程并行产出与串行逐字节一致（该脚本无随机数）。
前置环境变量（必须在 import 之前设好，模块级读取）：
  AB_CAL AB_VIEWS AB_ANNOTS AB_W_IMG AB_H_IMG AB_DET_DIR
"""
import os, sys, time
B = '/root/autodl-tmp'
sys.path.insert(0, B + '/rtd2')
sys.path.insert(0, B + '/emoff/EasyMocap-master')

pid = int(sys.argv[1]); out = sys.argv[2]
start, end = int(sys.argv[3]), int(sys.argv[4])
lams = [float(x) for x in sys.argv[5].split(',')]
mc = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
order = int(os.environ.get('AB_ORDER', '2'))

import tri_temporal_ab as T          # noqa: E402  —— 必须在上面的环境变量之后

t0 = time.time()
CAL = os.environ.get('AB_CAL', B + '/calib_t11_原始数据求解未修改版')
cams = T.build_cams(CAL + '/intri.yml', CAL + '/extri.yml')
detidx = {v: T.index_det(v) for v in T.VIEWS}
print('[tri_par] pid%d start=%d end=%d lams=%s order=%d calib=%s'
      % (pid, start, end, lams, order, CAL), flush=True)
T.run_pid(pid, out, start, end, mc, detidx, cams, lams, order)
print('[tri_par] pid%d DONE %.1f min' % (pid, (time.time() - t0) / 60), flush=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect_h4d.py — 行 B：用**你自己的检测层**在 Harmony4D 上跑（检测/选人/跟踪逻辑逐字不改）

★ 本脚本**不复制** `stage1_detect/rtdetr_pipeline.py` 的任何逻辑：
  以 importlib 导入原模块，只覆盖它的**模块级常量**（路径/尺寸/置信度/设备），然后复用
      candidates()      —— ROI 过滤 + **面积 top-2 选人**（★待评估的选人规则）
      TwoSlotTracker()  —— 速度预测双槽（身份 = 纯几何连续）
      expand_head()     —— 框外扩
      build_labelme()   —— LabelMe 写法
  唯一新增的是**帧来源**：Harmony4D 给的是已抽好的 `%06d.png`，不是 .avi。

为什么这样做：行 B 要评的就是"你的检测+选人"在公开数据集上的准度，
少复制一份就少一份"跑的不是你的代码"的质疑。

用法（容器内）：
  COMBAT3D_ROOT=/workshop/Lym/combat3d \
  DET_MODEL=/workshop/Lym/combat3d/port/weights/rtdetr-l.pt \
  python detect_h4d.py --frames-root $ROOT/frames/15/4 --out $ROOT/det_ours \
      --views 01,03,04,07,09,14 --tag 016_mma4 --start 1 --end 741 \
      --imgsz 640 [--conf 0.15] [--device 0]
输出：{out}/{view}/{tag}_{fr:06d}.json（与 vp_h4d.py / conv 的读取约定一致）
"""
import os, argparse, importlib.util, json
import numpy as np
import cv2

B = os.environ.get('COMBAT3D_ROOT', '/workshop/Lym/combat3d')
SRC = os.environ.get('DET_SRC', os.path.join(B, 'code', 'stage1_detect', 'rtdetr_pipeline.py'))


def load_original():
    """导入你的检测模块（只执行模块级定义，不跑它的 main）"""
    spec = importlib.util.spec_from_file_location('rtd_orig', SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def process_view(rtd, model, view, frames_dir, out_dir, tag, start, end, W, H, render=None):
    """镜像原 process_video 的循环，但帧来自图片；框外扩/写盘/LabelMe 全用原函数"""
    os.makedirs(out_dir, exist_ok=True)
    tr = rtd.TwoSlotTracker()
    frames = sorted(int(f.split('.')[0]) for f in os.listdir(frames_dir) if f.endswith('.png'))
    frames = [f for f in frames if start <= f <= end]
    n = 0
    for fr in frames:
        img = cv2.imread(os.path.join(frames_dir, '%06d.png' % fr))
        if img is None:
            continue
        objs = tr.step(rtd.candidates(img, model, view))          # ★ 你的选人 + 跟踪
        shapes = []
        for o in objs:
            b = rtd.expand_head(o.box, W, H)                      # ★ 你的框外扩
            shapes.append({"label": "person",
                           "points": [[float(b[0]), float(b[1])], [float(b[2]), float(b[3])]],
                           "group_id": 0 if o.name == "P0" else 1,
                           "description": "color:red" if o.name == "P0" else "color:black",
                           "shape_type": "rectangle", "flags": {}})
        data = rtd.build_labelme('%s_%06d.jpg' % (tag, fr), H, W, shapes)   # ★ 你的 LabelMe 写法
        json.dump(data, open(os.path.join(out_dir, '%s_%06d.json' % (tag, fr)), 'w'),
                  ensure_ascii=False, indent=1)
        n += 1
        if n % 100 == 0:
            print('    view %s: %d/%d' % (view, n, len(frames)), flush=True)
    print('  view %s: %d 帧 -> %s' % (view, n, out_dir), flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-root', required=True, help='含 {view}/%06d.png 的目录')
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='01,03,04,07,09,14')
    ap.add_argument('--tag', default='h4d')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=10 ** 9)
    ap.add_argument('--imgsz', type=int, default=int(os.environ.get('DET_IMGSZ', '640')))
    ap.add_argument('--conf', type=float, default=float(os.environ.get('DET_CONF', '0.15')))
    ap.add_argument('--device', type=int, default=int(os.environ.get('DET_DEVICE', '0')))
    ap.add_argument('--W', type=int, default=3840)
    ap.add_argument('--H', type=int, default=2160)
    a = ap.parse_args()

    rtd = load_original()
    # ---- 只覆盖模块级常量，不改函数体 ----
    rtd.MODEL_PATH = os.environ.get('DET_MODEL', B + '/port/weights/rtdetr-l.pt')
    rtd.ROI_JSON = os.environ.get('DET_ROI', B + '/empty_roi.json')   # 空 {} -> 该视角不过滤
    rtd.OUTPUT_ROOT = a.out
    rtd.IMG_SIZE = a.imgsz
    rtd.PERSON_CONF = a.conf
    rtd.DEVICE = a.device
    assert os.path.exists(rtd.MODEL_PATH), rtd.MODEL_PATH
    assert os.path.exists(rtd.ROI_JSON), rtd.ROI_JSON
    print('[detect_h4d] imgsz=%d conf=%.2f device=%d model=%s'
          % (rtd.IMG_SIZE, rtd.PERSON_CONF, rtd.DEVICE, os.path.basename(rtd.MODEL_PATH)), flush=True)

    from ultralytics import RTDETR
    model = RTDETR(rtd.MODEL_PATH)
    views = [v.zfill(2) for v in a.views.split(',')]
    tot = 0
    for v in views:
        fd = os.path.join(a.frames_root, v)
        if not os.path.isdir(fd):
            print('  跳过 view %s（无帧目录）' % v); continue
        tot += process_view(rtd, model, v, fd, os.path.join(a.out, v), a.tag,
                            a.start, a.end, a.W, a.H)
    print('[OK] 共写 %d 个 LabelMe json -> %s' % (tot, a.out))


if __name__ == '__main__':
    main()

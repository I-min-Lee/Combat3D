# 环境兼容垫片(py3.12/numpy2 适配), 不改官方任何文件
# ★ 来源：照抄 westc fit 环境 (`/root/autodl-tmp/emoff/env/lib/python3.12/site-packages/sitecustomize.py`)
#   你那边已经用它把 chumpy 0.70 + numpy 2.x 跑通了（EasyMocap fit 依赖 chumpy 读 smpl.pkl）。
#   放到目标环境的 site-packages/ 下即可被 Python 启动时自动加载（site.py 会 import sitecustomize）。
#
# 解决的问题（2026-09-29 实测）：
#   1. chumpy 0.70 用 `inspect.getargspec`（py3.11 已删）→ 崩
#   2. chumpy `from numpy import bool, int, ...`（numpy 2 已删这些别名）→ 崩
#   3. EasyMocap 写 json 时直接 dump numpy 标量/数组 → TypeError
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
import numpy as _np
for _n, _o in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("unicode", str), ("str", str), ("alltrue", all),
               ("sometrue", any), ("product", _np.prod), ("cumproduct", _np.cumprod),
               ("round_", round), ("NAN", float("nan")), ("Infinity", float("inf"))]:
    if not hasattr(_np, _n):
        setattr(_np, _n, _o)

# json 序列化兼容: ndarray/numpy标量 (官方 write_all/save_annot 会直接 dump numpy)
import json as _json
_ed = _json.JSONEncoder.default
def _default(self, o):
    import numpy as _n
    if isinstance(o, _n.ndarray): return o.tolist()
    if isinstance(o, (_n.floating, _n.integer, _n.bool_)): return o.item()
    return _ed(self, o)
_json.JSONEncoder.default = _default

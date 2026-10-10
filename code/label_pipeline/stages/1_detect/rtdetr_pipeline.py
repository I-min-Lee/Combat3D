# -*- coding: utf-8 -*-
"""
长兵对抗 · 人框检测流水线 v2 (无颜色依赖, 纯几何双槽)
================================================================
改动(vs 固化版 2026-09-06):
  - 去掉 每框HSV颜色分类投票 / 同色union合并 / UNK丢框 / 25帧颜色AUDIT对调
  - 双槽匹配升级为 带速度预测(Kalman-lite): 两人交错/近身时各自沿轨迹穿行, 不抢号
  - 身份 = 纯几何连续(id0/id1), 与颜色完全解耦
  - ROI 多边形缺失的视角(如 view2)自动跳过过滤
  - 颜色只在写 LabelMe 时给 description 用(显示层可忽略)

输出: LabelMe JSON, group_id=0/1 (person id), description color:red/black(仅标注用途)
用法: python rtdetr_pipeline.py <视频或目录> [--max-frames N] [--render]
"""
import os, sys, argparse, json, cv2, numpy as np
from ultralytics import RTDETR

# ============================ 参数 ============================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rtdetr-l.pt")
ROI_JSON   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_polygons.json")
OUTPUT_ROOT= os.path.join(os.path.dirname(os.path.abspath(__file__)), "输出")

PERSON_CONF = 0.15
IMG_SIZE    = 640
DEVICE      = 0
ROI_TOL     = 10

# 双槽跟踪 (速度预测版)
EMA_A      = 0.35          # 槽位框 EMA(匹配后校正强度)
VEL_A      = 0.30          # 速度学习率
VEL_MAX    = 45.0          # 速度上限 px/帧 (200fps)
VEL_DECAY  = 0.90          # 未匹配帧 速度衰减(惯性 coast)
MAX_MISS   = 20            # coast 最大帧, 超过清空槽
MATCH_GATE = 160.0         # 匹配距离下限 px
MATCH_GATE_R = 0.9         # 匹配距离 = max(GATE, r*槽对角)

BOX_OUT     = 0.02
TOP_EXTRA   = 0.05
BOTTOM_EXTRA= 0.03
LABELME_VERSION = "5.3.1"
SKIP_EXISTING = True
# ================================================================

poly_cache = {}

def load_poly(view):
    if view not in poly_cache:
        data = json.load(open(ROI_JSON))
        poly_cache[view] = None if str(view) not in data else np.array(data[str(view)], np.float32)
    return poly_cache[view]

def roi_keep(boxes, view):
    poly = load_poly(view)
    if poly is None:
        return boxes                                  # 该视角无标定多边形 -> 不过滤
    return [b for b in boxes if cv2.pointPolygonTest(poly, ((b[0]+b[2])/2, float(b[3])), True) >= -ROI_TOL]

def candidates(img, model, view):
    """rtdetr 检测 -> ROI -> 面积最大2框, 一人一框, 无颜色逻辑"""
    r = model.predict(img, conf=PERSON_CONF, imgsz=IMG_SIZE, classes=[0],
                      device=DEVICE, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    boxes = roi_keep([b[:4].cpu().numpy() for b in r.boxes.xyxy], view)
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes[:2]                                  # 双人对抗, 至多两人

# ---------------- 双槽跟踪(速度预测, 无颜色) ----------------
class DSlot:
    def __init__(self, name):
        self.name, self.box, self.miss, self.life = name, None, 0, 0
        self.vel = np.array([0.0, 0.0])               # (vx,vy) px/帧
    def cen(self):
        return np.array([(self.box[0]+self.box[2])/2, (self.box[1]+self.box[3])/2])
    def diag(self):
        return float(np.hypot(self.box[2]-self.box[0], self.box[3]-self.box[1]))
    def pred_box(self):
        """按速度外推一帧的预测框"""
        return [self.box[0]+self.vel[0], self.box[1]+self.vel[1],
                self.box[2]+self.vel[0], self.box[3]+self.vel[1]]

class TwoSlotTracker:
    """全场最多两个几何 id(id0/id1), 带速度预测, 交错不换人"""
    def __init__(self):
        self.s0, self.s1 = DSlot("P0"), DSlot("P1")
        self.slots = [self.s0, self.s1]
        self.frame = 0
    def _act(self, s):
        return s.box is not None and s.miss <= MAX_MISS
    def step(self, cands):
        self.frame += 1
        cands = list(cands)
        act = [s for s in self.slots if self._act(s)]
        # 2) NO-SWAP 偏好分区: 每个候选声明它"最愿意跟"的槽(距其预测中心最近),
        #    每个槽只收"声明自己"的候选里最近的那个 —— 禁止抢别人已绑定的对象
        pref = {}                                     # slot -> best (ci, dist)
        for ci in range(len(cands)):
            b = cands[ci]
            pc = np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])
            best_s, bd = None, 1e18
            for s in act:
                d = float(np.linalg.norm(pc - (s.cen()+s.vel)))
                if d < bd:
                    bd, best_s = d, s
            if best_s is None:
                continue
            if bd > max(MATCH_GATE, MATCH_GATE_R*best_s.diag()):
                continue
            cur = pref.get(best_s)
            if cur is None or bd < cur[1]:
                pref[best_s] = (ci, bd)
        # 3) 匹配成功: 速度学习 + EMA 校正(只更新各自绑定对象, 绝不换人)
        for s, (ci, _) in pref.items():
            b = cands[ci]
            oldc = s.cen()
            pred = s.pred_box()
            newb = [pred[k] + EMA_A*(b[k]-pred[k]) for k in range(4)]
            s.box = newb
            s.miss, s.life = 0, s.life + 1
            v = (np.array([(newb[0]+newb[2])/2, (newb[1]+newb[3])/2]) - oldc)
            s.vel = np.clip(VEL_A*v + (1-VEL_A)*s.vel, -VEL_MAX, VEL_MAX)
        # 4) 未匹配的活跃槽: coast(按速度惯性), 超时清空 —— 期间身份锁住
        for s in act:
            if s not in pref:
                s.miss += 1
                if s.miss > MAX_MISS:
                    s.box = None; s.vel = np.array([0.0, 0.0])
                else:
                    s.box = s.pred_box()
                    s.vel = np.clip(s.vel*VEL_DECAY, -VEL_MAX, VEL_MAX)
        # 5) 没人要的候选: 只允许填空槽(身份空白), 不顶替在位者
        taken = {ci for ci, _ in pref.values()}
        for ci in range(len(cands)):
            if ci in taken:
                continue
            b = cands[ci]
            empty = [s for s in self.slots if s.box is None]
            if not empty:
                continue
            s = empty[0]
            s.box, s.miss, s.vel = list(b), 0, np.array([0.0, 0.0])
            s.life += 1
        return [s for s in self.slots if self._act(s)]

# ---------------- 输出 ----------------
def expand_head(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    w, h = x2-x1, y2-y1
    x1 = max(0, x1 - w*BOX_OUT); y1 = max(0, y1 - h*(BOX_OUT+TOP_EXTRA))
    x2 = min(img_w, x2 + w*BOX_OUT); y2 = min(img_h, y2 + h*(BOX_OUT+BOTTOM_EXTRA))
    return x1, y1, x2, y2

def parse_video_name(name):
    import re
    m = re.match(r"^(\d+)\.(\d+)_Miqus_(\d+)_\d+\.avi$", name, re.IGNORECASE)
    return m.groups() if m else None

def build_labelme(img_name, img_h, img_w, shapes):
    return {"version": LABELME_VERSION, "flags": {}, "shapes": shapes,
            "imagePath": img_name, "imageData": None,
            "imageHeight": img_h, "imageWidth": img_w}

def process_video(video_path, model, max_frames=0, render=False, start=0, end=0):
    name = os.path.basename(video_path)
    parsed = parse_video_name(name)
    if not parsed:
        print(f"[跳过] 文件名不符合规则: {name}"); return 0
    x, y, z = parsed
    json_dir = os.path.join(OUTPUT_ROOT, x, y, "json", z)
    os.makedirs(json_dir, exist_ok=True)
    video_base = os.path.splitext(name)[0]
    if SKIP_EXISTING:
        existing = [f for f in os.listdir(json_dir)
                    if f.startswith(video_base+"_") and f.lower().endswith(".json")]
        if existing:
            print(f"[已处理] {name} 已有 {len(existing)} 条, 跳过"); return 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 打不开 {video_path}"); return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w, img_h = int(cap.get(3)), int(cap.get(4))
    print(f"\n===== {name} | 场{x}半场{y}视角{z} | {total}帧 {img_w}x{img_h} =====", flush=True)
    tr = TwoSlotTracker()
    vw = None
    if render:
        vw = cv2.VideoWriter(os.path.join(OUTPUT_ROOT, f"{video_base}_annot.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 200, (int(img_w*0.8), int(img_h*0.8)))
    n = saved = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if end and n >= end:
            break
        if n < start:
            n += 1
            continue
        if max_frames and n >= max_frames:
            break
        objs = tr.step(candidates(fr, model, z))
        shapes = []
        for o in objs:
            gid = 0 if o.name == "P0" else 1
            desc = "color:red" if o.name == "P0" else "color:black"
            b = expand_head(o.box, img_w, img_h)
            shapes.append({"label": "person",
                           "points": [[float(b[0]), float(b[1])], [float(b[2]), float(b[3])]],
                           "group_id": gid, "description": desc,
                           "shape_type": "rectangle", "flags": {}})
        img_name = f"{video_base}_{n:06d}.jpg"
        data = build_labelme(img_name, img_h, img_w, shapes)
        with open(os.path.join(json_dir, f"{video_base}_{n:06d}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        saved += 1
        if render and vw is not None:
            for o in objs:
                x1, y1, x2, y2 = [int(v) for v in expand_head(o.box, img_w, img_h)]
                col = (255, 0, 255) if o.name == "P0" else (0, 255, 255)
                cv2.rectangle(fr, (x1, y1), (x2, y2), col, 2)
                cv2.putText(fr, o.name, (x1, max(y1-8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            vw.write(cv2.resize(fr, (int(img_w*0.8), int(img_h*0.8))))
        n += 1
        if n % 2000 == 0:
            print(f"  {n}/{total or '?'}", flush=True)
    cap.release()
    if vw is not None:
        vw.release()
    print(f"  done {n} 帧, 写 json {saved} 条 -> {json_dir}", flush=True)
    return saved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="单个 .avi 或含 .avi 的目录")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    args = ap.parse_args()
    model = RTDETR(MODEL_PATH)
    if os.path.isdir(args.input):
        avis = sorted(f for f in os.listdir(args.input) if f.lower().endswith(".avi"))
        print(f"批量 {len(avis)} 个视频")
        for a in avis:
            process_video(os.path.join(args.input, a), model, args.max_frames, args.render, args.start, args.end)
    else:
        process_video(args.input, model, args.max_frames, args.render, args.start, args.end)
    print("\n全部完成.")

if __name__ == "__main__":
    main()

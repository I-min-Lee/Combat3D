# -*- coding: utf-8 -*-
"""
视频连续检测（锚点版·批量）—— 完全独立脚本，不依赖 video_detect.py

流程：
  1. 开局多帧投票自动锁定红/黑方（红黑分类模型 + 颜色比例融合打分 + 位置聚类）
  2. 每帧 YOLO 检测 + 锚点联合最优分配（2×2 枚举，防双认领、防贴身 ID 互换）
  3. 位置绝对连续：候选与近 30 帧中位数参照比较，跳崖式变化一律拒绝（防路人顶替）
  4. 出画挂起：贴边丢失 → 挂起，回归需过颜色+面积+长宽比+距离四重校验；
     挂起超时转为幽灵锚点，重新认领只能在最后已知位置附近
  5. 检测不到就不标（无补框）；棍子经 近人/长度中位数/突变 三重校验

输入：单个 avi 视频，或装有 avi 的文件夹（批量模式自动连跑）
输出：LabelMe JSON，按 场次/半场/json/视角 分目录，与 video_detect.py 同格式

本地运行带预览窗口（P 暂停纠偏）；服务器上设 SHOW_PREVIEW=False 无界面运行。
"""
import os
import re
import cv2
import json
import math
import numpy as np

# ======================== 参数设置区（按需修改） ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ★ 输入：单个 avi 视频路径，或装有 avi 的文件夹路径
# 服务器运行：优先读环境变量 VIDEO_INPUT（如 VIDEO_INPUT=/data/xx.avi）；
# 未设置时默认"项目内比赛视频文件夹"（与 视频连续检测 同级），整个项目上传即可批量跑
VIDEO_INPUT = os.environ.get("VIDEO_INPUT") or "/root/autodl-tmp/all_videos"

# 输出根目录（脚本目录下的"输出"文件夹）
OUTPUT_ROOT = os.path.join(BASE_DIR, "输出")

# 模型路径（models 文件夹与本脚本同级）
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8s-pose.pt")
STICK_MODEL_PATH = os.path.join(BASE_DIR, "models", "staff_pose_best.pt")

DEVICE = "cuda"           # "cuda" 或 "cpu"
SHOW_PREVIEW = False      # 服务器/无界面环境 False；本地想开预览窗口改回 True

# 批量模式筛选（仅当 VIDEO_INPUT 是文件夹时生效）
TARGET_SESSIONS = ["1"]   # 只处理这些场次，如 ["1", "2"]；空列表 = 全部
EXCLUDE_VIEWS = [2]        # 排除这些视角，如 ["2"]；空列表 = 不排除
SKIP_EXISTING = True      # 已处理过的视频跳过（断点续跑）

# 置信度阈值（放宽保证召回，后续靠规则筛）
PERSON_CONF = 0.15
STICK_CONF = 0.15
IMG_SIZE = 640

MAX_PERSON_CANDIDATES = 6
MAX_STICKS = 2

# --- 突变过滤 ---
SHIFT_FACTOR = 0.35
MIN_SHIFT_PX = 50
AREA_MAX_RATIO = 2.0
AREA_MIN_RATIO = 0.5
STICK_LEN_MAX_RATIO = 1.8
STICK_LEN_MIN_RATIO = 0.45
STICK_ANGLE_THRESH = 60
STICK_CENTER_THRESH = 250
RECOVER_AFTER = 10        # 棍子专用：连续被判突变多少帧后接受并重置参照系

# --- 锚点分配 ---
COLOR_WEIGHT = 3.0        # 颜色匹配权重
DIST_WEIGHT = 2.0         # 距离惩罚权重
MAX_CLAIM_DIST = 1.5      # 正常认领最大距离（×参照框对角线，位置不会突变）
GHOST_CLAIM_DIST = 1.5    # 幽灵锚点重新认领的最大距离

# --- 丢失重找（遮挡期逐步扩大锚点搜索范围）---
LOST_EXPAND_RATE = 0.05      # 每连续丢失1帧，认领半径扩大比例（20帧后约×2）
LOST_EXPAND_MAX = 3.0        # 认领半径最大倍数（相对 MAX_CLAIM_DIST）
REDETECT_MIN_LOST = 10       # 连续丢失≥该帧数后找回：直接重建锚点，跳过突变过滤
AUTO_INIT_COLOR = 0.15    # 挂起回归的颜色门槛
AUTO_INIT_STRICT = 0.30   # 自动初始化严格颜色线（无棍共现时须达到）
AUTO_INIT_CONFIRM = 10    # 自动初始化：候选需连续该帧数都合格且位置稳定才锁定（防单帧误识别）
AUTO_INIT_MATCH = 0.5     # 连续帧间候选中心距 < 候选框对角线 × 该系数 → 视为同一候选
OUT_RESET_FRAMES = 400    # 挂起超过该帧数 → 转幽灵锚点等待回归（200fps 下 = 2 秒）
HIST_LEN = 30             # 中位数参照系的帧数

# --- 棍子校验 ---
STICK_NEAR_FACTOR = 1.2   # 棍中心到持有者距离 < 人框对角线 × 该系数
STICK_MED_MIN = 0.6
STICK_MED_MAX = 1.4
STICK_MED_WARMUP = 20

# --- 开局多帧投票锁定 ---
OPEN_FRAMES = 30          # 看前 N 帧投票
OPEN_MIN_VOTES = 3        # 当选所需最少票数
OPEN_AREA_W = 2.0         # 面积权重
OPEN_CENTER_W = 1.0       # 离画面中心距离权重

# --- 输出框 ---
BOX_OUT = 0.08            # 均匀外扩
TOP_EXTRA = 0.30          # 顶部额外外扩（YOLO 框常切头，只加长不加宽）
BOTTOM_EXTRA = 0.08       # 底部额外外扩
FALLBACK_HALF_W = 55      # 暂停纠偏点击的兜底框半宽
FALLBACK_UP = 160
FALLBACK_DOWN = 120

# --- 防红黑互换（贴身遮挡）---
AMBIG_IOU = 0.30          # 两候选框 IoU 超过该值 = 贴身歧义，启用硬颜色约束
COLOR_DIFF_MARGIN = 0.02  # 硬颜色约束下，红度须比另一色高出的最小差值
SIG_ALPHA = 0.10          # 槽位颜色签名 EMA 更新系数
SIG_WARMUP = 20           # 颜色签名至少累积帧数后才参与互换判定
SWAP_CONFIRM_FRAMES = 15  # 颜色签名持续异常多少帧后判定红黑互换并自愈

# --- 红黑分类模型（只负责红/黑方向，不负责"是不是运动员"）---
CLS_MODEL_PATH = os.path.join(BASE_DIR, "red_black_cls", "models", "red_black_cls.pt")
CLS_IMGSZ = 224            # 分类模型输入尺寸
CLS_CONF_HIGH = 0.80       # 模型置信度 >= 该值 → 用模型修正红黑方向；否则完全沿用旧版颜色逻辑
PLAYER_COLOR_MIN = 0.30    # 运动员颜色底线：开局投票时红/黑颜色占比达不到该值 → 不投（排除路人）

# --- 面积差严格模式（仅开局投票阶段生效，按视角启用）---
# 摄像机号 = 文件名 Miqus_ 后的数字（注意：解析出来是字符串，此处已按 int 匹配）
AREA_STRICT_VIEWS = [3, 4]                   # 启用该机制的摄像机号；空列表 = 全部不启用
AREA_MAX_RATIO_STRICT = {3: 2.5, 4: 2.5}     # 开局投票时允许的红黑面积比上限（路人通常小10倍左右，真运动员完整入镜时面积接近）
# ======================================================================

STICK_CLASS_NAMES = {0: "red_s", 1: "black_s"}
LABELME_VERSION = "5.3.1"

COL_RED = (0, 0, 255)
COL_BLACK = (60, 60, 60)
COL_PENDING = (0, 255, 255)


# ===================== 几何工具 =====================
def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_area(bbox):
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_diagonal(bbox):
    return math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])


def expand_bbox(bbox, ratio, img_w, img_h):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    dx, dy = w * ratio, h * ratio
    return [max(0, x1 - dx), max(0, y1 - dy), min(img_w, x2 + dx), min(img_h, y2 + dy)]


def expand_head(bbox, ratio, top_extra, img_w, img_h, bottom_extra=0.0):
    """均匀外扩 + 顶部/底部额外延伸（YOLO 框常切头脚，只加长不加宽）"""
    eb = expand_bbox(bbox, ratio, img_w, img_h)
    h = bbox[3] - bbox[1]
    eb[1] = max(0.0, eb[1] - h * top_extra)
    eb[3] = min(float(img_h), eb[3] + h * bottom_extra)
    return eb


def touches_edge(bbox, img_w, img_h, margin=10):
    """框是否贴到画面边缘（贴边丢失 → 判定出画）"""
    x1, y1, x2, y2 = bbox
    return x1 <= margin or y1 <= margin or x2 >= img_w - margin or y2 >= img_h - margin


def bbox_iou(a, b):
    """两框交集/并集比，用于判断贴身遮挡"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _area_limit(view):
    """该摄像机允许的红黑面积比上限（严格模式）；dict 则按摄像机取值"""
    if isinstance(AREA_MAX_RATIO_STRICT, dict):
        return float(AREA_MAX_RATIO_STRICT.get(int(view), 4.0))
    return float(AREA_MAX_RATIO_STRICT)


def stick_length(pts):
    return math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])


def stick_angle(pts):
    return math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))


def stick_center(pts):
    return ((pts[0][0] + pts[1][0]) / 2.0, (pts[0][1] + pts[1][1]) / 2.0)


def angle_diff(a1, a2):
    return abs((a1 - a2 + 180) % 360 - 180)


def center_dist_pts(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ===================== 颜色分数 =====================
def color_scores(img, bbox):
    """框中心 60% 区域的红色像素占比 / 暗像素占比，范围 [0,1]"""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    center = crop[int(ch * 0.2):int(ch * 0.8), int(cw * 0.2):int(cw * 0.8)]
    if center.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255])),
        cv2.inRange(hsv, np.array([165, 40, 40]), np.array([180, 255, 255])))
    redness = float(np.count_nonzero(mask_red)) / mask_red.size
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    blackness = float(np.count_nonzero(gray < 70)) / gray.size
    return redness, blackness


# ===================== 红黑分类模型（融合身份判断） =====================
_CLS_MODEL = None
_CLS_NAMES = None


def load_cls_model(path):
    """加载训练好的红/黑人物分类模型（ultralytics .pt）"""
    global _CLS_MODEL, _CLS_NAMES
    from ultralytics import YOLO
    _CLS_MODEL = YOLO(path)
    _CLS_NAMES = list(_CLS_MODEL.names.values())
    if "red" not in _CLS_NAMES or "black" not in _CLS_NAMES:
        raise ValueError(f"分类模型类别应为 red/black，实际为: {_CLS_NAMES}")


def _classify_bbox(img, bbox):
    """分类模型对人物框返回 (红概率, 黑概率)；未加载或无法裁剪时返回 None"""
    if _CLS_MODEL is None:
        return None
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    res = _CLS_MODEL.predict(crop, imgsz=CLS_IMGSZ, verbose=False)[0]
    probs = res.probs.data.cpu().numpy()
    out = {name: float(probs[i]) for i, name in enumerate(_CLS_NAMES)}
    return out.get("red", 0.5), out.get("black", 0.5)


def identity_scores(img, bbox, cls_map=None):
    """
    身份分数 = 旧版颜色比例逻辑，分类模型只修正红黑方向：
      - "是不是运动员"完全由旧版颜色门槛负责（open_score 0.15 /
        auto_init_slot 0.30 / joint_pick 的距离+颜色打分），模型不参与；
      - 仅当模型置信度 >= CLS_CONF_HIGH 时，把该框的方向强制改成模型判断
        （红方背对镜头、颜色分不出时由模型纠偏），并把另一方向的分数清零；
      - 分数永远是"该框本色"的颜色占比（红槽看红度、黑槽看黑度），
        绝不用颜色最大值顶替，避免路人被模型误判时拿到虚高分。
    cls_map: 本帧预计算的 {框(int元组): (红概率, 黑概率)}，避免重复推理。
    """
    r_c, b_c = color_scores(img, bbox)
    if cls_map is not None:
        m = cls_map.get(tuple(map(int, bbox)))
    else:
        m = _classify_bbox(img, bbox)
    if m is None:
        return r_c, b_c
    r_m, b_m = m
    conf = max(r_m, b_m)
    if conf < CLS_CONF_HIGH:
        return r_c, b_c   # 模型没把握 → 完全沿用旧版颜色逻辑
    if r_m >= b_m:
        return r_c, 0.0      # 模型判红 → 红分=本色红度，黑分=0
    return 0.0, b_c          # 模型判黑 → 黑分=本色黑度，红分=0


# ===================== 突变检查 =====================
def is_person_spike(ref_bbox, new_bbox):
    """与中位数参照比较：位置/面积跳崖式变化 → True（一律拒绝，无恢复）"""
    diag = bbox_diagonal(ref_bbox)
    shift_thresh = max(MIN_SHIFT_PX, diag * SHIFT_FACTOR)
    cd = center_dist_pts(bbox_center(ref_bbox), bbox_center(new_bbox))
    if cd > shift_thresh:
        return True
    area_ref = bbox_area(ref_bbox)
    if area_ref > 0:
        ratio = bbox_area(new_bbox) / area_ref
        if ratio > AREA_MAX_RATIO or ratio < AREA_MIN_RATIO:
            return True
    return False


def is_stick_spike(prev_pts, new_pts):
    prev_len = stick_length(prev_pts)
    if prev_len > 0:
        ratio = stick_length(new_pts) / prev_len
        if ratio > STICK_LEN_MAX_RATIO or ratio < STICK_LEN_MIN_RATIO:
            return True
    if angle_diff(stick_angle(prev_pts), stick_angle(new_pts)) > STICK_ANGLE_THRESH:
        return True
    if center_dist_pts(stick_center(prev_pts), stick_center(new_pts)) > STICK_CENTER_THRESH:
        return True
    return False


# ===================== YOLO 推理 =====================
def detect_persons(person_model, img):
    res = person_model.predict(img, conf=PERSON_CONF, classes=[0],
                               imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]
    boxes = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes.xyxy.cpu().numpy():
            boxes.append([float(v) for v in b])
    boxes.sort(key=bbox_area, reverse=True)
    return boxes[:MAX_PERSON_CANDIDATES]


def detect_sticks(stick_model, img):
    res = stick_model.predict(img, conf=STICK_CONF, imgsz=IMG_SIZE,
                              device=DEVICE, verbose=False)[0]
    sticks = []
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None or len(res.keypoints.xy) == 0:
        return sticks
    xy = res.keypoints.xy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    for i in range(len(xy)):
        pts = xy[i]
        if np.isnan(pts).any() or len(pts) < 2:
            continue
        label = STICK_CLASS_NAMES.get(int(cls_ids[i]), str(cls_ids[i]))
        sticks.append((label, [[float(pts[0][0]), float(pts[0][1])],
                               [float(pts[1][0]), float(pts[1][1])]],
                       float(confs[i])))
    sticks.sort(key=lambda x: x[2], reverse=True)
    return [(s[0], s[1]) for s in sticks[:MAX_STICKS]]


# ===================== 棍子分配 =====================
def assign_sticks_to_persons(sticks, red_center, black_center):
    """棍中心离谁近就跟谁；2 根同色时按距离纠正为一红一黑"""
    if not sticks:
        return []
    person_centers = {}
    if red_center is not None:
        person_centers["red"] = red_center
    if black_center is not None:
        person_centers["black"] = black_center
    if not person_centers:
        return sticks

    assigned = []
    for yolo_label, pts in sticks:
        sc = stick_center(pts)
        best_color, best_dist = None, float('inf')
        for color, pc in person_centers.items():
            d = center_dist_pts(sc, pc)
            if d < best_dist:
                best_dist, best_color = d, color
        assigned.append(("red_s" if best_color == "red" else "black_s", pts))

    if len(assigned) == 2 and assigned[0][0] == assigned[1][0]:
        (_, p0), (_, p1) = assigned
        c0, c1 = stick_center(p0), stick_center(p1)
        if red_center is not None and black_center is not None:
            if center_dist_pts(c0, red_center) < center_dist_pts(c1, red_center):
                assigned = [("red_s", p0), ("black_s", p1)]
            else:
                assigned = [("red_s", p1), ("black_s", p0)]
        elif red_center is not None:
            assigned = [("red_s", p0), ("black_s", p1)]
        else:
            assigned = [("black_s", p0), ("red_s", p1)]
    return assigned


# ===================== 锚点联合分配 =====================
def joint_pick(cands, img, red_slot, black_slot, red_gate=0.0, black_gate=0.0,
               red_strict=False, black_strict=False, hard_color=False, cls_map=None,
               red_scale=1.0, black_scale=1.0):
    """
    红黑槽位 2×2 联合最优分配：枚举全部组合取总分最高。
    防止双认领同一人和贴身 ID 互换。
    gate: 挂起回归的颜色门槛；strict: 挂起回归的严格形态校验（面积/长宽比/近距离）。
    hard_color: 贴身歧义时启用硬颜色约束（红槽必须红度>黑度，黑槽必须黑度>红度）。
    """
    def score(bbox, slot, color, gate, strict, hard, scale):
        diag = max(bbox_diagonal(slot), 1.0)
        d = center_dist_pts(bbox_center(bbox), bbox_center(slot)) / diag
        if d > (1.0 if strict else MAX_CLAIM_DIST) * scale:
            return None
        r, b = identity_scores(img, bbox, cls_map)
        col = r if color == "red" else b
        other = b if color == "red" else r
        if col < gate:
            return None
        if hard and col - other < COLOR_DIFF_MARGIN:
            return None
        if strict:
            ar = bbox_area(bbox) / max(bbox_area(slot), 1.0)
            if ar < 0.5 or ar > 2.0:
                return None
            w_n, h_n = max(bbox[2] - bbox[0], 1.0), max(bbox[3] - bbox[1], 1.0)
            w_o, h_o = max(slot[2] - slot[0], 1.0), max(slot[3] - slot[1], 1.0)
            aspect_ratio = (w_n / h_n) / (w_o / h_o)
            if aspect_ratio < 1 / 1.5 or aspect_ratio > 1.5:
                return None
        return COLOR_WEIGHT * col - DIST_WEIGHT * d

    red_opts = ([None] + list(cands)) if red_slot is not None else [None]
    black_opts = ([None] + list(cands)) if black_slot is not None else [None]

    best_total, best_pair = -1e9, (None, None)
    for rb in red_opts:
        sr = score(rb, red_slot, "red", red_gate, red_strict, hard_color, red_scale) if rb is not None else 0.0
        if sr is None:
            continue
        for bb in black_opts:
            if bb is not None and bb is rb:
                continue
            sb = score(bb, black_slot, "black", black_gate, black_strict, hard_color, black_scale) if bb is not None else 0.0
            if sb is None:
                continue
            if sr + sb > best_total:
                best_total, best_pair = sr + sb, (rb, bb)
    return best_pair


def auto_init_slot(cands, img, color, other_slot, ghost=None, cls_map=None):
    """
    为未初始化的槽位自动找目标。防路人三重校验：
      1. 位置连续：有幽灵锚点（最后已知位置）时，候选必须在其附近
      2. 严格颜色线（防背景色污染）
      3. 离另一槽位足够远
    ghost: 槽位最后已知的框；None 表示本视频从未锁定过（全场可认领）
    """
    best, best_col = None, -1
    for bbox in cands:
        r, b = identity_scores(img, bbox, cls_map)
        col = r if color == "red" else b
        other = b if color == "red" else r
        if col < AUTO_INIT_COLOR:
            continue
        if ghost is not None:
            gd = center_dist_pts(bbox_center(bbox), bbox_center(ghost))
            if gd > GHOST_CLAIM_DIST * max(bbox_diagonal(ghost), 1.0):
                continue  # 离最后已知位置太远 → 不可能是同一个人
        if col < AUTO_INIT_STRICT:
            continue
        # 贴身对抗时两运动员的框会重叠：若模型已高置信认定方向（另一色分数=0），
        # 说明该框不可能是另一槽位那个人 → 跳过"离另一槽位够远"门槛；
        # 模型没把握（另一色仍有分数）时保留该门槛，防止把同一人认成双方。
        if other_slot is not None and other > 0:
            d = center_dist_pts(bbox_center(bbox), bbox_center(other_slot))
            if d < bbox_diagonal(other_slot):
                continue
        if col > best_col:
            best_col, best = col, bbox
    return best


def nearest_box_to_point(cands, point, max_dist=150):
    """暂停纠偏时，点击吸附到最近的候选框"""
    best, best_d = None, max_dist
    for bbox in cands:
        d = center_dist_pts(bbox_center(bbox), point)
        if d < best_d:
            best_d, best = d, bbox
    return best


def majority_box(boxes, cluster_dist=120):
    """投票聚类：位置相近的框归为一簇，返回票数最多簇的最新框和票数"""
    clusters = []
    for b in boxes:
        c = bbox_center(b)
        for rep, lst in clusters:
            if center_dist_pts(c, rep) < cluster_dist:
                lst.append(b)
                break
        else:
            clusters.append((c, [b]))
    if not clusters:
        return None, 0
    _, lst = max(clusters, key=lambda t: len(t[1]))
    return lst[-1], len(lst)


def median_box(hist):
    """最近若干帧框的逐坐标中位数——比单帧更稳的参照系"""
    return [float(v) for v in np.median(np.array(hist), axis=0)]


# ===================== 文件工具 =====================
def parse_video_name(filename):
    m = re.match(r"^(\d+)\.(\d+)_Miqus_(\d+)_\d+\.avi$", filename, re.IGNORECASE)
    return m.groups() if m else None


def build_labelme_json(img_name, img_h, img_w, shapes):
    return {"version": LABELME_VERSION, "flags": {}, "shapes": shapes,
            "imagePath": img_name, "imageData": None,
            "imageHeight": img_h, "imageWidth": img_w}


# ===================== 界面绘制 =====================
def draw_hud(img, info):
    vis = img.copy()
    bar = np.full((40, vis.shape[1], 3), (30, 30, 30), np.uint8)
    cv2.putText(bar, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, vis])


def draw_state(vis, red_box, black_box, sticks, pending, img_w, img_h, y_off=40):
    """画槽位框（画的是外扩后的输出框，所见即所得）、棍、待确认选点"""
    for box, col, name in ((red_box, COL_RED, "RED"), (black_box, COL_BLACK, "BLACK")):
        if box is None:
            continue
        b = expand_head(box, BOX_OUT, TOP_EXTRA, img_w, img_h, BOTTOM_EXTRA)
        cv2.rectangle(vis, (int(b[0]), int(b[1] + y_off)), (int(b[2]), int(b[3] + y_off)), col, 2)
        cv2.putText(vis, name, (int(b[0]), int(b[1] + y_off) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    for label, pts in sticks:
        col = COL_RED if label == "red_s" else COL_BLACK
        cv2.line(vis, (int(pts[0][0]), int(pts[0][1] + y_off)),
                 (int(pts[1][0]), int(pts[1][1] + y_off)), col, 2)
    if pending is not None:
        cv2.drawMarker(vis, (int(pending[0]), int(pending[1] + y_off)),
                       COL_PENDING, cv2.MARKER_CROSS, 24, 2)
    return vis


# ===================== 单视频处理 =====================
def process_video(video_path, person_model, stick_model, show_preview):
    name = os.path.basename(video_path)
    parsed = parse_video_name(name)
    if not parsed:
        print(f"[跳过] 文件名不符合规则：{name}")
        return 0
    x, y, z = parsed
    json_dir = os.path.join(OUTPUT_ROOT, x, y, "json", z)
    os.makedirs(json_dir, exist_ok=True)
    video_base = os.path.splitext(name)[0]

    if SKIP_EXISTING:
        existing = [f for f in os.listdir(json_dir)
                    if f.startswith(video_base + "_") and f.lower().endswith(".json")]
        if existing:
            print(f"[已处理] {name}（已有 {len(existing)} 条），跳过")
            return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频：{video_path}")
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_diag = float(np.hypot(img_w, img_h))
    center = (img_w / 2.0, img_h / 2.0)
    print(f"\n===== 处理: {name} =====")
    print(f"  场次{x} 半场{y} 视角{z} | 总帧数 {total} | 分辨率 {img_w}x{img_h}")

    # ---- 状态 ----
    red_slot = black_slot = None
    red_initialized = black_initialized = False
    red_out = black_out = False
    red_out_frames = black_out_frames = 0
    red_hist, black_hist = [], []
    prev_red_s = prev_black_s = None
    stick_spike_run = {"red_s": 0, "black_s": 0}
    stick_len_hist = {"red_s": [], "black_s": []}
    # 颜色签名（防红黑互换：维护每槽位红度/黑度 EMA）
    red_sig = {"r": 0.0, "b": 0.0, "n": 0}
    black_sig = {"r": 0.0, "b": 0.0, "n": 0}
    swap_suspect = 0
    pending = None
    cur_cands = []
    paused = False
    # 面积严格匹配：仅开局投票阶段生效
    area_strict_view = int(z) in AREA_STRICT_VIEWS
    # 自动初始化候选确认（连续多帧合格才锁定）
    red_pending = black_pending = None
    red_pending_n = black_pending_n = 0
    red_lost = black_lost = 0   # 连续未检出帧数（丢失重找用，逐步扩大认领半径）

    # 预览窗口（仅本地）
    state = {"click": None}
    if show_preview:
        cv2.namedWindow("manual_detect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("manual_detect", img_w, img_h + 40)  # 固定 1:1，点击坐标不偏移

        def on_mouse(event, mx, my, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["click"] = (mx, my - 40)

        cv2.setMouseCallback("manual_detect", on_mouse)

    # ---- 开局：多帧投票自动锁定 ----
    def open_score(fr, bbox, color, cls_map=None):
        r, b = identity_scores(fr, bbox, cls_map)
        col = r if color == "red" else b
        if col < PLAYER_COLOR_MIN:
            return None
        a = bbox_area(bbox) / (img_w * img_h)
        dc = center_dist_pts(bbox_center(bbox), center) / img_diag
        return 3.0 * col + OPEN_AREA_W * a - OPEN_CENTER_W * dc

    red_votes, black_votes = [], []
    last_frame = None
    for _ in range(OPEN_FRAMES):
        ret, fr = cap.read()
        if not ret:
            break
        last_frame = fr
        cands = detect_persons(person_model, fr)
        cls_map = {tuple(map(int, b)): _classify_bbox(fr, b) for b in cands}
        rb_best, rb_s = None, -1e9
        for bbox in cands:
            s = open_score(fr, bbox, "red", cls_map)
            if s is not None and s > rb_s:
                rb_s, rb_best = s, bbox
        bb_best, bb_s = None, -1e9
        for bbox in cands:
            if bbox is rb_best:
                continue
            s = open_score(fr, bbox, "black", cls_map)
            if s is not None and s > bb_s:
                bb_s, bb_best = s, bbox
        # 面积差严格模式（遮挡严重视角）：两候选面积差太大 → 小的一方视为路人，
        # 该帧只投大的，直到两个面积接近的运动员同时出现
        if (area_strict_view and rb_best is not None and bb_best is not None):
            a_r = bbox_area(rb_best)
            a_b = bbox_area(bb_best)
            if a_r > 0 and a_b > 0:
                ratio = a_r / a_b
                limit = _area_limit(z)
                if ratio > limit:
                    bb_best = None
                elif ratio < 1.0 / limit:
                    rb_best = None
        # 合并框防护：两候选高度重叠 → 该帧不投票（防同一框投成两个人）
        if (rb_best is not None and bb_best is not None
                and bbox_iou(rb_best, bb_best) > AMBIG_IOU):
            continue
        if rb_best is not None:
            red_votes.append(rb_best)
        if bb_best is not None:
            black_votes.append(bb_best)

    red_box, red_n = majority_box(red_votes)
    black_box, black_n = majority_box(black_votes)
    if red_box is not None and red_n >= OPEN_MIN_VOTES:
        red_slot, red_initialized = list(red_box), True
        red_hist.append(list(red_box))
        print(f"  开局锁定红方（{red_n} 票）")
    if black_box is not None and black_n >= OPEN_MIN_VOTES:
        black_slot, black_initialized = list(black_box), True
        black_hist.append(list(black_box))
        print(f"  开局锁定黑方（{black_n} 票）")
    if not red_initialized and not black_initialized:
        print("  开局未锁定任何一方，处理中将自动识别")

    if show_preview and last_frame is not None:
        vis = draw_hud(last_frame,
                       f"开局锁定: 红={'Y' if red_initialized else 'N'} 黑={'Y' if black_initialized else 'N'} | 即将开始 | Q退出")
        draw_state(vis, red_slot, black_slot, [], None, img_w, img_h)
        cv2.imshow("manual_detect", vis)
        if (cv2.waitKey(800) & 0xFF) in (ord('q'), ord('Q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            return 0
    state["click"] = None  # 清掉开局展示期间的误点击

    # ---- 处理循环 ----
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    saved = 0

    while True:
        k = -1
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            raw_persons = detect_persons(person_model, frame)
            raw_sticks = detect_sticks(stick_model, frame)
            cls_map = {tuple(map(int, b)): _classify_bbox(frame, b) for b in raw_persons}
            # 丢失重找：按"连续丢失帧数"逐步扩大该槽位的认领半径（封顶）
            red_scale = min(1.0 + LOST_EXPAND_RATE * red_lost, LOST_EXPAND_MAX)
            black_scale = min(1.0 + LOST_EXPAND_RATE * black_lost, LOST_EXPAND_MAX)
            red_searching = red_initialized and red_lost > 0
            black_searching = black_initialized and black_lost > 0

            # ---- 人体：锚点联合分配 + 位置绝对连续 + 出画挂起 ----
            current_red_bbox = None
            current_black_bbox = None

            red_det, black_det = joint_pick(
                raw_persons, frame,
                red_slot if red_initialized else None,
                black_slot if black_initialized else None,
                red_gate=AUTO_INIT_COLOR if (red_out or red_searching) else 0.0,
                black_gate=AUTO_INIT_COLOR if (black_out or black_searching) else 0.0,
                red_strict=(red_out or red_searching),
                black_strict=(black_out or black_searching),
                red_scale=red_scale, black_scale=black_scale,
                cls_map=cls_map)

            # 贴身歧义：两候选框高度重叠 → 用硬颜色约束重跑，宁可不标也不互换；
            # 单锚点期（另一方未初始化或挂起）同样启用硬颜色，防止黑槽抓住红方
            single_slot = (red_initialized != black_initialized) or (red_out != black_out)
            if ((red_det is not None and black_det is not None
                    and bbox_iou(red_det, black_det) > AMBIG_IOU)
                    or single_slot):
                red_det, black_det = joint_pick(
                    raw_persons, frame,
                    red_slot if red_initialized else None,
                    black_slot if black_initialized else None,
                    red_gate=AUTO_INIT_COLOR if (red_out or red_searching) else 0.0,
                    black_gate=AUTO_INIT_COLOR if (black_out or black_searching) else 0.0,
                    red_strict=(red_out or red_searching),
                    black_strict=(black_out or black_searching),
                    red_scale=red_scale, black_scale=black_scale,
                    cls_map=cls_map,
                    hard_color=True)

            # 红方槽位更新（跳崖式变化一律拒绝，不输出不更新）
            if red_initialized:
                if red_det is not None:
                    if red_lost >= REDETECT_MIN_LOST:
                        # 丢失较久后找回：位置已跳变，直接重建锚点（跳过突变过滤）
                        red_slot = list(red_det)
                        red_hist.clear()
                        red_hist.append(list(red_det))
                        current_red_bbox = red_det
                        red_out = False
                        red_out_frames = 0
                        print(f"  [帧{frame_idx}] 红方重新锁定（丢失 {red_lost} 帧后找回）")
                    else:
                        ref = median_box(red_hist) if red_hist else red_slot
                        if not is_person_spike(ref, red_det):
                            red_slot = list(red_det)
                            red_hist.append(red_slot)
                            if len(red_hist) > HIST_LEN:
                                red_hist.pop(0)
                            current_red_bbox = red_det
                            red_out = False
                            red_out_frames = 0
                if current_red_bbox is None:
                    if touches_edge(red_slot, img_w, img_h):
                        red_out = True
                        red_out_frames += 1
                        if red_out_frames > OUT_RESET_FRAMES:
                            # 挂起超时：解除初始化，保留幽灵锚点等待回归
                            red_initialized = False
                            red_out = False
                            red_out_frames = 0
                            red_lost = 0
                            print(f"  [帧{frame_idx}] 红方挂起超时，转为幽灵锚点等待回归")
                if red_initialized:
                    red_lost = 0 if current_red_bbox is not None else red_lost + 1
            else:
                det = auto_init_slot(raw_persons, frame, "red", black_slot,
                                     ghost=red_slot, cls_map=cls_map)
                if det is not None:
                    if (red_pending is not None
                            and center_dist_pts(bbox_center(det), bbox_center(red_pending))
                            < AUTO_INIT_MATCH * max(bbox_diagonal(red_pending), 1.0)):
                        red_pending_n += 1
                    else:
                        red_pending_n = 1
                    red_pending = list(det)
                    if red_pending_n >= AUTO_INIT_CONFIRM:
                        red_slot = list(det)
                        red_initialized = True
                        red_out = False
                        red_hist.clear()
                        red_hist.append(list(det))
                        current_red_bbox = det
                        red_pending = None
                        red_pending_n = 0
                        print(f"  [帧{frame_idx}] 红方自动初始化（连续 {AUTO_INIT_CONFIRM} 帧确认）")
                else:
                    red_pending = None
                    red_pending_n = 0

            # 黑方槽位更新（同上）
            if black_initialized:
                if black_det is not None:
                    if black_lost >= REDETECT_MIN_LOST:
                        # 丢失较久后找回：位置已跳变，直接重建锚点（跳过突变过滤）
                        black_slot = list(black_det)
                        black_hist.clear()
                        black_hist.append(list(black_det))
                        current_black_bbox = black_det
                        black_out = False
                        black_out_frames = 0
                        print(f"  [帧{frame_idx}] 黑方重新锁定（丢失 {black_lost} 帧后找回）")
                    else:
                        ref = median_box(black_hist) if black_hist else black_slot
                        if not is_person_spike(ref, black_det):
                            black_slot = list(black_det)
                            black_hist.append(black_slot)
                            if len(black_hist) > HIST_LEN:
                                black_hist.pop(0)
                            current_black_bbox = black_det
                            black_out = False
                            black_out_frames = 0
                if current_black_bbox is None:
                    if touches_edge(black_slot, img_w, img_h):
                        black_out = True
                        black_out_frames += 1
                        if black_out_frames > OUT_RESET_FRAMES:
                            black_initialized = False
                            black_out = False
                            black_out_frames = 0
                            black_lost = 0
                            print(f"  [帧{frame_idx}] 黑方挂起超时，转为幽灵锚点等待回归")
                if black_initialized:
                    black_lost = 0 if current_black_bbox is not None else black_lost + 1
            else:
                det = auto_init_slot(raw_persons, frame, "black", red_slot,
                                     ghost=black_slot, cls_map=cls_map)
                if det is not None:
                    if (black_pending is not None
                            and center_dist_pts(bbox_center(det), bbox_center(black_pending))
                            < AUTO_INIT_MATCH * max(bbox_diagonal(black_pending), 1.0)):
                        black_pending_n += 1
                    else:
                        black_pending_n = 1
                    black_pending = list(det)
                    if black_pending_n >= AUTO_INIT_CONFIRM:
                        black_slot = list(det)
                        black_initialized = True
                        black_out = False
                        black_hist.clear()
                        black_hist.append(list(det))
                        current_black_bbox = det
                        black_pending = None
                        black_pending_n = 0
                        print(f"  [帧{frame_idx}] 黑方自动初始化（连续 {AUTO_INIT_CONFIRM} 帧确认）")
                else:
                    black_pending = None
                    black_pending_n = 0

            # ---- 颜色签名维护 + 红黑互换自愈 ----
            for color, box, sig in (("red", current_red_bbox, red_sig),
                                    ("black", current_black_bbox, black_sig)):
                if box is not None:
                    r, b = identity_scores(frame, box, cls_map)
                    if sig["n"] == 0:
                        sig["r"], sig["b"] = r, b
                    else:
                        sig["r"] += SIG_ALPHA * (r - sig["r"])
                        sig["b"] += SIG_ALPHA * (b - sig["b"])
                    sig["n"] += 1

            if (red_initialized and black_initialized
                    and current_red_bbox is not None and current_black_bbox is not None
                    and red_sig["n"] >= SIG_WARMUP and black_sig["n"] >= SIG_WARMUP):
                # 红槽长期不够红、黑槽长期不够黑 → 判定已互换，交换槽位历史自愈
                if (red_sig["r"] + COLOR_DIFF_MARGIN < red_sig["b"]
                        and black_sig["b"] + COLOR_DIFF_MARGIN < black_sig["r"]):
                    swap_suspect += 1
                    if swap_suspect >= SWAP_CONFIRM_FRAMES:
                        print(f"  [帧{frame_idx}] 检测到红黑互换，交换槽位历史自愈")
                        red_slot, black_slot = black_slot, red_slot
                        red_hist, black_hist = black_hist, red_hist
                        prev_red_s, prev_black_s = prev_black_s, prev_red_s
                        stick_len_hist["red_s"], stick_len_hist["black_s"] = (
                            stick_len_hist["black_s"], stick_len_hist["red_s"])
                        stick_spike_run["red_s"], stick_spike_run["black_s"] = (
                            stick_spike_run["black_s"], stick_spike_run["red_s"])
                        red_sig, black_sig = black_sig, red_sig
                        red_out, black_out = black_out, red_out
                        red_out_frames, black_out_frames = black_out_frames, red_out_frames
                        current_red_bbox, current_black_bbox = current_black_bbox, current_red_bbox
                        swap_suspect = 0
                else:
                    swap_suspect = 0

            # ---- 棍子：近人 + 长度中位数 + 突变 三重校验 ----
            red_center = bbox_center(red_slot) if red_initialized else None
            black_center = bbox_center(black_slot) if black_initialized else None
            assigned = assign_sticks_to_persons(raw_sticks, red_center, black_center)

            current_sticks = []
            for label, pts in assigned:
                anchor = red_slot if label == "red_s" else black_slot
                if anchor is not None:
                    max_d = bbox_diagonal(anchor) * STICK_NEAR_FACTOR
                    if center_dist_pts(stick_center(pts), bbox_center(anchor)) > max_d:
                        continue
                L = stick_length(pts)
                hist = stick_len_hist[label]
                if len(hist) >= STICK_MED_WARMUP:
                    med = float(np.median(hist))
                    if med > 0 and not (STICK_MED_MIN * med <= L <= STICK_MED_MAX * med):
                        continue
                prev = prev_red_s if label == "red_s" else prev_black_s
                if prev is not None:
                    if is_stick_spike(prev, pts):
                        stick_spike_run[label] += 1
                        if stick_spike_run[label] < RECOVER_AFTER:
                            continue
                        stick_spike_run[label] = 0
                    else:
                        stick_spike_run[label] = 0
                current_sticks.append((label, pts))
                hist.append(L)
                if len(hist) > 100:
                    hist.pop(0)
                if label == "red_s":
                    prev_red_s = [list(p) for p in pts]
                else:
                    prev_black_s = [list(p) for p in pts]

            # ---- 写 JSON ----
            shapes = []
            if current_red_bbox is not None:
                eb = expand_head(current_red_bbox, BOX_OUT, TOP_EXTRA, img_w, img_h, BOTTOM_EXTRA)
                shapes.append({
                    "label": "person",
                    "points": [[float(eb[0]), float(eb[1])], [float(eb[2]), float(eb[3])]],
                    "group_id": 0, "description": "color:red",
                    "shape_type": "rectangle", "flags": {},
                })
            if current_black_bbox is not None:
                eb = expand_head(current_black_bbox, BOX_OUT, TOP_EXTRA, img_w, img_h, BOTTOM_EXTRA)
                shapes.append({
                    "label": "person",
                    "points": [[float(eb[0]), float(eb[1])], [float(eb[2]), float(eb[3])]],
                    "group_id": 1, "description": "color:black",
                    "shape_type": "rectangle", "flags": {},
                })
            for label, pts in current_sticks:
                shapes.append({
                    "label": label,
                    "points": [[float(pts[0][0]), float(pts[0][1])],
                               [float(pts[1][0]), float(pts[1][1])]],
                    "group_id": None, "description": "",
                    "shape_type": "line", "flags": {},
                })
            img_name = f"{video_base}_{frame_idx:06d}.jpg"
            data = build_labelme_json(img_name, img_h, img_w, shapes)
            with open(os.path.join(json_dir, f"{video_base}_{frame_idx:06d}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            saved += 1

            # ---- 预览（可选） ----
            if show_preview:
                n_p = int(current_red_bbox is not None) + int(current_black_bbox is not None)
                vis = draw_hud(frame, f"处理中 {frame_idx}/{total} | 人:{n_p} 棍:{len(current_sticks)} | P暂停纠正 | Q退出")
                draw_state(vis, current_red_bbox, current_black_bbox,
                           current_sticks, pending, img_w, img_h)
                cv2.imshow("manual_detect", vis)
                k = cv2.waitKey(1) & 0xFF
            frame_idx += 1
            if frame_idx % 500 == 0:
                n_p = int(current_red_bbox is not None) + int(current_black_bbox is not None)
                print(f"  已处理 {frame_idx}/{total} 帧 | 本帧人:{n_p} 棍:{len(current_sticks)}")
        else:
            # ---- 暂停：人工纠偏（仅预览模式） ----
            vis = draw_hud(frame, "已暂停: 点击目标 → R(红)/B(黑) 重新指定 | P继续 | Q退出")
            draw_state(vis, red_slot, black_slot, [], pending, img_w, img_h)
            cv2.imshow("manual_detect", vis)
            k = cv2.waitKey(30) & 0xFF

        if show_preview:
            if state["click"] is not None:
                pending = state["click"]
                state["click"] = None
            if k == ord('p') or k == ord('P'):
                paused = not paused
                if paused:
                    cur_cands = detect_persons(person_model, frame)
                    print(f"  [帧{frame_idx}] 已暂停，可点击重新指定目标")
            elif (k == ord('r') or k == ord('R')) and pending is not None:
                box = nearest_box_to_point(cur_cands, pending)
                red_slot = list(box) if box is not None else [
                    pending[0] - FALLBACK_HALF_W, pending[1] - FALLBACK_UP,
                    pending[0] + FALLBACK_HALF_W, pending[1] + FALLBACK_DOWN]
                red_initialized = True
                red_hist.clear()
                red_hist.append(list(red_slot))
                red_out = False
                red_out_frames = 0
                red_pending = None
                red_pending_n = 0
                red_lost = 0
                print(f"  [帧{frame_idx}] 红方已重新指定 @ {pending}")
                pending = None
            elif (k == ord('b') or k == ord('B')) and pending is not None:
                box = nearest_box_to_point(cur_cands, pending)
                black_slot = list(box) if box is not None else [
                    pending[0] - FALLBACK_HALF_W, pending[1] - FALLBACK_UP,
                    pending[0] + FALLBACK_HALF_W, pending[1] + FALLBACK_DOWN]
                black_initialized = True
                black_hist.clear()
                black_hist.append(list(black_slot))
                black_out = False
                black_out_frames = 0
                black_pending = None
                black_pending_n = 0
                black_lost = 0
                print(f"  [帧{frame_idx}] 黑方已重新指定 @ {pending}")
                pending = None
            elif k == ord('c') or k == ord('C'):
                pending = None
            elif k in (ord('q'), ord('Q'), 27):
                break

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    print(f"  完成：{name} 处理 {frame_idx}/{total} 帧，输出 {saved} 个 JSON")
    return saved


# ===================== 入口 =====================
def main():
    for p in (PERSON_MODEL_PATH, STICK_MODEL_PATH):
        if not os.path.exists(p):
            print(f"[错误] 模型不存在：{p}")
            return

    # 加载红黑分类模型（身份判断用；失败则退回纯颜色逻辑）
    if not os.path.exists(CLS_MODEL_PATH):
        print(f"[警告] 分类模型不存在：{CLS_MODEL_PATH}，将退回纯颜色判断")
    else:
        try:
            load_cls_model(CLS_MODEL_PATH)
        except Exception as e:
            print(f"[警告] 分类模型加载失败：{e}，将退回纯颜色判断")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 输入：单文件或文件夹
    if os.path.isfile(VIDEO_INPUT):
        videos = [VIDEO_INPUT]
    elif os.path.isdir(VIDEO_INPUT):
        videos = [os.path.join(VIDEO_INPUT, f) for f in sorted(os.listdir(VIDEO_INPUT))
                  if f.lower().endswith(".avi")]
        if TARGET_SESSIONS or EXCLUDE_VIEWS:
            filtered = []
            for vp in videos:
                p = parse_video_name(os.path.basename(vp))
                if not p:
                    continue
                if TARGET_SESSIONS and p[0] not in TARGET_SESSIONS:
                    continue
                if int(p[2]) in EXCLUDE_VIEWS:
                    continue
                filtered.append(vp)
            videos = filtered
    else:
        print(f"[错误] 输入不存在：{VIDEO_INPUT}")
        return
    if not videos:
        print("未找到 avi 视频，请检查 VIDEO_INPUT 设置")
        return

    print(f"共 {len(videos)} 个视频待处理，加载模型...")
    from ultralytics import YOLO
    person_model = YOLO(PERSON_MODEL_PATH)
    stick_model = YOLO(STICK_MODEL_PATH)
    if DEVICE == "cuda":
        person_model.to("cuda")
        stick_model.to("cuda")

    total_saved = 0
    for vp in videos:
        total_saved += process_video(vp, person_model, stick_model, SHOW_PREVIEW)

    print(f"\n===== 全部完成 =====")
    print(f"共处理 {len(videos)} 个视频，累计输出 {total_saved} 个 JSON")
    print(f"输出目录：{os.path.abspath(OUTPUT_ROOT)}")


if __name__ == "__main__":
    main()

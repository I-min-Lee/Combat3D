import os, re, glob, cv2, json, collections

V = "/workshop/Lym/combat3d/all_videos"
names = os.listdir(V)
avi = [n for n in names if n.lower().endswith(".avi")]
print("总 avi:", len(avi))
pat = re.compile(r"^(\d+\.\d+)_Miqus_(\d+)_(\d+)\.avi$")
ok, bad = 0, []
per = collections.defaultdict(list)
for n in avi:
    m = pat.match(n)
    if m:
        ok += 1
        per[m.group(1)].append(int(m.group(2)))
    else:
        bad.append(n)
print("符合 <take>_Miqus_<view>_<id>.avi 的:", ok, " 不符合:", len(bad))
print("不符合样例:", bad[:6])
for t in ("1.2", "0.1", "12.3"):
    print(" ", t, sorted(per.get(t, [])))

# 取一个 take 测试抽帧
take = "1.2"
cands = sorted(per.get(take, []))
if cands:
    v = cands[0]
    f = glob.glob(f"{V}/{take}_Miqus_{v}_*.avi")[0]
    cap = cv2.VideoCapture(f)
    print("\n文件:", os.path.basename(f))
    print("  帧数:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
          " fps:", cap.get(cv2.CAP_PROP_FPS),
          " 分辨率:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 测试定位能力（是否逐帧精确 seek）
    for target in (0, 800, 8000):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        got = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        r, img = cap.read()
        print(f"  seek {target} -> POS_FRAMES={got} read={'ok' if r else 'fail'} shape={None if img is None else img.shape}")
    cap.release()

# 与场次 tag 的对应：take "1.2" 是否就是 tag f12
print("\n场次 tag 例:", sorted(set(re.sub(r"(\d+)\.(\d+)", lambda m: "f%s%s" % (m.group(1), m.group(2)), t) for t in per))[:12])

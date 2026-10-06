import json
for tag in ("f12", "f01", "f21", "f51"):
    try:
        d = json.load(open("/workshop/Lym/combat3d/rtd2/idflip_color_%s.json" % tag))
    except Exception as e:
        print(tag, "ERR", e)
        continue
    ks = sorted(int(k) for k in d)
    print("%s  keys n=%d range=[%d,%d]" % (tag, len(ks), ks[0], ks[-1]))
    for v in ("1", "3", "4", "7", "11"):
        seq = [d[str(k)].get(v) for k in ks]
        jumps = [ks[i] for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
        print("   v%-3s start=%s end=%s jumps=%s" % (v, seq[0], seq[-1], jumps[:10]))
    print()

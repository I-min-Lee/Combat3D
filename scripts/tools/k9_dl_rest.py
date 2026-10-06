#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""k9_dl_rest.py —— 补下 Harmony4D train 里还缺的 4 条序列
（02_grappling / 04_sword_part1 / 04_sword_part2 / 07_ballroom，约 119 GB）

走 hf-mirror；支持断点续传（.p2 分片，大小对得上才改名）；先列出仓库实际有的 zip 再决定。
"""
import os, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

H = {'User-Agent': 'Mozilla/5.0'}
BASE = 'https://hf-mirror.com/datasets/Jyun-Ting/Harmony4D/resolve/main'
OUT = '/workshop/Lym/combat3d/data/harmony4d/zips'
os.makedirs(OUT, exist_ok=True)

WANT_KEYS = ('02_grappling', '04_sword_part1', '04_sword_part2', '07_ballroom')

print('=== 本地已有 zip ===', flush=True)
have = sorted(f for f in os.listdir(OUT) if f.endswith('.zip'))
for f in have:
    print('  %-28s %.1f GB' % (f, os.path.getsize(os.path.join(OUT, f)) / 1024**3))

print('=== 仓库 tree ===', flush=True)
sizes = {}
for sub in ('train', 'test'):
    try:
        t = json.load(urllib.request.urlopen(
            urllib.request.Request(
                f'https://hf-mirror.com/api/datasets/Jyun-Ting/Harmony4D/tree/main/{sub}',
                headers=H), timeout=40))
        for e in t:
            if e.get('type') == 'file' and e.get('path', '').endswith('.zip'):
                sizes[e['path']] = e.get('size', 0)
    except Exception as e:
        print('  取尺寸失败', sub, str(e)[:90])


def fetch(path, size):
    name = path.split('/')[-1]
    dst = os.path.join(OUT, name)
    if os.path.exists(dst) and abs(os.path.getsize(dst) - size) < 4096:
        return name, os.path.getsize(dst), 'skip'
    part = dst + '.p2'
    for attempt in range(8):
        try:
            pos = os.path.getsize(part) if os.path.exists(part) else 0
            hdr = dict(H)
            if pos:
                hdr['Range'] = f'bytes={pos}-'
            req = urllib.request.Request(f'{BASE}/{path}', headers=hdr)
            with urllib.request.urlopen(req, timeout=180) as r, open(part, 'ab') as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            if abs(os.path.getsize(part) - size) < 4096:
                os.replace(part, dst)
                return name, os.path.getsize(dst), 'ok'
            raise RuntimeError(f'长度不符 {os.path.getsize(part)} vs {size}')
        except Exception as e:
            if attempt == 7:
                return name, os.path.getsize(part) if os.path.exists(part) else 0, \
                       f'FAIL {str(e)[:70]}'
            time.sleep(5 * (attempt + 1))


todo = [(p, s) for p, s in sizes.items() if s > 0 and any(k in p for k in WANT_KEYS)]
todo.sort(key=lambda x: x[1])
if not todo:
    print('!! 没在仓库里找到这 4 条（检查 tree 是否取到）')
else:
    print('计划下载 %d 个，合计 %.1f GB' % (len(todo), sum(s for _, s in todo) / 1024**3), flush=True)
    for p, s in todo:
        print('   %-28s %.1f GB' % (p, s / 1024**3))
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(fetch, p, s): p for p, s in todo}
        for fu in as_completed(futs):
            n, got, st = fu.result()
            print('  [%s] %-28s %.1f GB' % (st, n, got / 1024**3), flush=True)
print('DL_REST_DONE', flush=True)

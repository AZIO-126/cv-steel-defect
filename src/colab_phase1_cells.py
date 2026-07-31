# ===== CV Final — Phase 0/1/2 全流程（Colab，无需 Drive）=====
# 不挂 Drive：Colab 的 drive.mount() 在这个环境稳定报 mount failed (drive-timeout)，
# 而且挂 Drive 会让 notebook 不自包含 —— 别人跑就得先有同一个 Drive 文件夹。
# 原始数据是可再生的（Kaggle 一分钟拉完），所以放 /content 临时盘；
# 真正要留的产物（index.csv / splits / 图）体积很小，走 git。
import os, sys, time, zipfile, random, json, subprocess
import numpy as np, pandas as pd, requests
from google.colab import userdata

WORK = '/content/steel'
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
COMP = 'severstal-steel-defect-detection'
API  = 'https://www.kaggle.com/api/v1/competitions/data'
HDR  = {'Authorization': 'Bearer ' + userdata.get('KAGGLE_API_TOKEN')}
print('workdir:', WORK)

# ---- 规则门探测：data/list 返回 200 不代表能下载（列元数据不需要接受规则，下载需要）
r = requests.get(f'{API}/download/{COMP}/sample_submission.csv', headers=HDR, timeout=60)
assert r.status_code == 200 and not r.content.startswith(b'{"code":403'), r.content[:200]
print('rules gate: OK')

# ---- 下载 + 解压
ZIP = f'{WORK}/{COMP}.zip'
if not os.path.exists(f'{WORK}/train.csv'):
    if not os.path.exists(ZIP):
        t0, total = time.time(), 0
        with requests.get(f'{API}/download-all/{COMP}', headers=HDR, stream=True, timeout=3600) as resp:
            resp.raise_for_status()
            with open(ZIP, 'wb') as f:
                for chunk in resp.iter_content(1 << 22):
                    f.write(chunk); total += len(chunk)
                    if total % (1 << 28) < (1 << 22): print('  %.2f GB' % (total/1e9), flush=True)
        print('downloaded %.2f GB in %.0fs' % (total/1e9, time.time()-t0))
    with zipfile.ZipFile(ZIP) as z: z.extractall(WORK)
    print('unzipped')
print(sorted(os.listdir(WORK))[:6])

# ---- 几何确认（从真实文件读，不靠推断）
from PIL import Image
TR = f'{WORK}/train_images'
files = sorted(os.listdir(TR))
sizes = {}
for fn in random.Random(0).sample(files, 20):
    s = Image.open(f'{TR}/{fn}').size
    sizes[s] = sizes.get(s, 0) + 1
assert len(sizes) == 1, sizes
(W, H), = sizes
SHAPE = (H, W)
print('W x H =', (W, H), '| total px', W*H, '| rle shape (H,W) =', SHAPE)
assert W*H == 409600
print('train_images:', len(files))

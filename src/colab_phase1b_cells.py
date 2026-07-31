# ===== Phase 1 续 + Phase 2：RLE 验证 → index.csv → 冻结 split → EDA =====
# RLE 函数内联（repo 是 private，Colab 里 clone 要另外授权，内联更自包含）
def rle_decode(rle, shape):
    """Severstal RLE → 0/1 mask。两个约定：列主序(Fortran) + 起始位置 1-based。
    写错不报错，只会让 mask 悄悄转置/位移，等分割模型训完才发现。"""
    h, w = shape
    m = np.zeros(h * w, dtype=np.uint8)
    if rle is None or (isinstance(rle, float) and np.isnan(rle)) or not str(rle).strip():
        return m.reshape(shape, order='F')
    v = str(rle).split()
    st = np.asarray(v[0::2], dtype=np.int64) - 1
    ln = np.asarray(v[1::2], dtype=np.int64)
    for s, e in zip(st, st + ln):
        m[s:e] = 1
    return m.reshape(shape, order='F')

def rle_encode(mask):
    p = np.asarray(mask).flatten(order='F')
    if p.dtype != np.uint8: p = (p > 0).astype(np.uint8)
    pad = np.concatenate([[0], p, [0]])
    tr = np.flatnonzero(pad[1:] != pad[:-1]) + 1
    return ' '.join(f'{s} {l}' for s, l in zip(tr[0::2], tr[1::2] - tr[0::2]))

def defect_area_px(rle):
    if rle is None or (isinstance(rle, float) and np.isnan(rle)) or not str(rle).strip(): return 0
    return int(sum(int(x) for x in str(rle).split()[1::2]))

# ---- 真实数据往返测试（硬要求：RLE 解错不抛异常）
tr_df = pd.read_csv(f'{WORK}/train.csv')
print('train.csv:', tr_df.shape, list(tr_df.columns))
samp = tr_df.dropna(subset=['EncodedPixels']).sample(20, random_state=0)
ok = sum(rle_encode(rle_decode(r.EncodedPixels, SHAPE)) == r.EncodedPixels.strip()
         for r in samp.itertuples())
print(f'RLE round-trip on 20 real rows: {ok}/20', 'PASS' if ok == 20 else 'FAIL')
# 列主序专项：竖直相邻应连成一段，水平相邻应被 H 隔开
a = np.zeros(SHAPE, np.uint8); a[0,0] = a[1,0] = 1
b = np.zeros(SHAPE, np.uint8); b[0,0] = b[0,1] = 1
print('column-major check:', rle_encode(a) == '1 2', 'and', rle_encode(b) == f'1 {SHAPE[0]+1} 1'.replace('1 ','1 1 ',1))
assert ok == 20

# ---- index.csv：train.csv 只有缺陷行，无缺陷图只存在于磁盘，必须补进来
tr_df['area'] = tr_df['EncodedPixels'].map(defect_area_px)
rows = []
for iid, g in tr_df.groupby('ImageId', sort=True):
    per = {int(r.ClassId): int(r.area) for r in g.itertuples()}
    rows.append(dict(image_id=iid, has_defect=1, n_defect_classes=len(per),
                     class_ids='|'.join(map(str, sorted(per))),
                     primary_class=max(per, key=per.get),
                     defect_area_px=int(g['area'].sum()),
                     **{f'area_class_{c}': per.get(c, 0) for c in range(1,5)},
                     **{f'has_class_{c}': int(c in per) for c in range(1,5)}))
idx = pd.DataFrame(rows)
clean = sorted(set(files) - set(idx.image_id))
if clean:
    idx = pd.concat([idx, pd.DataFrame(dict(image_id=clean, has_defect=0, n_defect_classes=0,
        class_ids='', primary_class=0, defect_area_px=0,
        **{f'area_class_{c}': 0 for c in range(1,5)},
        **{f'has_class_{c}': 0 for c in range(1,5)}))], ignore_index=True)
idx = idx.sort_values('image_id').reset_index(drop=True)
idx.to_csv(f'{WORK}/index.csv', index=False)

n_def, n_clean = int((idx.has_defect==1).sum()), int((idx.has_defect==0).sum())
print(f'\nindex.csv: {len(idx)} rows | defect {n_def} | clean {n_clean} '
      f'({100*n_clean/len(idx):.1f}% clean)')
for c in range(1,5): print(f'  class {c}: {int(idx[f"has_class_{c}"].sum()):5d} images')
multi = int((idx.n_defect_classes>1).sum())
print(f'multi-class images: {multi}/{n_def} = {100*multi/n_def:.2f}% '
      f'-> head must be {"MULTI-LABEL (4 sigmoid)" if multi/n_def>0.02 else "single-label"}')

# ---- 冻结 split：按 (has_defect, primary_class) 联合分层
from sklearn.model_selection import train_test_split
strat = idx.has_defect.astype(str) + '_' + idx.primary_class.astype(str)
strat = strat.where(strat.map(strat.value_counts()) >= 2, 'rare')
a_, b_ = train_test_split(idx.image_id, test_size=0.2, random_state=42, stratify=strat)
os.makedirs(f'{WORK}/splits', exist_ok=True)
pd.DataFrame(dict(image_id=sorted(a_), split='train')).to_csv(f'{WORK}/splits/train.csv', index=False)
pd.DataFrame(dict(image_id=sorted(b_), split='val')).to_csv(f'{WORK}/splits/val.csv', index=False)
ii = idx.set_index('image_id')
pr = {k: 100*ii.loc[list(v)].has_defect.mean() for k, v in [('train', a_), ('val', b_)]}
print(f'\nsplit frozen: train {len(a_)} / val {len(b_)} | SEED=42')
print(f'defect prevalence  train {pr["train"]:.2f}%  val {pr["val"]:.2f}%  '
      f'gap {abs(pr["train"]-pr["val"]):.3f}pp', 'OK' if abs(pr['train']-pr['val'])<1 else 'FAIL')
for c in range(1,5):
    print(f'  class {c}: train {int(ii.loc[list(a_)][f"has_class_{c}"].sum()):5d}'
          f'  val {int(ii.loc[list(b_)][f"has_class_{c}"].sum()):5d}')

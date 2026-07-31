# ===== Phase 2 — EDA（评分表第 2 项，15 分）=====
# 逐条对应作业原文要求：
#   "Summarize data: record counts, missing values, and schema"
#   "Visualize the raw dataset using charts and tables"
#   "Image Data: Histograms, samples, outliers"
import os, random, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

FIGS = f'{WORK}/figs'; os.makedirs(FIGS, exist_ok=True)
idx = pd.read_csv(f'{WORK}/index.csv')
tr  = pd.read_csv(f'{WORK}/train.csv')
TR  = f'{WORK}/train_images'
CLS = [1,2,3,4]
plt.rcParams.update({'figure.dpi':110, 'font.size':9})

# ---------- (1) record counts / missing values / schema ----------
print('='*64); print('RECORD COUNTS'); print('='*64)
print(f'images on disk        : {len(os.listdir(TR))}')
print(f'index.csv rows        : {len(idx)}')
print(f'  with defect         : {int((idx.has_defect==1).sum())}')
print(f'  defect-free         : {int((idx.has_defect==0).sum())}')
print(f'train.csv rows        : {len(tr)}  (one row per defect INSTANCE)')
print('\nMISSING VALUES (train.csv)'); print(tr.isna().sum().to_string())
print('\nMISSING VALUES (index.csv)'); print(idx.isna().sum().sum(), 'total')
print('\nSCHEMA (index.csv)')
schema = pd.DataFrame({'dtype': idx.dtypes.astype(str), 'n_unique': idx.nunique(),
                       'example': [idx[c].iloc[0] for c in idx.columns]})
print(schema.to_string())
schema.to_csv(f'{FIGS}/schema_table.csv')

# ---------- (2) charts ----------
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

# a) per-class image counts — the 21:1 skew between defect classes
cnt = [int(idx[f'has_class_{c}'].sum()) for c in CLS]
b = ax[0,0].bar([f'class {c}' for c in CLS], cnt, color=['#4C72B0','#DD8452','#55A868','#C44E52'])
ax[0,0].bar_label(b, fmt='%d'); ax[0,0].set_title('Images per defect class (note class 3 vs 2 ≈ 21:1)')
ax[0,0].set_ylabel('images')

# b) defect vs defect-free — nearly balanced, which is the opposite of the usual assumption
vc = idx.has_defect.value_counts().sort_index()
ax[0,1].bar(['defect-free','has defect'], [vc.get(0,0), vc.get(1,0)], color=['#8C8C8C','#C44E52'])
for i,v in enumerate([vc.get(0,0), vc.get(1,0)]):
    ax[0,1].text(i, v, f'{v}\n{100*v/len(idx):.1f}%', ha='center', va='bottom')
ax[0,1].set_ylim(0, max(vc.get(0,0), vc.get(1,0)) * 1.18)   # headroom so labels clear the title
ax[0,1].set_title('Defect vs defect-free (53/47 — binary task is balanced)')

# c) defect area distribution per class (log) — feeds the small/large Dice split in phase 4
data = [np.log10(idx.loc[idx[f'has_class_{c}']==1, f'area_class_{c}'].clip(lower=1)) for c in CLS]
ax[1,0].violinplot(data, showmedians=True)
ax[1,0].set_xticks(CLS); ax[1,0].set_xticklabels([f'class {c}' for c in CLS])
ax[1,0].set_ylabel('log10(defect area px)')
ax[1,0].set_title('Defect area by class — class 2 is both rare AND small')

# d) co-occurrence heatmap — the evidence for a multi-label head
M = np.zeros((4,4), int)
for i,a in enumerate(CLS):
    for j,b2 in enumerate(CLS):
        M[i,j] = int(((idx[f'has_class_{a}']==1) & (idx[f'has_class_{b2}']==1)).sum())
im_ = ax[1,1].imshow(M, cmap='Blues')
for i in range(4):
    for j in range(4):
        ax[1,1].text(j, i, M[i,j], ha='center', va='center',
                     color='white' if M[i,j] > M.max()*0.5 else 'black', fontsize=8)
ax[1,1].set_xticks(range(4)); ax[1,1].set_xticklabels(CLS)
ax[1,1].set_yticks(range(4)); ax[1,1].set_yticklabels(CLS)
ax[1,1].set_title('Class co-occurrence (off-diagonal ⇒ multi-label)')
plt.tight_layout(); plt.savefig(f'{FIGS}/eda_structured.png', bbox_inches='tight'); plt.close()
print('\nsaved figs/eda_structured.png')

# ---------- (3) image data: histograms / samples / outliers ----------
rng = random.Random(0)
# grayscale histogram: global + per class (mean intensity per image, 150 sampled per group)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
allpx = []
for fn in rng.sample(sorted(os.listdir(TR)), 150):
    allpx.append(np.asarray(Image.open(f'{TR}/{fn}').convert('L')).ravel()[::37])
ax[0].hist(np.concatenate(allpx), bins=64, color='#4C72B0')
ax[0].set_title('Global grayscale intensity (150 sampled images)'); ax[0].set_xlabel('pixel value')
for c in CLS:
    ids = idx.loc[idx[f'has_class_{c}']==1, 'image_id'].tolist()
    means = [np.asarray(Image.open(f'{TR}/{i}').convert('L')).mean()
             for i in rng.sample(ids, min(120, len(ids)))]
    ax[1].hist(means, bins=28, alpha=0.5, label=f'class {c}')
ax[1].legend(); ax[1].set_title('Mean image intensity by defect class'); ax[1].set_xlabel('mean pixel')
plt.tight_layout(); plt.savefig(f'{FIGS}/eda_intensity.png', bbox_inches='tight'); plt.close()
print('saved figs/eda_intensity.png')

# sample grid: one representative per class, image + mask overlay
fig, axes = plt.subplots(4, 1, figsize=(15, 9))
for a, c in zip(axes, CLS):
    row = tr[(tr.ClassId==c) & tr.EncodedPixels.notna()].sample(1, random_state=1).iloc[0]
    img = np.array(Image.open(f'{TR}/{row.ImageId}').convert('RGB'))
    m = rle_decode(row.EncodedPixels, SHAPE)
    ov = img.copy(); ov[m==1] = [255,0,0]
    a.imshow(np.hstack([img, ov])); a.axis('off')
    a.set_title(f'class {c} — {row.ImageId} — defect area {int(m.sum())} px', fontsize=9)
plt.tight_layout(); plt.savefig(f'{FIGS}/eda_samples.png', bbox_inches='tight'); plt.close()
print('saved figs/eda_samples.png')

# outliers: darkest / brightest / largest / smallest defect
stats = []
for fn in rng.sample(sorted(os.listdir(TR)), 400):
    stats.append((fn, float(np.asarray(Image.open(f'{TR}/{fn}').convert('L')).mean())))
stats.sort(key=lambda t: t[1])
big = idx.loc[idx.has_defect==1].nlargest(1,'defect_area_px').image_id.iloc[0]
small = idx.loc[idx.has_defect==1].nsmallest(1,'defect_area_px').image_id.iloc[0]
picks = [(stats[0][0], f'darkest (mean {stats[0][1]:.1f})'),
         (stats[-1][0], f'brightest (mean {stats[-1][1]:.1f})'),
         (big, f'largest defect ({int(idx.set_index("image_id").loc[big].defect_area_px)} px)'),
         (small, f'smallest defect ({int(idx.set_index("image_id").loc[small].defect_area_px)} px)')]
fig, axes = plt.subplots(4, 1, figsize=(15, 9))
for a,(fn,lab) in zip(axes, picks):
    a.imshow(np.array(Image.open(f'{TR}/{fn}').convert('L')), cmap='gray'); a.axis('off')
    a.set_title(f'{lab} — {fn}', fontsize=9)
plt.tight_layout(); plt.savefig(f'{FIGS}/eda_outliers.png', bbox_inches='tight'); plt.close()
print('saved figs/eda_outliers.png')

print('\n' + '='*64)
print('EDA CONCLUSIONS (report-ready)')
print('='*64)
n_def = int((idx.has_defect==1).sum()); multi = int((idx.n_defect_classes>1).sum())
print(f'1. Binary has-defect is nearly balanced ({n_def} vs {len(idx)-n_def}, '
      f'{100*n_def/len(idx):.1f}% defective), so the binary task needs no resampling.')
print(f'2. The severe imbalance is BETWEEN defect classes: class 3 has {cnt[2]} images vs '
      f'class 2 with {cnt[1]} — a {cnt[2]/cnt[1]:.0f}:1 ratio. Weighting belongs here.')
print(f'3. {multi} of {n_def} defect images ({100*multi/n_def:.2f}%) carry more than one class, '
      f'so classification must be multi-label (4 sigmoids), not 4-way softmax.')
print('4. Class 2 is rare AND small-area, so it will be the hardest to segment; report '
      'per-class Dice split by defect area rather than one pooled number.')
print(f'5. Images are uniformly {SHAPE[1]}x{SHAPE[0]} wide strips, so square-crop pipelines '
      'would discard most of the frame; resize or tile along the width instead.')

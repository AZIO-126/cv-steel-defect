# ===== Phase 5 - deployment architecture figure =====
# Renders outputs/figs/deploy_architecture.png, the diagram required by
# phases/phase5/README.md step 1:
#   line camera -> frame grabber -> classifier triage -> (flagged only) segmentation
#   -> QA database + operator UI -> alarm / reject arm, with the ONNX serving
#   boundary and the shadow-mode gate on the actuation link made explicit.
# Pure matplotlib on purpose: no graphviz / draw.io dependency, so the figure
# regenerates from this committed script in any environment.
#   /opt/anaconda3/bin/python3 src/ops_arch_diagram.py
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, '..', 'outputs', 'figs')
os.makedirs(FIGS, exist_ok=True)
OUT = os.path.join(FIGS, 'deploy_architecture.png')

# muted palette: plain services vs learned models vs data/UI vs actuation
C_SERVICE = '#dfe6ec'
C_MODEL   = '#c9dcc4'
C_STORE   = '#e6dcc8'
C_ACT     = '#eccfc9'
EDGE      = '#4a5560'
GREY      = '#7a8896'
INK       = '#1f2933'

W, H = 16.0, 9.0
BW, BH = 2.85, 1.15              # box width / height
ROW1, ROW2 = 6.90, 2.60          # box centre-line y for the two rows
LANE = 4.55                      # y of the triage bypass lane
X1 = [0.35, 4.25, 8.15, 12.05]   # left edges, row 1 (camera -> segmentation)

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_axis_off()


def box(x, y, title, lines, color, w=BW, h=BH):
    """Rounded box with a bold title line and small detail lines under it."""
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                boxstyle='round,pad=0.06,rounding_size=0.12',
                                linewidth=1.2, edgecolor=EDGE, facecolor=color,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2 - 0.28, title, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=INK, zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 - 0.56 - 0.235 * i, ln, ha='center',
                va='center', fontsize=8.1, color=INK, zorder=4)
    return {'x0': x, 'x1': x + w, 'yc': y, 'y0': y - h / 2, 'y1': y + h / 2,
            'xc': x + w / 2}


def arrow(p0, p1, color=EDGE, style='-', rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=15,
                                 linewidth=lw, color=color, zorder=2,
                                 linestyle=style,
                                 connectionstyle=f'arc3,rad={rad}'))


def note(x, y, text, fs=8.2, color='#5a6673', ha='center', bold=False):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=color,
            zorder=5, fontweight='bold' if bold else 'normal')


def lat(bx, text):
    """Per-stage latency tag under a box, so the figure agrees with the latency
    budget section of report/04_model_ops.md."""
    ax.text(bx['xc'], bx['y0'] - 0.28, text, ha='center', va='center',
            fontsize=8.0, style='italic', color='#5a6673', zorder=5)


# ---------- row 1: acquisition and inference ----------
cam = box(X1[0], ROW1, 'Line camera',
          ['grayscale 1600 x 256 strip', 'fixed lighting, constant speed'],
          C_SERVICE)
grab = box(X1[1], ROW1, 'Frame grabber service',
           ['ring buffer, normalise', 'drop + log on overrun'], C_SERVICE)
clf = box(X1[2], ROW1, 'Classifier triage',
          ['ResNet-50, 4 sigmoids (ONNX)', 'has-defect score > tau'], C_MODEL)
seg = box(X1[3], ROW1, 'Segmentation model',
          ['U-Net / DeepLabV3+ (ONNX)', 'pixel contours, flagged only'], C_MODEL)

for a, b, lab, dx in ((cam, grab, 'frames', 0.0), (grab, clf, 'tensor', -0.20),
                      (clf, seg, 'flagged', 0.0)):
    arrow((a['x1'], ROW1), (b['x0'], ROW1))
    note((a['x1'] + b['x0']) / 2 + dx, ROW1 + 0.30, lab, fs=8.3, color=EDGE)

lat(cam, 'frame period 500-1000 ms')
lat(grab, 'grab + preprocess 15 ms')
lat(clf, 'classify 12 ms, every frame')
lat(seg, 'segment 45 ms, flagged only')

# ---------- serving boundary around the two learned models ----------
sx0, sx1 = clf['x0'] - 0.40, seg['x1'] + 0.40
sy0, sy1 = ROW1 - 1.45, ROW1 + BH / 2 + 0.42
ax.add_patch(FancyBboxPatch((sx0, sy0), sx1 - sx0, sy1 - sy0,
                            boxstyle='round,pad=0.02,rounding_size=0.14',
                            linewidth=1.3, edgecolor='#6b8e6b',
                            facecolor='none', linestyle=(0, (5, 3)), zorder=1))
note((sx0 + sx1) / 2, sy1 - 0.22,
     'inference server: ONNX opset 17, Triton, dynamic batching, one T4-class GPU',
     fs=9.0, color='#3f6b3f', bold=True)

# ---------- row 2: persistence, review, actuation (right to left) ----------
qadb = box(X1[3], ROW2, 'QA database',
           ['frame id, coil / grade, scores,', 'mask RLE, model version, shadow flag'],
           C_STORE)
oui = box(7.85, ROW2, 'Operator UI',
          ['mask overlay on live frame', 'operator confirms or clears'], C_STORE)
act = box(2.60, ROW2, 'Alarm / reject arm',
          ['light stack + PLC reject', 'actuation written back to QA DB'], C_ACT)

lat(qadb, 'write 8 ms')
lat(oui, 'refresh <= 200 ms')
lat(act, 'actuate 20 ms')

# segmentation results down into the QA database
SEGX, SEGY = qadb['x1'] - 0.55, 5.15
ax.plot([seg['xc'], seg['xc']], [seg['y0'] - 0.55, SEGY], color=EDGE, lw=1.5,
        zorder=2)
ax.plot([seg['xc'], SEGX], [SEGY, SEGY], color=EDGE, lw=1.5, zorder=2)
arrow((SEGX, SEGY), (SEGX, qadb['y1']))
note(SEGX - 0.18, 4.15, 'masks + scores', fs=8.3, color=EDGE, ha='right')

arrow((qadb['x0'], ROW2), (oui['x1'], ROW2))
note((qadb['x0'] + oui['x1']) / 2, ROW2 + 0.30, 'highlight', fs=8.3, color=EDGE)

# ---------- the triage bypass: the compute saving, labelled ----------
BYX = qadb['x0'] + 0.35
ax.plot([clf['xc'], clf['xc']], [sy0, LANE], color=GREY, lw=1.5, zorder=2)
ax.plot([clf['xc'], BYX], [LANE, LANE], color=GREY, lw=1.5, zorder=2)
arrow((BYX, LANE), (BYX, qadb['y1']), color=GREY)
note(clf['xc'] + 0.15, LANE + 0.26,
     'no defect: bypass segmentation', fs=9.0, color='#3d4c5a', ha='left',
     bold=True)
note(0.35, LANE + 0.26,
     'Triage saving: 47.0% of the labelled frames are defect-free (phase 1) and',
     fs=8.4, ha='left')
note(0.35, LANE - 0.02,
     'the live clean rate is normally higher, so the segmentation model runs on',
     fs=8.4, ha='left')
note(0.35, LANE - 0.30,
     'well under half the frames and the GPU is sized for that fraction.',
     fs=8.4, ha='left')

# ---------- shadow-mode gate on the actuation link ----------
gx = (oui['x0'] + act['x1']) / 2
ax.add_patch(Polygon([(gx, ROW2 + 0.42), (gx + 0.52, ROW2), (gx, ROW2 - 0.42),
                      (gx - 0.52, ROW2)], closed=True, facecolor='#f2e2b8',
                     edgecolor=EDGE, linewidth=1.2, zorder=4))
arrow((oui['x0'], ROW2), (gx + 0.52, ROW2))
arrow((gx - 0.52, ROW2), (act['x1'], ROW2))
note(gx, ROW2 - 0.72, 'shadow-mode gate', fs=8.4, color=INK)
note(gx, ROW2 + 0.98,
     'OFF during shadow rollout: predictions are logged, nothing is actuated',
     fs=8.4, color='#8a5a2b')

# operator verdicts feed the retrain label pool
arrow((oui['xc'], oui['y0']), (qadb['x0'] - 0.04, ROW2 - 0.34), color=GREY,
      style='--', rad=-0.20)
note((oui['xc'] + qadb['xc']) / 2, 1.02,
     'reviewed verdicts return to the QA DB as retrain labels (trigger T4)',
     fs=8.2)

# ---------- title and footer ----------
ax.text(0.35, 8.60, 'Severstal steel defect detection: production deployment',
        ha='left', va='center', fontsize=14, fontweight='bold', color=INK)
ax.text(0.35, 8.20,
        'Phase 5, model operations. The classifier triages every frame; only '
        'flagged frames reach the segmentation model.',
        ha='left', va='center', fontsize=9.5, color='#5a6673')
ax.text(0.35, 0.35,
        'Worst-case inference path 15 + 12 + 45 + 8 + 20 = 100 ms against a '
        '500-1000 ms frame period. P99 budget 250 ms; a frame that misses its '
        'slot is dropped and counted, never queued.',
        ha='left', va='center', fontsize=8.6, color='#5a6673')

fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print('wrote', os.path.normpath(OUT))

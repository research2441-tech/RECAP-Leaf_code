"""Fig. 1: RECAP-Leaf overview (grayscale).  Solid arrows: forward data flow;
dashed arrow: reverse-mode gradient of the distillation objective (student only);
dotted arrows: on-device feedback (recapture instruction, statistics rollback)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman'], 'font.size': 7.6,
                     'mathtext.fontset': 'stix'})
fig = plt.figure(figsize=(7.16, 3.7)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis('off')


def box(x, y, w, h, text, fill='white', lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.7', fc=fill, ec='black', lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', linespacing=1.2)


def arrow(p, q, ls='-', lw=0.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, lw=lw, ls=ls, color='black',
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def poly(pts, ls=':', lw=1.0):
    for a, b in zip(pts[:-1], pts[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color='black', ls=ls, lw=lw, solid_capstyle='butt')
    (x1, y1), (x2, y2) = pts[-2], pts[-1]                 # solid arrowhead on the dotted path
    ax.annotate('', xy=(x2, y2), xytext=(x2 - 0.01 * (x2 - x1), y2 - 0.01 * (y2 - y1)),
                arrowprops=dict(arrowstyle='-|>', mutation_scale=8, color='black', lw=0.8, shrinkA=0, shrinkB=0))


def label(x, y, text, ha='center'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=6.8, bbox=dict(fc='white', ec='none', pad=0.5))


# panels
for x0, w, title in ((0.8, 39.0, '(a) Offline: distillation and prototypes'), (41.6, 57.6, '(b) On device: capture-aware diagnosis loop')):
    ax.add_patch(FancyBboxPatch((x0, 0.8), w, 50.2, boxstyle='round,pad=0,rounding_size=1.0', fc='#F3F3F3', ec='black', lw=0.6, ls='--'))
    ax.text(x0 + w / 2, 48.9, title, ha='center', va='center', fontsize=8.0, weight='bold')

# ---------------- (a)
box(3.0, 38.5, 34.6, 6.6, 'Training images: laboratory leaves (labelled) +\nunlabelled in-the-wild photos of non-target species')
box(3.0, 26.5, 15.2, 6.8, 'Frozen teacher\nDINOv2 ViT-S/14', fill='#DCDCDC')
box(22.4, 26.5, 15.2, 6.8, 'Student $f_\\theta$\nMobileNetV4-Conv-S', lw=1.3)
box(5.0, 14.5, 25.0, 7.0, 'Distillation objective\n$\\mathcal{L}_{\\mathrm{tok}}+\\omega\\mathcal{L}_{\\mathrm{glb}}$ ($+\\beta\\mathcal{L}_{\\mathrm{ce}}$ on labelled leaves)')
box(3.0, 2.6, 15.2, 7.4, 'Support images\n(20 per class)')
box(22.4, 2.6, 15.2, 7.4, 'Class prototypes\n$\\mathbf{p}_y$ (Eq. 5)')
arrow((10.6, 38.5), (10.6, 33.3)); arrow((30.0, 38.5), (30.0, 33.3))
arrow((10.6, 26.5), (12.5, 21.5)); label(7.6, 24.0, '$\\mathbf{t},\\mathbf{T}$')
arrow((26.0, 26.5), (22.0, 21.5)); label(26.2, 23.8, '$\\mathbf{h},\\mathbf{Z}$')
arrow((30.0, 17.2), (32.6, 26.5), ls='--', lw=0.9, rad=0.32); label(34.4, 21.0, '$\\nabla_\\theta\\mathcal{L}$')
arrow((36.3, 26.5), (36.3, 10.0)); label(36.3, 12.6, 'frozen')
arrow((18.2, 6.3), (22.4, 6.3))

# ---------------- (b)
box(45.0, 37.5, 15.0, 6.8, 'Camera frame\n$\\mathbf{x}_t$')
box(66.0, 37.5, 20.0, 6.8, 'Student + prototypes\n$\\pi_y$, $u$, $o$ (Eqs. 6–9)')
box(45.0, 25.0, 15.0, 7.0, 'Capture factors\n$\\mathbf{q}$ (Eqs. 10–13)')
box(66.0, 25.0, 20.0, 7.0, 'Capture-risk model\n$r$ and $\\Delta R_a$ (Eqs. 14–17)', fill='#DCDCDC', lw=1.3)
box(45.0, 12.6, 15.0, 7.4, 'Recapture\ninstruction $a^{\\star}$')
box(64.0, 12.6, 14.0, 7.4, 'Accept diagnosis\n$r_{\\mathrm{acc}}\\leq\\hat{\\lambda}$ (Eq. 24)')
box(82.0, 12.6, 13.6, 7.4, 'Refer case\n$r_{\\mathrm{acc}}>\\hat{\\lambda}$')
box(62.0, 2.6, 26.0, 6.4, 'Guarded BN adaptation and drift guard\n(Eqs. 18–20)', fill='#DCDCDC')
arrow((60.0, 40.9), (66.0, 40.9))
arrow((52.5, 37.5), (52.5, 32.0))
arrow((60.0, 28.5), (66.0, 28.5))
arrow((76.0, 37.5), (76.0, 32.0))
arrow((68.5, 25.0), (56.0, 20.0)); label(60.4, 23.4, '$\\Delta R_{a^\\star}>c$')
arrow((73.5, 25.0), (71.0, 20.0))
arrow((80.0, 25.0), (86.5, 20.0))
poly([(45.0, 16.3), (43.4, 16.3), (43.4, 40.9), (45.0, 40.9)]); label(43.4, 34.8, 'retake')
arrow((71.0, 12.6), (71.0, 9.0)); label(77.2, 10.8, 'low-risk frames')
poly([(88.0, 5.8), (97.6, 5.8), (97.6, 46.2), (80.0, 46.2), (80.0, 44.3)]); label(91.0, 46.2, 'statistics / rollback')
fig.savefig('fig1_overview.png', dpi=400)
print('ok')

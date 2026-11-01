"""Result figures (grayscale, no overlapping legends) from work/results/summary.json."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'font.size': 8, 'mathtext.fontset': 'stix',
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})
S = json.load(open(sys.argv[1])); OUT = sys.argv[2]
STUD, TEACH, CE = 'RECAP-Leaf student', 'DINOv2-S/14 (frozen teacher)', 'MobileNetV4-Conv-S (CE)'


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=400, bbox_inches='tight', pad_inches=0.03); plt.close(fig)


# ---------------- Fig. 3: k-shot curves and accuracy versus compute
fig, axs = plt.subplots(2, 1, figsize=(3.45, 4.6), gridspec_kw=dict(hspace=0.5))
ax = axs[0]
styles = {TEACH: ('k', '-', 'o'), STUD: ('k', '--', 's'), CE: ('0.55', ':', '^')}
names = {TEACH: 'DINOv2-S/14 teacher', STUD: 'RECAP-Leaf student', CE: 'MobileNetV4-Conv-S (CE)'}
for m, d in S['kshot'].items():
    if not d: continue
    ks = sorted(int(k) for k in d); c, ls, mk = styles[m]
    ax.plot(ks, [100 * d[str(k)] for k in ks], color=c, ls=ls, marker=mk, ms=4, lw=1.1, label=names[m])
ax.set_xticks([0, 5, 10, 20]); ax.set_xlabel('Field support images per class'); ax.set_ylabel('PlantDoc field accuracy (%)')
ax.legend(loc='lower right', fontsize=7); ax.set_ylim(0, None); ax.set_title('(a) Field support', fontsize=8.5)
ax = axs[1]
eff = {e['model']: e for e in S.get('eff', [])}
pts = [(TEACH, 'DINOv2-S/14 teacher', 'DINOv2-S/14 teacher', 'o'), (STUD, 'RECAP-Leaf student (MobileNetV4-Conv-S + projection)', 'RECAP-Leaf student', 's'),
       (CE, 'MobileNetV4-Conv-S (CE)', 'MobileNetV4-Conv-S (CE)', '^'), ('DenseNet-121 (CE)', 'DenseNet-121', 'DenseNet-121 (CE)', 'D'),
       ('Compact three-block CNN (CE)', 'Compact three-block CNN', 'Compact CNN (CE)', 'v')]
cls = {(c['model'], c['head'], c['regime']): c for c in S['cls']}
for m, ek, lab, mk in pts:
    c = cls.get((m, 'prototype', 'field20'))
    if c is None or ek not in eff: continue
    x = eff[ek]['ms_1t']; yv = 100 * c['field_acc'][0]; e = 100 * c['field_acc'][1]
    ax.errorbar(x, yv, yerr=e, fmt=mk, color='k', mfc='white' if 'CE' in lab or 'Compact' in lab else 'k', ms=5, capsize=2, lw=0.8)
    off = {'DINOv2-S/14 teacher': (-7, 2), 'RECAP-Leaf student': (6, 4), 'MobileNetV4-Conv-S (CE)': (6, -10), 'DenseNet-121 (CE)': (-7, -12), 'Compact CNN (CE)': (6, 4)}[lab]
    ax.annotate(lab, (x, yv), textcoords='offset points', xytext=off, fontsize=6.6, ha='left' if off[0] > 0 else 'right')
ax.set_xscale('log'); ax.set_xlim(7, 160); ax.set_ylim(5, 53); ax.set_xlabel('CPU latency per frame, 1 thread (ms, log scale)'); ax.set_ylabel('Field accuracy, 20-shot (%)')
ax.set_title('(b) Accuracy against cost', fontsize=8.5)
save(fig, 'fig3_shots_cost.png')

# ---------------- Fig. 4: recapture accuracy versus photographs
keys = [k for k in ('student|field20', 'teacher|field20') if k in S['recap']]
fig, axs = plt.subplots(len(keys), 1, figsize=(3.45, 5.2), gridspec_kw=dict(hspace=0.32), squeeze=False)
axs = axs.T
st = dict(recap=('k', '-', 'o', 'RECAP-Leaf (targeted)'), qgate=('0.35', '--', 's', 'Quality gate + retake'),
          uncert=('0.6', ':', '^', 'Uncertainty + retake'))
for ax, key in zip(axs[0], keys):
    R = S['recap'][key]; single = R['single'][0]['acc']; orc = R['oracle'][0]
    for pol, (c, ls, mk, lab) in st.items():
        cv = sorted(R[pol], key=lambda d: d['recaptures'])
        xs = [0.0] + [d['recaptures'] for d in cv]; ys = [single] + [d['acc'] for d in cv]
        ax.plot(xs, [100 * v for v in ys], color=c, ls=ls, marker=mk, ms=3.5, lw=1.1, label=lab)
    ax.plot([orc['recaptures']], [100 * orc['acc']], marker='*', color='k', ms=8, ls='none', label='Oracle (defects known)')
    ax.axhline(100 * single, color='0.75', lw=0.7, zorder=0)
    ax.set_xlabel('Mean extra photographs per diagnosis'); ax.set_ylabel('Capture-benchmark accuracy (%)')
    ax.set_title('(%s) %s' % ('ab'[keys.index(key)], 'Student' if key.startswith('student') else 'Teacher'), fontsize=8.5)
    ax.set_xlim(-0.05, 2.05)
axs[0][-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=6.6, columnspacing=1.0, handlelength=2.2)
for a in axs[0]: a.set_xlabel('')
axs[0][-1].set_xlabel('Mean extra photographs per diagnosis')
save(fig, 'fig4_recapture.png')

# ---------------- Fig. 5: per-defect accuracy before / after recapture (student, field20)
key = 'student|field20' if 'student|field20' in S['recap'] else keys[0]
R = S['recap'][key]
recap_default = min(R['recap'], key=lambda d: abs(d['tau'] - 0.02))
qg = min(R['qgate'], key=lambda d: abs(d['recaptures'] - recap_default['recaptures']))
defs = ['blur', 'glare', 'occ', 'far']; lbl = ['Blur', 'Glare', 'Thumb', 'Distant\n(all items)']
fig, axs = plt.subplots(2, 1, figsize=(3.45, 4.7), gridspec_kw=dict(hspace=0.75, height_ratios=[1.3, 1]))
ax = axs[0]; w = 0.2; x = np.arange(4)
for i, (d, lab, fc) in enumerate(((R['single'][0], 'Single shot', 'white'), (qg, 'Quality gate + retake', '0.8'),
                                  (recap_default, 'RECAP-Leaf', '0.45'), (R['oracle'][0], 'Oracle', 'black'))):
    ax.bar(x + (i - 1.5) * w, [100 * d['per_defect'][k] for k in defs], w, color=fc, edgecolor='k', lw=0.5, label=lab)
ax.set_xticks(x); ax.set_xticklabels(lbl); ax.set_ylabel('Accuracy on affected items (%)')
ax.legend(ncol=2, fontsize=6.6, loc='lower center', bbox_to_anchor=(0.5, 1.02)); ax.set_title('(a) Accuracy by initial defect', fontsize=8.5, pad=30)
ax = axs[1]
ins = recap_default['instructions']; names_i = ['steady', 'shade', 'clear_thumb', 'move_closer']
tot = sum(ins.values()) or 1
ax.barh(np.arange(4), [100 * ins[a] / tot for a in names_i], color='0.55', edgecolor='k', lw=0.5)
ax.set_yticks(np.arange(4)); ax.set_yticklabels(['Steady camera', 'Shade leaf', 'Clear thumb', 'Move closer'])
ax.invert_yaxis(); ax.set_xlabel('Share of issued instructions (%)'); ax.set_title('(b) Instructions issued', fontsize=8.5)
save(fig, 'fig5_defects.png')

# ---------------- Fig. 6: test-time adaptation
if S.get('tta'):
    meth = ['none', 'bn_adapt', 'ema_bn', 'recap_noguard', 'tent', 'eata', 'sar', 'recap']
    lab = ['Source', 'Test-batch BN', 'EMA BN', 'Gated EMA', 'TENT', 'EATA', 'SAR', 'RECAP-Leaf']
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 4.8), gridspec_kw=dict(hspace=0.85))
    for ax, stream, ttl in zip(axs, ('iid', 'blocked'), ('(a) Shuffled stream', '(b) Disease-blocked stream')):
        d = S['tta'][stream]; ms = [m for m in meth if m in d]
        v = [100 * d[m]['acc'][0] for m in ms]; e = [100 * d[m]['acc'][1] for m in ms]
        cols = ['black' if m == 'recap' else ('0.45' if m == 'recap_noguard' else '0.8') for m in ms]
        ax.bar(range(len(ms)), v, yerr=e, color=cols, edgecolor='k', lw=0.5, capsize=1.5, error_kw=dict(lw=0.6))
        ax.axhline(100 * d['none']['acc'][0], color='k', lw=0.6, ls='--')
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([lab[meth.index(m)] for m in ms], rotation=40, ha='right', fontsize=7)
        ax.set_ylabel('Field accuracy (%)'); ax.set_title(ttl, fontsize=8.5)
        lo = min(v) - 6; ax.set_ylim(max(0, lo), max(v) + 5)
    save(fig, 'fig6_tta.png')

# ---------------- Fig. 7: risk-coverage and validity
if S.get('abst'):
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 4.6), gridspec_kw=dict(hspace=0.5))
    ax = axs[0]
    rc = json.load(open(os.path.join(os.path.dirname(sys.argv[1]), 'riskcov_curves.json')))
    for name, c, ls, lab in (('risk', 'k', '-', 'Capture-aware risk $r$'), ('risk_nofactor', '0.45', '--', 'Risk without factors'), ('softmax', '0.65', ':', 'Softmax confidence')):
        cv = rc[name]; ax.plot(100 * np.array(cv['coverage']), 100 * np.array(cv['risk']), color=c, ls=ls, lw=1.2, label=lab)
    ax.set_xlabel('Coverage (%)'); ax.set_ylabel('Selective risk (%)'); ax.legend(fontsize=7, loc='lower right')
    ax.set_title('(a) Risk–coverage, student, 20-shot', fontsize=8.5)
    ax = axs[1]
    al = [0.4, 0.5, 0.6]; w = 0.19
    bars = ((TEACH, 'samedata', '0.85', '', 'Teacher, same-data'), (TEACH, 'split', '0.35', '', 'Teacher, split'),
            (STUD, 'samedata', '0.85', '////', 'Student, same-data'), (STUD, 'split', '0.35', '////', 'Student, split'))
    for j, (m, kind, col, hat, lab) in enumerate(bars):
        d = S['abst'][f'{m}|field20']
        ax.bar(np.arange(3) + (j - 1.5) * w, [100 * d[f'viol_{kind}_{a}'] for a in al], w * 0.95, color=col, edgecolor='k', lw=0.5, hatch=hat, label=lab)
    ax.axhline(10, color='k', ls='--', lw=0.7); ax.text(2.5, 10.5, '$\\delta=0.1$', fontsize=7, ha='right')
    ax.set_xticks(range(3)); ax.set_xticklabels([f'$\\alpha={a}$' for a in al]); ax.set_ylabel('Violation frequency (%)')
    ax.set_ylim(0, 21); ax.legend(fontsize=6.6, loc='upper left', ncol=2, columnspacing=0.9, handlelength=1.6)
    ax.set_title('(b) Validity over 200 re-splits, 20-shot', fontsize=8.5)
    save(fig, 'fig7_risk.png')
print('figures written')

# ---------------- Fig. 8: confusion matrix of the student (20-shot), rows normalised
if S.get('confusion') and STUD in S['confusion']:
    C = np.array(S['confusion'][STUD]['matrix']); labs = S['confusion'][STUD]['labels']
    short = {'maize': 'Mz', 'pepper': 'Pp', 'potato': 'Pt', 'tomato': 'Tm'}
    abbr = {'common_rust': 'rust', 'gray_leaf_spot': 'gray spot', 'northern_leaf_blight': 'N. blight', 'bacterial_spot': 'bact. spot',
            'healthy': 'healthy', 'early_blight': 'early bl.', 'late_blight': 'late bl.', 'leaf_mold': 'leaf mold', 'mosaic_virus': 'mosaic',
            'septoria_leaf_spot': 'septoria', 'yellow_leaf_curl_virus': 'YLCV'}
    nm = [f"{short[l.split('/')[0]]} {abbr[l.split('/')[1]]}" if '/' in l else 'other crop' for l in labs]
    fig, ax = plt.subplots(figsize=(3.45, 3.55))
    im = ax.imshow(100 * C, cmap='Greys', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(nm))); ax.set_xticklabels(nm, rotation=90, fontsize=5.6)
    ax.set_yticks(range(len(nm) - 1)); ax.set_yticklabels(nm[:-1], fontsize=5.6)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            v = 100 * C[i, j]
            if v >= 8: ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=4.8, color='white' if v > 50 else 'black')
    ax.set_xlabel('Predicted class', fontsize=7); ax.set_ylabel('True class', fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=5.6); cb.set_label('Row percentage', fontsize=6)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(True)
    save(fig, 'fig8_confusion.png')
print('confusion figure written')

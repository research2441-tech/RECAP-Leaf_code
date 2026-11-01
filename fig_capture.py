"""Fig. 2: capture-benchmark examples with the student's risk before and after the
instruction chosen by RECAP-Leaf (partition 0, field-support regime)."""
import os, sys, json
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import WORK, stratified_split, to_tensor_np
from recap import CaptureRisk, uncertainty, Student
from evalkit import META, CLASSES, STATS, splits, fit_temperature, GlobalHead, class_prototypes
from capture import ACTION_FACTORS, render, FACTORS
import recapture_frames as RF

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'font.size': 7, 'mathtext.fontset': 'stix'})
seed = 0; out = sys.argv[1]
bank = np.load(os.path.join(WORK, f'recap_bank_student_s{seed}.npz')); index = json.load(open(os.path.join(WORK, f'recap_index_student_s{seed}.json')))
Q = bank['Q']; D = bank['D'].astype(np.float32); Dsup = np.load(os.path.join(WORK, f'recap_support_student_s{seed}.npy')).astype(np.float32)
E = np.load(os.path.join(WORK, f'emb_student_s{seed}.npz')); h = GlobalHead(os.path.join(WORK, f'student_s{seed}.pt'))
g = lambda t: h(torch.tensor(t).float().mean(1)).numpy()
y_lab = np.array(META['lab']['y']); y_reg = np.array(META['reg']['y'])
sp = splits(0); scenes_ = [it for it in index if it['kind'] == 'scene']
s_sup, s_rest = stratified_split(scenes_, lambda it: it['y'], [20], seed=700)
s_cal, s_test = stratified_split(s_rest, lambda it: it['y'], [0.4], seed=500)
P = class_prototypes(g(E['tok_lab']), y_lab, g(E['tok_reg'][sp['r_sup']]), y_reg[sp['r_sup']],
                     Dsup[np.array([it['idx'] for it in s_sup])], np.array([it['y'] for it in s_sup]))
S = D @ P.T
yf = np.zeros(len(D), int)
for it in index: yf[it['first']:it['first'] + it['n']] = it['y']
cal = s_cal[0::2]; cf = np.concatenate([np.arange(it['first'], it['first'] + it['n']) for it in cal])
init = np.array([it['first'] + it['paths'][''] for it in s_cal])
def prob(t, idx=slice(None)):
    s = 20 * S[idx] / t; s -= s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)
T = fit_temperature(lambda t: prob(t, init), yf[init]); p = prob(T); U = uncertainty(p); O = 1 - S.max(1)
rm = CaptureRisk().fit_reference(STATS['q_lab']).fit(Q[cf], U[cf], O[cf], (p.argmax(1) != yf).astype(float)[cf])
R = rm.risk(Q, U, O); dR = rm.expected_reduction(Q, U, O, list(ACTION_FACTORS.values())); ACTS = list(ACTION_FACTORS)
regs = [it for it in index if it['kind'] == 'reg']
want = [('blur', 'steady', 'scene'), ('glare', 'shade', 'scene'), ('occ', 'clear_thumb', 'scene'), ('far', 'move_closer', 'scene'), ('far', 'move_closer', 'reg')]
items = {(it['kind'], it['idx']): it for it in RF.items()}
chosen = []
for d, a, kind in want:
    pool = (s_test if kind == 'scene' else regs)
    best = None; cands = []
    for it in pool:
        if not it['defects'][d] or a not in it['paths']: continue
        f0 = it['first'] + it['paths']['']; f1 = it['first'] + it['paths'][a]
        if kind == 'scene' and d != 'far' and it['defects']['blur'] and d != 'blur': continue
        gain = (p[f1].argmax() == it['y']) * 1.0 + (p[f0].argmax() != it['y']) * 0.5 + (R[f0] - R[f1])
        cands.append((gain, it, f0, f1))
    cands.sort(key=lambda c: -c[0])
    for c in cands:
        if kind == 'reg':   # show a clearly distant composite so that the effect of moving closer is visible
            fr, pf, _ = RF.tree(items[(c[1]['kind'], c[1]['idx'])])
            if fr[pf[()]]['far_scale'] > 0.42: continue
        best = c; break
    if best: chosen.append((d, a, kind) + best[1:])
fig, axs = plt.subplots(2, len(chosen), figsize=(7.0, 3.25), gridspec_kw=dict(wspace=0.08, hspace=0.45))
names = dict(steady='steady camera', clear_thumb='clear thumb', shade='shade leaf', move_closer='move closer')
for j, (d, a, kind, it, f0, f1) in enumerate(chosen):
    frames, p2f, _ = RF.tree(items[(it['kind'], it['idx'])])
    st0 = frames[p2f[()]]; st1 = frames[p2f[(a,)]]
    for i, (st, f) in enumerate(((st0, f0), (st1, f1))):
        ax = axs[i, j]; ax.imshow(render(st)); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_linewidth(0.5)
        ok = 'correct' if p[f].argmax() == it['y'] else 'wrong'
        if i == 0:
            lab = {'blur': 'blur', 'glare': 'specular glare', 'occ': 'thumb occlusion', 'far': 'distant framing'}[d]
            src = 'scene' if kind == 'scene' else 'mango/citrus'
            ax.set_title(f'{lab} ({src})', fontsize=6.8, pad=2)
            ax.set_xlabel(f'r = {R[f]:.2f}, {ok}\nΔR({names[a]}) = {dR[f, ACTS.index(a)]:.2f}', fontsize=6.6, labelpad=2)
        else:
            ax.set_xlabel(f'after "{names[a]}"\nr = {R[f]:.2f}, {ok}', fontsize=6.6, labelpad=2)
fig.savefig(os.path.join(out, 'fig2_capture.png'), dpi=400, bbox_inches='tight', pad_inches=0.03)
print([(c[0], c[1], c[2], round(float(R[c[4]]), 3), round(float(R[c[5]]), 3)) for c in chosen])

"""Aggregate all experiment outputs into results/summary.json with paired tests."""
import os, json, glob, itertools
import numpy as np
from scipy.stats import wilcoxon, binomtest
from common import WORK

RES = os.path.join(WORK, 'results'); S = {}
rng = np.random.default_rng(0)


def boot_ci(d, n=10000):
    d = np.asarray(d, float); b = rng.choice(d, (n, len(d))).mean(1); return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


def paired(a, b):
    """Wilcoxon signed-rank on paired runs (two-sided) and bootstrap CI of the mean difference."""
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) and len(d) >= 5 else 1.0
    return dict(diff=float(d.mean()), ci=boot_ci(d), p=p, n=len(d))


# ------------------------------------------------------------------ E1/E2 classification
cls = json.load(open(os.path.join(RES, 'cls_runs_all.json')))
tab = {}
for r in cls:
    key = (r['model'], r['head'], r['regime'])
    tab.setdefault(key, []).append(r)
S['cls'] = []
for (m, h, reg), rows in tab.items():
    def ms(f): v = [f(r) for r in rows]; return [float(np.mean(v)), float(np.std(v))]
    S['cls'].append(dict(model=m, head=h, regime=reg, n=len(rows),
                         field_acc=ms(lambda r: r['field']['acc']), field_f1=ms(lambda r: r['field']['f1']),
                         field_bacc=ms(lambda r: r['field']['bacc']), field_ece=ms(lambda r: r['field']['ece']),
                         crop_acc=ms(lambda r: r['field_crop_acc']), reg_acc=ms(lambda r: r['reg']['acc']), reg_f1=ms(lambda r: r['reg']['f1']),
                         regf_acc=ms(lambda r: r['regf']['acc']),
                         native=ms(lambda r: r.get('native_field_acc', np.nan)) if any('native_field_acc' in r for r in rows) else None))
# paired tests: student vs others, per regime (prototype head), paired over (seed, split) or split means
stud = 'RECAP-Leaf student'
S['cls_tests'] = {}
for reg in ('lab', 'field20'):
    base = {(r['seed'], r['split']): r for r in cls if r['model'] == stud and r['head'] == 'prototype' and r['regime'] == reg}
    comps, ps = [], []
    for other in sorted({r['model'] for r in cls} - {stud}):
        rows = [r for r in cls if r['model'] == other and r['head'] == 'prototype' and r['regime'] == reg]
        if not rows: continue
        if len({r['seed'] for r in rows}) == 1:     # single-seed model: pair split means
            a = [np.mean([b['field']['acc'] for (s, k), b in base.items() if k == kk]) for kk in range(5)]
            bb = [next(r for r in rows if r['split'] == kk)['field']['acc'] for kk in range(5)]
        else:
            keys = sorted(set(base) & {(r['seed'], r['split']) for r in rows})
            a = [base[q]['field']['acc'] for q in keys]; bb = [next(r for r in rows if (r['seed'], r['split']) == q)['field']['acc'] for q in keys]
        t = paired(a, bb); t['other'] = other; comps.append(t); ps.append(t['p'])
    for t, pa in zip(comps, holm(ps) if ps else []): t['p_holm'] = pa
    S['cls_tests'][reg] = comps
# k-shot curve
S['kshot'] = {}
for m in ('DINOv2-S/14 (frozen teacher)', stud, 'MobileNetV4-Conv-S (CE)'):
    S['kshot'][m] = {str(sh): float(np.mean([r['field']['acc'] for r in cls if r['model'] == m and r['head'] == 'prototype' and r['shots'] == sh]))
                     for sh in (0, 5, 10, 20) if any(r['model'] == m and r['shots'] == sh for r in cls)}

# ------------------------------------------------------------------ E3 recapture
rec = []
for f in sorted(glob.glob(os.path.join(RES, 'recap_*_s*.json'))):
    if os.path.getsize(f): rec += json.load(open(f))
S['recap'] = {}
for bb, reg in itertools.product(('teacher', 'student'), ('field20', 'lab')):
    rows = [r for r in rec if r['backbone'] == bb and r['regime'] == reg]
    if not rows: continue
    out = {}
    for pol in ('single', 'oracle', 'recap', 'uncert', 'qgate'):
        pr = [r for r in rows if r['policy'] == pol]
        taus = sorted({round(r['tau'], 6) for r in pr}) if pol in ('single', 'oracle', 'recap') else list(range(6))
        curve = []
        for ti, t in enumerate(taus):
            if pol in ('uncert', 'qgate'):           # thresholds are per-split quantiles: index by rank
                g = {}
                for r in pr: g.setdefault((r['seed'], r['split']), []).append(r)
                sel = [sorted(v, key=lambda r: r['tau'])[ti] for v in g.values()]
            else:
                sel = [r for r in pr if abs(r['tau'] - t) < 1e-9]
            curve.append(dict(tau=float(t), acc=float(np.mean([r['acc'] for r in sel])), recaptures=float(np.mean([r['recaptures'] for r in sel])),
                              acc_sd=float(np.std([r['acc'] for r in sel])), runs=[(r['seed'], r['split'], r['acc'], r['recaptures']) for r in sel],
                              per_defect={d: float(np.nanmean([r[f'acc_{d}'] for r in sel])) for d in ('blur', 'glare', 'occ', 'far')},
                              acc_scene=float(np.mean([r['acc_scene'] for r in sel])), acc_reg=float(np.mean([r['acc_reg'] for r in sel])),
                              aurc={n: float(np.mean([r[f'aurc_{n}'] for r in sel])) for n in ('risk', 'risk_nofactor', 'softmax')},
                              cov={f'{n}_{a}': [float(np.mean([r[f'cov_{n}_{a}'] for r in sel])), float(np.nanmean([r[f'srisk_{n}_{a}'] for r in sel])) if any(r[f'cov_{n}_{a}'] > 0 for r in sel) else None]
                                   for n in ('risk', 'risk_nofactor', 'softmax') for a in (0.6, 0.7, 0.8)},
                              instructions=({a: float(np.mean([r['instructions'][a] for r in sel])) for a in sel[0]['instructions']} if pol == 'recap' else None)))
        out[pol] = curve
    out['weights'] = np.mean([r['weights'] for r in rows if r['policy'] == 'single'], 0).tolist()
    out['n_test'] = int(np.mean([r['n_test'] for r in rows]))
    S['recap'][f'{bb}|{reg}'] = out

# matched-budget comparison: interpolate each run's accuracy at given recapture budgets
def interp_at(curve_runs, budget):
    pts = sorted(curve_runs, key=lambda t: t[1])
    x = [p[1] for p in pts]; yv = [p[0] for p in pts]
    return float(np.interp(budget, x, yv))

S['recap_matched'] = {}
for key, out in S['recap'].items():
    res = {}
    for budget in (0.5, 1.0):
        per = {}
        for pol in ('recap', 'uncert', 'qgate'):
            runs = {}
            for c in out[pol]:
                for (sd, sp, acc, nr) in c['runs']: runs.setdefault((sd, sp), []).append((acc, nr))
            single = {(sd, sp): acc for c in out['single'] for (sd, sp, acc, nr) in c['runs']}
            per[pol] = {q: interp_at(v + [(single[q], 0.0)], budget) for q, v in runs.items()}
        tests = {o: paired([per['recap'][q] for q in sorted(per['recap'])], [per[o][q] for q in sorted(per['recap'])]) for o in ('uncert', 'qgate')}
        ad = holm([tests[o]['p'] for o in tests])
        for o, pa in zip(tests, ad): tests[o]['p_holm'] = pa
        res[str(budget)] = dict(acc={pol: float(np.mean(list(v.values()))) for pol, v in per.items()}, tests=tests)
    S['recap_matched'][key] = res

# ------------------------------------------------------------------ item-level McNemar tests
def mcnemar(a, b):
    """Exact two-sided McNemar test on paired correctness vectors a (proposed) and b (competitor)."""
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(wins=n01, losses=n10, p=float(p))

from evalkit import META, splits as _splits
yF = np.array(META['field']['y'])
P = np.load(os.path.join(RES, 'cls_probs_all.npz'))
S['cls_mcnemar'] = {}
for reg in ('lab', 'field20'):
    for other in sorted({r['model'] for r in cls} - {stud}):
        ps, w, l = [], 0, 0
        for k in range(5):
            ka, kb = f'{stud}|0|{k}|{reg}|f_test', f'{other}|0|{k}|{reg}|f_test'
            if ka not in P or kb not in P: continue
            yt = yF[_splits(k)['f_test']]
            t = mcnemar(P[ka].argmax(1) == yt, P[kb].argmax(1) == yt); ps.append(t['p']); w += t['wins']; l += t['losses']
        if ps: S['cls_mcnemar'][f'{reg}|{other}'] = dict(max_p=float(max(ps)), wins=w, losses=l, n_splits=len(ps))
# per-class F1 and confusion (20-shot, prototype head)
from sklearn.metrics import f1_score
from evalkit import CLASSES as _CL
fcls = sorted(set(yF.tolist()))
S['perclass'] = {}; S['confusion'] = {}
for m in (stud, 'DINOv2-S/14 (frozen teacher)'):
    f1s, conf = [], np.zeros((len(fcls), len(fcls) + 1))
    for k in range(5):
        key = f'{m}|0|{k}|field20|f_test'
        if key not in P: continue
        yt = yF[_splits(k)['f_test']]; pr = P[key].argmax(1)
        f1s.append(f1_score(yt, pr, labels=fcls, average=None, zero_division=0))
        for a, b in zip(yt, pr):
            conf[fcls.index(a), fcls.index(b) if b in fcls else len(fcls)] += 1
    if f1s:
        S['perclass'][m] = {'/'.join(_CL[c]): float(v) for c, v in zip(fcls, np.mean(f1s, 0))}
        S['confusion'][m] = dict(labels=['/'.join(_CL[c]) for c in fcls] + ['other crop'], matrix=(conf / conf.sum(1, keepdims=True)).tolist())
S['recap_mcnemar'] = {}
for bb in ('student', 'teacher'):
    f = os.path.join(RES, f'recapcorr_{bb}_s0.npz')
    if not os.path.exists(f): continue
    C = np.load(f); keys = [k for k in C.files if not k.endswith('|n')]
    for reg in ('field20', 'lab'):
        for tau in (0.02, 0.04):
            out = {}
            for other in ('qgate', 'uncert'):
                ps, w, l, nr_r, nr_o = [], 0, 0, [], []
                for k in range(5):
                    kr = f'{reg}|{k}|recap|{tau:.6f}'
                    if kr not in C: continue
                    nrec = C[kr + '|n'].mean()
                    cands = [q for q in keys if q.startswith(f'{reg}|{k}|{other}|')]
                    # competitor setting with at least as many photographs as RECAP-Leaf (conservative pairing)
                    cands = sorted(cands, key=lambda q: C[q + '|n'].mean())
                    sel = next((q for q in cands if C[q + '|n'].mean() >= nrec), cands[-1])
                    t = mcnemar(C[kr], C[sel]); ps.append(t['p']); w += t['wins']; l += t['losses']
                    nr_r.append(float(nrec)); nr_o.append(float(C[sel + '|n'].mean()))
                if ps: out[other] = dict(max_p=float(max(ps)), wins=w, losses=l, photos_recap=float(np.mean(nr_r)), photos_other=float(np.mean(nr_o)))
            S['recap_mcnemar'][f'{bb}|{reg}|{tau}'] = out

# ------------------------------------------------------------------ E4 TTA
tta = []
for f in sorted(glob.glob(os.path.join(RES, 'tta_s*_k*.json'))):
    if os.path.getsize(f): tta += json.load(open(f))
S['tta'] = {}
for stream in ('iid', 'blocked'):
    rows = [r for r in tta if r['stream'] == stream]
    S['tta'][stream] = {}
    for m in ('none', 'bn_adapt', 'ema_bn', 'recap_noguard', 'tent', 'eata', 'sar', 'recap'):
        rr = [r for r in rows if r['method'] == m]
        if not rr: continue
        S['tta'][stream][m] = dict(acc=[float(np.mean([r['acc'] for r in rr])), float(np.std([r['acc'] for r in rr]))],
                                   f1=float(np.mean([r['f1'] for r in rr])), ece=float(np.mean([r['ece'] for r in rr])),
                                   top=float(np.mean([r['top_class_share'] for r in rr])), rollbacks=float(np.mean([r['rollbacks'] for r in rr])),
                                   sec=float(np.mean([r['sec'] for r in rr])), n=len(rr))
    keyed = lambda m: {(r['seed'], r['split']): r['acc'] for r in rows if r['method'] == m}
    base = keyed('recap'); tests = {}
    for m in ('none', 'bn_adapt', 'ema_bn', 'recap_noguard', 'tent', 'eata', 'sar'):
        o = keyed(m); q = sorted(set(base) & set(o))
        if q: tests[m] = paired([base[i] for i in q], [o[i] for i in q])
    for m, pa in zip(tests, holm([t['p'] for t in tests.values()])): tests[m]['p_holm'] = pa
    S['tta'][stream]['tests'] = tests

# ------------------------------------------------------------------ E5 abstention, E6 efficiency
if os.path.exists(os.path.join(RES, 'abstention.json')):
    ab = json.load(open(os.path.join(RES, 'abstention.json'))); S['abst'] = {}
    for m in sorted({r['model'] for r in ab}):
        for reg in ('field20', 'lab'):
            rr = [r for r in ab if r['model'] == m and r['regime'] == reg]
            d = dict(acc=float(np.mean([r['acc'] for r in rr])))
            for k in rr[0]:
                if k.startswith(('aurc_', 'cov_', 'srisk_', 'viol_', 'meancov_')):
                    v = [r[k] for r in rr if r[k] == r[k]]
                    d[k] = float(np.mean(v)) if v else None
            S['abst'][f'{m}|{reg}'] = d
if os.path.exists(os.path.join(RES, 'efficiency.json')):
    S['eff'] = json.load(open(os.path.join(RES, 'efficiency.json')))
    S['eff_factors'] = next(e['ms_1t'] for e in S['eff'] if e['model'].startswith('capture'))
json.dump(S, open(os.path.join(RES, 'summary.json'), 'w'), indent=1)
print('summary written', list(S))

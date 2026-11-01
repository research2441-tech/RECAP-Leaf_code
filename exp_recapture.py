"""E3/E5: recapture policies and risk-controlled abstention on the frame bank.
Usage: python exp_recapture.py BACKBONE SEED
Policies (budget of two recaptures per diagnosis):
  single   one capture, no recapture
  qgate    unguided retake while an unweighted capture-quality score exceeds tau
  uncert   unguided retake while prediction uncertainty exceeds tau
  recap    RECAP-Leaf: targeted instruction a* when its expected risk reduction exceeds c
  oracle   targeted instruction for every defect actually present (upper reference)
"""
import os, sys, json
import numpy as np
from common import WORK, stratified_split
from recap import CaptureRisk, uncertainty, selective_threshold
from evalkit import META, CLASSES, STATS, splits, fit_temperature, GlobalHead
from capture import ACTION_FACTORS
import torch

backbone, seed = sys.argv[1], int(sys.argv[2]); BUDGET = 2
RES = os.path.join(WORK, 'results'); os.makedirs(RES, exist_ok=True)
bank = np.load(os.path.join(WORK, f'recap_bank_{backbone}_s{seed}.npz'))
index = json.load(open(os.path.join(WORK, f'recap_index_{backbone}_s{seed}.json')))
Q = bank['Q']; D = bank['D'].astype(np.float32); ACTS = list(ACTION_FACTORS); AF = [ACTION_FACTORS[a] for a in ACTS]
y_lab = np.array(META['lab']['y']); y_reg = np.array(META['reg']['y'])


def support_descriptors():
    if backbone == 'teacher':
        E = np.load(os.path.join(WORK, 'emb_teacher_s0.npz'))
        n = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
        return n(E['cls_lab'].astype(np.float32)), n(E['cls_reg'].astype(np.float32))
    E = np.load(os.path.join(WORK, f'emb_student_s{seed}.npz')); h = GlobalHead(os.path.join(WORK, f'student_s{seed}.pt'))
    g = lambda t: h(torch.tensor(t).float().mean(1)).numpy()
    return g(E['tok_lab']), g(E['tok_reg'])


H_lab, H_reg = support_descriptors()


D_sup = np.load(os.path.join(WORK, f'recap_support_{backbone}_s{seed}.npy')).astype(np.float32)


def prototypes(sup, field_idx=None, field_y=None):
    """Class means; with field support, a class that has field support images is
    represented by them (field-domain prototype) instead of its laboratory images."""
    X = np.concatenate([H_lab, H_reg[sup]]); yy = np.concatenate([y_lab, y_reg[sup]])
    P = np.stack([X[yy == k].mean(0) for k in range(len(CLASSES))])
    if field_idx is not None:
        for k in np.unique(field_y):
            P[k] = D_sup[field_idx[field_y == k]].mean(0)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def aurc(score, err):
    """Area under the risk-coverage curve (lower is better)."""
    o = np.argsort(score, kind='stable'); e = err[o]
    return float(np.mean(np.cumsum(e) / np.arange(1, len(e) + 1)))


def frame_of(it, path):
    f = it['paths'].get('|'.join(path)); return None if f is None else it['first'] + f


def simulate(items, policy, U, QG, dR, tau):
    fin, nrec = [], []
    for it in items:
        path, cur, n = (), frame_of(it, ()), 0
        if policy == 'oracle':
            for a, d in (('move_closer', 'far'), ('steady', 'blur'), ('shade', 'glare'), ('clear_thumb', 'occ')):
                if n < BUDGET and it['defects'][d]:
                    nxt = frame_of(it, path + (a,))
                    if nxt is not None: path, cur = path + (a,), nxt
                    n += 1
        else:
            while n < BUDGET and policy != 'single':
                if policy == 'recap':
                    j = int(np.argmax(dR[cur]))
                    if dR[cur, j] <= tau: break
                    nxt = frame_of(it, path + (ACTS[j],)); n += 1
                    if nxt is not None: path, cur = path + (ACTS[j],), nxt   # a no-op instruction still costs a capture
                    continue
                score = U[cur] if policy == 'uncert' else QG[cur]
                if score <= tau: break
                path = path + ('retake',); cur = frame_of(it, path); n += 1
        fin.append(cur); nrec.append(n)
    return np.array(fin), np.array(nrec)


CORR = {}
y_frame = np.zeros(len(D), int)
for it in index: y_frame[it['first']:it['first'] + it['n']] = it['y']
out = []
for regime, k in [(r, k) for r in ('field20', 'lab') for k in range(5)]:
    sp = splits(k); sup = sp['r_sup']
    scenes_ = [it for it in index if it['kind'] == 'scene']
    s_sup, s_rest = stratified_split(scenes_, lambda it: it['y'], [20], seed=700 + k)
    s_cal, s_test = stratified_split(s_rest, lambda it: it['y'], [0.4], seed=500 + k)
    if regime == 'field20':
        P = prototypes(sup, np.array([it['idx'] for it in s_sup]), np.array([it['y'] for it in s_sup]))
    else:
        P = prototypes(sup)
    regs = [it for it in index if it['kind'] == 'reg' and it['idx'] not in set(sup.tolist())]
    r_cal, r_test = stratified_split(regs, lambda it: it['y'], [0.4], seed=600 + k)
    cal, test = s_cal + r_cal, s_test + r_test
    # split calibration: the risk model is fitted on frames of half A of the calibration
    # items, and the acceptance threshold is selected on the other half B
    calA, calB = cal[0::2], cal[1::2]
    cal_frames = np.concatenate([np.arange(it['first'], it['first'] + it['n']) for it in calA])
    init_cal = np.array([frame_of(it, ()) for it in cal])
    S = D @ P.T

    def proba(t, idx=slice(None)):
        s = 20.0 * S[idx] / t; s = s - s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)
    T = fit_temperature(lambda t: proba(t, init_cal), y_frame[init_cal])
    Pall = proba(T); U = uncertainty(Pall); O = 1 - S.max(1); err = (Pall.argmax(1) != y_frame).astype(float)
    rms = {name: CaptureRisk(use_factors=uf).fit_reference(STATS['q_lab']).fit(Q[cal_frames], U[cal_frames], O[cal_frames], err[cal_frames])
           for name, uf in (('risk', True), ('risk_nofactor', False))}
    rm = rms['risk']; R = {n: m.risk(Q, U, O) for n, m in rms.items()}
    dR = rm.expected_reduction(Q, U, O, AF)
    QG = np.maximum(rm.standardise(Q) - rm.q_ref, 0).sum(1)
    base = dict(backbone=backbone, seed=seed, split=k, regime=regime, T=float(T), weights=rm.w.tolist(), n_test=len(test), n_cal=len(cal))
    grids = dict(single=[0.0], oracle=[0.0], recap=[0.0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.15, 0.25],
                 uncert=list(np.quantile(U[init_cal], [0.1, 0.3, 0.5, 0.7, 0.9, 0.99])),
                 qgate=list(np.quantile(QG[init_cal], [0.1, 0.3, 0.5, 0.7, 0.9, 0.99])))
    y_t = np.array([it['y'] for it in test]); y_c = np.array([it['y'] for it in cal])
    is_scene = np.array([it['kind'] == 'scene' for it in test])
    for pol, taus in grids.items():
        for tau in taus:
            f_t, n_t = simulate(test, pol, U, QG, dR, tau); f_c, n_c = simulate(calB, pol, U, QG, dR, tau)
            e_t = Pall[f_t].argmax(1) != y_t; e_c = Pall[f_c].argmax(1) != np.array([it['y'] for it in calB])
            CORR[f'{regime}|{k}|{pol}|{tau:.6f}'] = ~e_t; CORR[f'{regime}|{k}|{pol}|{tau:.6f}|n'] = n_t
            row = dict(base, policy=pol, tau=float(tau), acc=float(1 - e_t.mean()), recaptures=float(n_t.mean()),
                       acc_scene=float(1 - e_t[is_scene].mean()), acc_reg=float(1 - e_t[~is_scene].mean()),
                       acc_t1=None)
            for sname, sc_t in (('risk', R['risk'][f_t]), ('risk_nofactor', R['risk_nofactor'][f_t]), ('softmax', 1 - Pall[f_t].max(1))):
                row[f'aurc_{sname}'] = aurc(sc_t, e_t.astype(float))
            for alpha in (0.6, 0.7, 0.8):
                for sname, sc_t, sc_c in (('risk', R['risk'][f_t], R['risk'][f_c]), ('risk_nofactor', R['risk_nofactor'][f_t], R['risk_nofactor'][f_c]),
                                          ('softmax', 1 - Pall[f_t].max(1), 1 - Pall[f_c].max(1))):
                    lam = selective_threshold(sc_c, e_c.astype(float), alpha, delta=0.1)
                    if lam is None: cov, srisk = 0.0, float('nan')
                    else:
                        m = sc_t <= lam; cov = float(m.mean()); srisk = float(e_t[m].mean()) if m.any() else float('nan')
                    row[f'cov_{sname}_{alpha}'] = cov; row[f'srisk_{sname}_{alpha}'] = srisk
            for d in ('blur', 'glare', 'occ', 'far'):
                m = np.array([it['defects'][d] for it in test]); row[f'acc_{d}'] = float(1 - e_t[m].mean()) if m.any() else float('nan')
                row[f'n_{d}'] = int(m.sum())
            clean = np.array([not any(it['defects'][d] for d in ('blur', 'glare', 'occ', 'far')) for it in test])
            row['acc_clean'] = float(1 - e_t[clean].mean()) if clean.any() else float('nan'); row['n_clean'] = int(clean.sum())
            if pol == 'recap':                                     # which instructions were issued
                cnt = {a: 0 for a in ACTS}
                for it in test:
                    cur, path, n = frame_of(it, ()), (), 0
                    while n < BUDGET and dR[cur].max() > tau:
                        j = int(np.argmax(dR[cur])); cnt[ACTS[j]] += 1; nxt = frame_of(it, path + (ACTS[j],)); n += 1
                        if nxt is not None: path, cur = path + (ACTS[j],), nxt
                row['instructions'] = cnt
            out.append(row)
        print(regime, k, pol, [(round(r['tau'], 3), round(r['acc'], 3), round(r['recaptures'], 2)) for r in out if r['split'] == k and r['policy'] == pol and r['regime'] == regime], flush=True)
json.dump(out, open(os.path.join(RES, f'recap_{backbone}_s{seed}.json'), 'w'))
np.savez_compressed(os.path.join(RES, f'recapcorr_{backbone}_s{seed}.npz'), **CORR)

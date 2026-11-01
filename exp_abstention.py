"""E5: risk-controlled acceptance on natural PlantDoc field crops.
For each encoder, regime and partition: fit the capture-risk model on the
calibration portion, select the Learn-then-Test threshold, and measure coverage
and realised selective risk on the test portion.  A validity check re-draws the
calibration/test assignment 200 times and counts how often the realised risk
exceeds the target.  Usage: python exp_abstention.py"""
import os, json
import numpy as np, torch
from common import WORK
from recap import CaptureRisk, uncertainty, selective_threshold
from evalkit import META, CLASSES, STATS, splits, fit_temperature, GlobalHead, class_prototypes

RES = os.path.join(WORK, 'results')
y = {k: np.array(META[k]['y']) for k in ('lab', 'field', 'reg')}
unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
ALPHAS = (0.4, 0.5, 0.6)


def aurc(score, err):
    o = np.argsort(score, kind='stable'); e = err[o]; return float(np.mean(np.cumsum(e) / np.arange(1, len(e) + 1)))


def encoders():
    E = np.load(os.path.join(WORK, 'emb_teacher_s0.npz'))
    yield 'DINOv2-S/14 (frozen teacher)', 0, {k: unit(E[f'cls_{k}'].astype(np.float32)) for k in ('lab', 'field', 'reg')}
    for seed in range(3):
        f = os.path.join(WORK, f'emb_student_s{seed}.npz')
        if not os.path.exists(f): continue
        E = np.load(f); h = GlobalHead(os.path.join(WORK, f'student_s{seed}.pt'))
        yield 'RECAP-Leaf student', seed, {k: h(torch.tensor(E[f'tok_{k}']).float().mean(1)).numpy() for k in ('lab', 'field', 'reg')}


out = []; curves = {n: [] for n in ('risk', 'risk_nofactor', 'softmax')}
GRID = np.linspace(0.05, 1.0, 39)
for label, seed, H in encoders():
    for k in range(5):
        sp = splits(k)
        for regime in ('field20', 'lab'):
            fs = sp['f_sup'] if regime == 'field20' else None
            P = class_prototypes(H['lab'], y['lab'], H['reg'][sp['r_sup']], y['reg'][sp['r_sup']],
                                 None if fs is None else H['field'][fs], None if fs is None else y['field'][fs])
            S = H['field'] @ P.T
            prob = lambda t, idx: (lambda s: (lambda e: e / e.sum(1, keepdims=True))(np.exp(s - s.max(1, keepdims=True))))(20.0 * S[idx] / t)
            T = fit_temperature(lambda t: prob(t, sp['f_cal']), y['field'][sp['f_cal']])
            pool = np.concatenate([sp['f_cal'], sp['f_test']])
            p = prob(T, pool); U = uncertainty(p); O = 1 - S[pool].max(1); Qp = STATS['q_field'][pool]
            err = (p.argmax(1) != y['field'][pool]).astype(float); nc = len(sp['f_cal'])
            cal, test = np.arange(nc), np.arange(nc, len(pool))
            fitA, ltB = cal[: nc // 2], cal[nc // 2:]          # risk model on A, threshold on B (no double use)
            scores = {}
            for name, uf in (('risk', True), ('risk_nofactor', False)):
                rm = CaptureRisk(use_factors=uf).fit_reference(STATS['q_lab']).fit(Qp[fitA], U[fitA], O[fitA], err[fitA])
                scores[name] = rm.risk(Qp, U, O)
                # same-data variant (risk fitted and thresholded on the full calibration portion)
                if not uf:
                    rm2 = CaptureRisk(use_factors=False).fit_reference(STATS['q_lab']).fit(Qp[cal], U[cal], O[cal], err[cal]); scores['risk_samedata'] = rm2.risk(Qp, U, O)
            scores['softmax'] = 1 - p.max(1)
            rec = dict(model=label, seed=seed, split=k, regime=regime, acc=float(1 - err[test].mean()))
            if label == 'RECAP-Leaf student' and regime == 'field20':
                for name in curves:
                    o_ = np.argsort(scores[name][test], kind='stable'); e_ = err[test][o_]
                    cr = np.cumsum(e_) / np.arange(1, len(e_) + 1)
                    curves[name].append([float(cr[max(0, int(np.ceil(g * len(e_))) - 1)]) for g in GRID])
            for name, sc in scores.items():
                rec[f'aurc_{name}'] = aurc(sc[test], err[test])
                for a in ALPHAS:
                    cset = cal if name == 'risk_samedata' else ltB
                    lam = selective_threshold(sc[cset], err[cset], a, delta=0.1)
                    m = sc[test] <= lam if lam is not None else np.zeros(len(test), bool)
                    rec[f'cov_{name}_{a}'] = float(m.mean()); rec[f'srisk_{name}_{a}'] = float(err[test][m].mean()) if m.any() else float('nan')
            # validity: re-draw calibration/test assignments of the pooled data
            rng = np.random.default_rng(42 + k)
            for a in ALPHAS:
                for variant in ('split', 'samedata'):
                    viol, covs = 0, []
                    for _ in range(200):
                        perm = rng.permutation(len(pool)); c2, t2 = perm[:nc], perm[nc:]
                        A_, B_ = (c2[: nc // 2], c2[nc // 2:]) if variant == 'split' else (c2, c2)
                        rm = CaptureRisk(use_factors=False).fit_reference(STATS['q_lab']).fit(Qp[A_], U[A_], O[A_], err[A_]); sc = rm.risk(Qp, U, O)
                        lam = selective_threshold(sc[B_], err[B_], a, delta=0.1)
                        if lam is None: covs.append(0.0); continue
                        m = sc[t2] <= lam; covs.append(float(m.mean()))
                        if m.any() and err[t2][m].mean() > a: viol += 1
                    rec[f'viol_{variant}_{a}'] = viol / 200; rec[f'meancov_{variant}_{a}'] = float(np.mean(covs))
            out.append(rec)
            print(label, seed, k, regime, 'acc %.3f' % rec['acc'], 'AURC r/nf/s %.3f %.3f %.3f' % (rec['aurc_risk'], rec['aurc_risk_nofactor'], rec['aurc_softmax']),
                  ' '.join(f"a{a}: cov {rec[f'cov_risk_{a}']:.2f}/{rec[f'cov_softmax_{a}']:.2f} viol split {rec[f'viol_split_{a}']:.2f} same {rec[f'viol_samedata_{a}']:.2f}" for a in ALPHAS), flush=True)
json.dump(out, open(os.path.join(RES, 'abstention.json'), 'w'))
json.dump({n: dict(coverage=GRID.tolist(), risk=np.mean(v, 0).tolist()) for n, v in curves.items() if v},
          open(os.path.join(RES, 'riskcov_curves.json'), 'w'))

"""E1/E2: lab-to-field classification and few-shot regional recognition.
Every encoder is trained on laboratory labels only; regional classes enter
through 20 support images per class.  All encoders share the same cosine
prototype classifier (Eqs. 5-7) so that the comparison isolates the
representation; a logistic-regression probe is reported as a second head.
Usage: python exp_classification.py [model ...]"""
import os, sys, json
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from common import WORK
from evalkit import META, CLASSES, N_LAB, splits, fit_temperature, metrics, GlobalHead

RES = os.path.join(WORK, 'results'); os.makedirs(RES, exist_ok=True)
SETS = ('lab', 'field', 'reg', 'regf')
y = {k: np.array(META[k]['y']) for k in ('lab', 'field', 'reg')}; y['regf'] = y['reg']
unit = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)


def softmax(s):
    s = s - s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)


def encoders(name):
    """Yields (label, seed, dict set->descriptor, native lab-head logits or None)."""
    if name == 'teacher':
        E = np.load(os.path.join(WORK, 'emb_teacher_s0.npz'))
        yield 'DINOv2-S/14 (frozen teacher)', 0, {k: unit(E[f'cls_{k}'].astype(np.float32)) for k in SETS}, None
    elif name.startswith('student'):
        lab = {'student': 'RECAP-Leaf student', 'student_labonly': 'Student w/o field-domain corpus'}[name]
        for seed in range(3):
            f = os.path.join(WORK, f'emb_{name}_s{seed}.npz')
            if not os.path.exists(f): continue
            E = np.load(f); h = GlobalHead(os.path.join(WORK, f'{name}_s{seed}.pt'))
            yield lab, seed, {k: h(torch.tensor(E[f'tok_{k}']).float().mean(1)).numpy() for k in SETS}, None
    else:
        arch = name[3:]
        short = {'mobilenetv4_conv_small.e2400_r224_in1k': 'MobileNetV4-Conv-S (CE)', 'source_cnn': 'Compact three-block CNN (CE)',
                 'densenet121': 'DenseNet-121 (CE)'}[arch]
        for seed in range(3):
            f = os.path.join(WORK, f'emb_ce_{arch}_s{seed}.npz')
            if not os.path.exists(f): continue
            E = np.load(f)
            yield short, seed, {k: unit(E[f'feat_{k}'].astype(np.float32)) for k in SETS}, {k: E[f'logit_{k}'] for k in SETS}


runs, probs = [], {}
names = sys.argv[1:] or ['teacher', 'student', 'student_labonly', 'ce_mobilenetv4_conv_small.e2400_r224_in1k', 'ce_source_cnn', 'ce_densenet121']
from evalkit import class_prototypes
for name in names:
    for label, seed, H, native in encoders(name):
        for k in range(5):
            sp = splits(k); sup = sp['r_sup']
            for regime, shots in (('lab', 0), ('field20', 20), ('field10', 10), ('field5', 5)):
                fs = None
                if shots:
                    fs = np.concatenate([sp['f_sup'][y['field'][sp['f_sup']] == c][:shots] for c in np.unique(y['field'])])
                P = class_prototypes(H['lab'], y['lab'], H['reg'][sup], y['reg'][sup],
                                     None if fs is None else H['field'][fs], None if fs is None else y['field'][fs])
                heads = {'prototype': lambda s, idx, t, P=P: softmax(20.0 * H[s][idx] @ P.T / t)}
                if shots in (0, 20):
                    parts = [(H['lab'], y['lab']), (H['reg'][sup], y['reg'][sup])] + ([(H['field'][fs], y['field'][fs])] if shots else [])
                    Xs = np.concatenate([p[0] for p in parts]); ys = np.concatenate([p[1] for p in parts])
                    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced').fit(Xs, ys)
                    heads['linear probe'] = lambda s, idx, t, clf=clf: softmax(clf.decision_function(H[s][idx]) / t)
                for hname, fn in heads.items():
                    T = fit_temperature(lambda t: fn('field', sp['f_cal'], t), y['field'][sp['f_cal']])
                    rec = dict(model=label, head=hname, regime=regime, shots=shots, seed=seed, split=k, T=float(T))
                    pf = fn('field', sp['f_test'], T); rec['field'] = metrics(pf, y['field'][sp['f_test']])
                    rec['field_crop_acc'] = float(np.mean([CLASSES[a][0] == CLASSES[b][0] for a, b in zip(pf.argmax(1), y['field'][sp['f_test']])]))
                    for s in ('reg', 'regf'):
                        rec[s] = metrics(fn(s, sp['r_test'], T), y[s][sp['r_test']])
                    if native is not None and hname == 'prototype' and shots == 0:
                        lg = native['field'][sp['f_test']]; rec['native_field_acc'] = float((lg.argmax(1) == y['field'][sp['f_test']]).mean())
                    runs.append(rec)
                    if hname == 'prototype' and shots in (0, 20):
                        key = f'{label}|{seed}|{k}|{regime}'
                        probs[key + '|f_test'] = pf.astype(np.float32); probs[key + '|f_cal'] = fn('field', sp['f_cal'], T).astype(np.float32)
            print(label, seed, k, ' '.join(f"{r['regime']}/{r['head'][:5]} F{r['field']['acc']:.3f} R{r['reg']['acc']:.3f} RF{r['regf']['acc']:.3f}"
                                         for r in runs if r['seed'] == seed and r['split'] == k and r['model'] == label and r['shots'] in (0, 20)), flush=True)
tag = '_'.join(n.replace('.', '') for n in names) if len(names) < 3 else 'all'
json.dump(runs, open(os.path.join(RES, f'cls_runs_{tag}.json'), 'w'))
np.savez_compressed(os.path.join(RES, f'cls_probs_{tag}.npz'), **probs)

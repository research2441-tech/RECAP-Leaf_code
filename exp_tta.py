"""E4: test-time adaptation on PlantDoc field streams.
Usage: python exp_tta.py SEED SPLIT
Prototypes follow the field-support regime (20 field images per class).
Streams: i.i.d. order and class-blocked order (consecutive frames from one
disease, as in a single field visit).  Every method shares the same student,
prototypes and hierarchical head; only the adaptation rule differs."""
import os, sys, json, copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import WORK, load_rgb, to_tensor_np
from recap import Student, CaptureRisk, uncertainty, bn_layers, bn_state, bn_load, bn_update, norm_entropy
from evalkit import META, CLASSES, N_LAB, STATS, splits, fit_temperature, metrics, GlobalHead, class_prototypes

torch.set_num_threads(2)
seed, k = int(sys.argv[1]), int(sys.argv[2])
B = 16
GAMMA, A_MIN, MAX_ROLLBACKS = 0.85, 0.9, 3
RES = os.path.join(WORK, 'results'); os.makedirs(RES, exist_ok=True)
E = np.load(os.path.join(WORK, f'emb_student_s{seed}.npz'))
y = {s: np.array(META[s]['y']) for s in ('lab', 'field', 'reg')}
sp = splits(k)
head = GlobalHead(os.path.join(WORK, f'student_s{seed}.pt'))
g_np = lambda t: head(torch.tensor(t).float().mean(1)).numpy()
H_lab, H_sup = g_np(E['tok_lab']), g_np(E['tok_reg'][sp['r_sup']])
P_np = class_prototypes(H_lab, y['lab'], H_sup, y['reg'][sp['r_sup']], g_np(E['tok_field'][sp['f_sup']]), y['field'][sp['f_sup']])
H_cal = g_np(E['tok_field'][sp['f_cal']])


def np_proba(H, kap):
    s = kap * H @ P_np.T; s -= s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)


T = fit_temperature(lambda t: np_proba(H_cal, 20.0 / t), y['field'][sp['f_cal']]); KAP = 20.0 / T
p_cal = np_proba(H_cal, KAP); o_cal = 1 - (H_cal @ P_np.T).max(1)
risk = CaptureRisk().fit_reference(STATS['q_lab']).fit(STATS['q_field'][sp['f_cal']], uncertainty(p_cal), o_cal,
                                                         (p_cal.argmax(1) != y['field'][sp['f_cal']]).astype(float))
r_cal = risk.risk(STATS['q_field'][sp['f_cal']], uncertainty(p_cal), o_cal); R_TTA = float(np.quantile(r_cal, 0.5))
P = torch.tensor(P_np).float()


def torch_proba(z, pc=None, S=None):
    """Differentiable cosine-prototype posterior from student tokens (log-probabilities)."""
    W, b = (S.cls_proj.weight, S.cls_proj.bias) if S is not None else (head.W, head.b)
    h = F.normalize(z.mean(1) @ W.t() + b, dim=-1)
    return F.log_softmax(KAP * h @ P.t(), 1), h


test_idx = sp['f_test']
imgs = torch.tensor(np.stack([to_tensor_np(load_rgb(META['field']['paths'][i])) for i in test_idx]))
pcs = torch.tensor(STATS['pc_field'][test_idx]).float(); Qs = STATS['q_field'][test_idx]; yt = y['field'][test_idx]


def stream_order(kind, rng):
    if kind == 'iid': return rng.permutation(len(test_idx))
    order = []
    for c in rng.permutation(np.unique(yt)):                       # class-blocked
        order += list(rng.permutation(np.where(yt == c)[0]))
    return np.array(order)


def load_student():
    S = Student(N_LAB, pretrained=False); S.load_state_dict(torch.load(os.path.join(WORK, f'student_s{seed}.pt'))); S.eval(); return S


def bn_affine_params(S):
    ps = []
    for m in bn_layers(S):
        if m.weight is not None: ps += [m.weight, m.bias]
    return ps


def run(method, order, fisher=None):
    S = load_student(); src = load_student(); preds = np.zeros(len(order), int); probs = np.zeros((len(order), len(CLASSES)), np.float32)
    events = dict(rollbacks=0, updates=0, skipped=0)
    K = len(CLASSES); e0 = 0.4 * np.log(K)
    if method in ('tent', 'eata', 'sar'):
        for p in S.parameters(): p.requires_grad_(False)
        params = bn_affine_params(S)
        for p in params: p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=1e-4)
        init = copy.deepcopy(S.state_dict()); anchor = [p.detach().clone() for p in params]
        ema_loss, mprob = None, None
    source_state = bn_state(S); frozen = 0; win_a, win_s = [], []; disabled = False
    for b0 in range(0, len(order), B):
        idx = order[b0:b0 + B]; x = imgs[idx]; pc = pcs[idx]
        if method == 'none':
            with torch.no_grad(): lp, hh = torch_proba(S.tokens(x), pc, S)
        elif method == 'bn_adapt':                                   # test-batch statistics
            S.train()
            with torch.no_grad(): lp, hh = torch_proba(S.tokens(x), pc, S)
            S.eval(); S.load_state_dict(load_student().state_dict())
        elif method in ('ema_bn', 'recap', 'recap_noguard'):
            with torch.no_grad(): lp, hh = torch_proba(S.tokens(x), pc, S)
            p_np = lp.exp().numpy()
            if method == 'ema_bn':
                bn_update(S, x, 0.1); events['updates'] += 1
            else:
                with torch.no_grad(): lps, _ = torch_proba(src.tokens(x), pc, src)
                win_a += list(p_np); win_s += list(lps.exp().numpy())
                win_a, win_s = win_a[-64:], win_s[-64:]
                o_b = 1 - (hh.detach().numpy() @ P_np.T).max(1)
                r = risk.risk(Qs[idx], uncertainty(p_np), o_b); acc_m = r <= R_TTA
                if method == 'recap' and len(win_a) >= 32:          # drift guard (Eqs. 19-20)
                    pa, ps_ = np.array(win_a), np.array(win_s)
                    gamma = norm_entropy(pa.mean(0)) / max(norm_entropy(ps_.mean(0)), 1e-6)
                    agree = float((pa.argmax(1) == ps_.argmax(1)).mean())
                    if gamma < GAMMA or agree < A_MIN:              # roll back to the source statistics
                        bn_load(S, source_state); events['rollbacks'] += 1; frozen = 2; win_a, win_s = [], []
                        if events['rollbacks'] >= MAX_ROLLBACKS: disabled = True   # circuit breaker
                if disabled:
                    frozen = 10 ** 9; win_a, win_s = [], []
                if frozen > 0: frozen -= 1
                elif acc_m.sum() >= 4:
                    bn_update(S, x[torch.tensor(acc_m)], 0.1); events['updates'] += 1
                else: events['skipped'] += 1
        else:                                                       # gradient-based TTA on BN affine params
            S.train()
            lp, hh = torch_proba(S.tokens(x), pc, S); ent = -(lp.exp() * lp).sum(1)
            if method == 'tent':
                loss = ent.mean()
            elif method == 'eata':
                m = ent < e0
                pr = lp.exp().detach()
                if mprob is not None:
                    m = m & (F.cosine_similarity(pr, mprob.unsqueeze(0), dim=1) < 0.95)
                mprob = pr[m].mean(0) if m.any() and mprob is None else (0.9 * mprob + 0.1 * pr[m].mean(0) if m.any() else mprob)
                w = torch.exp(e0 - ent.detach())
                loss = (ent * w * m).sum() / max(1, int(m.sum()))
                if fisher is not None:
                    loss = loss + 2000.0 * sum((f * (p - a) ** 2).sum() for f, p, a in zip(fisher, params, anchor))
            else:                                                   # SAR: reliable entropy + sharpness-aware step
                m = ent < e0
                loss = (ent * m).sum() / max(1, int(m.sum()))
            opt.zero_grad()
            if method == 'sar' and loss.requires_grad:
                loss.backward()
                with torch.no_grad():
                    gn = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
                    eps = [(0.05 * p.grad / (gn + 1e-12)) if p.grad is not None else torch.zeros_like(p) for p in params]
                    for p, e in zip(params, eps): p.add_(e)
                lp2, _ = torch_proba(S.tokens(x), pc, S); ent2 = -(lp2.exp() * lp2).sum(1); m2 = ent2 < e0
                opt.zero_grad(); loss2 = (ent2 * m2).sum() / max(1, int(m2.sum()))
                if loss2.requires_grad: loss2.backward()
                with torch.no_grad():
                    for p, e in zip(params, eps): p.sub_(e)
                opt.step()
                ema_loss = loss2.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss2.item()
                if ema_loss is not None and ema_loss < 0.2:
                    S.load_state_dict(init); events['rollbacks'] += 1; ema_loss = None
            elif loss.requires_grad:
                loss.backward(); opt.step()
            events['updates'] += 1; lp = lp.detach(); S.eval()
        p_np = lp.exp().detach().numpy(); probs[b0:b0 + len(idx)] = p_np; preds[b0:b0 + len(idx)] = p_np.argmax(1)
    yo = yt[order]
    met = metrics(probs, yo)
    # collapse indicator: share of predictions taken by the most frequent class
    met['top_class_share'] = float(np.bincount(preds, minlength=len(CLASSES)).max() / len(preds))
    return met, events


def fisher_info():
    """Diagonal Fisher of BN affine parameters on 128 lab images (EATA anti-forgetting)."""
    S = load_student(); S.train()
    for p in S.parameters(): p.requires_grad_(False)
    params = bn_affine_params(S)
    for p in params: p.requires_grad_(True)
    fis = [torch.zeros_like(p) for p in params]
    rng = np.random.default_rng(seed); li = rng.choice(len(y['lab']), 128, replace=False)
    pcl = torch.tensor(STATS['pc_lab'][li]).float()
    for j in range(0, 128, 16):
        x = torch.tensor(np.stack([to_tensor_np(load_rgb(META['lab']['paths'][i])) for i in li[j:j + 16]]))
        lp, _ = torch_proba(S.tokens(x), None, S); loss = F.nll_loss(lp, lp.argmax(1))
        for p in params: p.grad = None
        loss.backward()
        for f, p in zip(fis, params): f += p.grad ** 2 / 8
    return [f.detach() for f in fis]


out = []; fis = fisher_info()
for kind in ('iid', 'blocked'):
    order = stream_order(kind, np.random.default_rng(1000 + k))
    for method in ('none', 'bn_adapt', 'ema_bn', 'recap_noguard', 'tent', 'eata', 'sar', 'recap'):
        t0 = time.time(); met, ev = run(method, order, fisher=fis if method == 'eata' else None)
        out.append(dict(seed=seed, split=k, stream=kind, method=method, **met, **ev, sec=time.time() - t0))
        print(out[-1], flush=True)
json.dump(out, open(os.path.join(RES, f'tta_s{seed}_k{k}.json'), 'w'))

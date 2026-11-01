"""Evaluation helpers shared by the experiment scripts."""
import os, json
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy.optimize import minimize_scalar
from common import WORK, stratified_split
from recap import HierPrototypes, lesion_pool

META = json.load(open(os.path.join(WORK, 'sets.json')))
CLASSES = [tuple(c) for c in META['classes']]
N_LAB = 19
STATS = np.load(os.path.join(WORK, 'image_stats.npz'))


def splits(seed):
    """Per-seed partition.  Field: 20 support images per class (used only in the
    field-support regime), then 40/60 calibration/test of the remainder; the test
    portion is identical in both regimes.  Regional: 20 support / 20 calibration /
    remainder test per class."""
    yf = np.array(META['field']['y']); yr = np.array(META['reg']['y'])
    fi = list(range(len(yf))); ri = list(range(len(yr)))
    f_sup, f_rest = stratified_split(fi, lambda i: yf[i], [20], seed + 900)
    f_cal, f_test = stratified_split(f_rest, lambda i: yf[i], [0.4], seed)
    r_sup, r_cal, r_test = stratified_split(ri, lambda i: yr[i], [20, 20], seed)
    return dict(f_sup=np.array(f_sup), f_cal=np.array(f_cal), f_test=np.array(f_test), r_sup=np.array(r_sup),
                r_cal=np.array(r_cal), r_test=np.array(r_test))


def class_prototypes(H_lab, y_lab, H_rsup, y_rsup, H_fsup=None, y_fsup=None):
    """Cosine class prototypes (Eq. 5).  A class with field support images is
    represented by them; other classes by laboratory or regional support images."""
    X = np.concatenate([H_lab, H_rsup]); yy = np.concatenate([y_lab, y_rsup])
    P = np.stack([X[yy == c].mean(0) for c in range(len(CLASSES))])
    if H_fsup is not None:
        for c in np.unique(y_fsup): P[c] = H_fsup[y_fsup == c].mean(0)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def healthy_bank(tok_lab, pc_lab, y_lab, tok_sup, pc_sup, y_sup):
    """One healthy-tissue prototype per crop from plant-tissue tokens of healthy images."""
    bank = []
    crops = list(dict.fromkeys(c for c, _ in CLASSES))
    for crop in crops:
        k = CLASSES.index((crop, 'healthy'))
        src = (tok_lab, pc_lab, y_lab) if k < N_LAB else (tok_sup, pc_sup, y_sup)
        t, pc, y = src; m = y == k
        z = F.normalize(torch.tensor(t[m]).float(), dim=-1)          # n x 49 x d
        w = torch.tensor(pc[m] > 0.5).float().unsqueeze(-1)
        v = (z * w).sum((0, 1)) / w.sum().clamp(min=1)
        bank.append(F.normalize(v, dim=0))
    return torch.stack(bank)


def descriptors(tok, pc, bank, mode, tau=0.05, mix=0.5):
    z = torch.tensor(tok).float()
    if mode == 'mean':
        return F.normalize(F.normalize(z, dim=-1).mean(1), dim=-1).numpy()
    zn = F.normalize(z, dim=-1)
    ell = 1 - (zn @ bank.t()).max(-1).values
    logit = ell / tau + torch.log(torch.tensor(pc).float() + 0.05)    # plant-tissue prior
    a = torch.softmax(logit, 1)
    les = F.normalize((a.unsqueeze(-1) * zn).sum(1), dim=-1)
    h = mix * les + (1 - mix) * F.normalize(zn.mean(1), dim=-1)
    return F.normalize(h, dim=-1).numpy()


def fit_temperature(p_fn, y):
    """Scalar temperature on calibration data: p_fn(T) -> probabilities."""
    r = minimize_scalar(lambda t: -np.mean(np.log(p_fn(t)[np.arange(len(y)), y] + 1e-12)), bounds=(0.2, 5.0), method='bounded')
    return r.x


def metrics(p, y, classes_subset=None):
    pred = p.argmax(1); acc = float((pred == y).mean())
    labels = np.unique(y) if classes_subset is None else classes_subset
    f1 = float(f1_score(y, pred, labels=labels, average='macro', zero_division=0))
    bacc = float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else acc
    conf = p.max(1); ece = 0.0; bins = np.linspace(0, 1, 16)
    for a, b in zip(bins[:-1], bins[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    nll = float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))
    return dict(acc=acc, f1=f1, bacc=bacc, ece=float(ece), nll=nll)


def proto_classifier(H_lab, y_lab, H_sup, y_sup, rho, hierarchical, kappa=20.0):
    hp = HierPrototypes(CLASSES, rho=rho, kappa=kappa)
    hp.fit(np.concatenate([H_lab, H_sup]), np.concatenate([y_lab, y_sup]))
    hp.hier = hierarchical
    return hp


class GlobalHead:
    """Student global projection (the CLS-mimicking head) applied to pooled tokens."""
    def __init__(self, ckpt):
        sd = torch.load(ckpt, map_location='cpu')
        self.W = sd['cls_proj.weight'].float(); self.b = sd['cls_proj.bias'].float()

    def __call__(self, pooled):
        return F.normalize(pooled @ self.W.t() + self.b, dim=-1)


def lesion_weights(tok, pc, bank, tau=0.1):
    """Lesion-grounded attention (Eq. 4-5): tokens far from every healthy-tissue
    prototype and inside plant tissue receive larger weights."""
    zn = F.normalize(torch.tensor(tok).float(), dim=-1)
    ell = 1 - (zn @ bank.t()).max(-1).values
    return torch.softmax(ell / tau + torch.log(torch.tensor(pc).float() + 0.05), 1)


def global_descriptors(tok, pc, bank, head, mode='lesion', tau=0.1):
    z = torch.tensor(tok).float()
    if mode == 'mean': pooled = z.mean(1)
    else: pooled = (lesion_weights(tok, pc, bank, tau).unsqueeze(-1) * z).sum(1)
    with torch.no_grad(): return head(pooled).numpy()

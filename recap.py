"""RECAP-Leaf core: student encoder, lesion-grounded pooling, hierarchical
shrinkage prototypes, capture-risk model, recapture policy, guarded test-time
adaptation and risk-controlled abstention."""
import copy, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from scipy.optimize import minimize
from scipy.stats import binom

D_TOK = 384


# ------------------------------------------------------------------ encoders
class Teacher(nn.Module):
    """Frozen DINOv2 ViT-S/14; returns CLS token and 7x7 pooled patch tokens."""
    def __init__(self):
        super().__init__()
        self.m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, img_size=224, num_classes=0)
        self.m.eval()
        for p in self.m.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        t = self.m.forward_features(x)                    # B x (1+256) x 384
        cls, pt = t[:, 0], t[:, 1 + self.m.num_reg_tokens if hasattr(self.m, 'num_reg_tokens') else 1:]
        B, N, C = pt.shape; g = int(math.sqrt(N))
        pt = pt.transpose(1, 2).reshape(B, C, g, g)
        pt = F.adaptive_avg_pool2d(pt, 7).flatten(2).transpose(1, 2)   # B x 49 x 384
        return cls, pt


class Student(nn.Module):
    """MobileNetV4-Conv-Small trunk + 1x1 projection to the teacher token space."""
    def __init__(self, n_lab_classes, arch='mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True):
        super().__init__()
        self.trunk = timm.create_model(arch, pretrained=pretrained, features_only=True, out_indices=(4,))
        c = self.trunk.feature_info.channels()[-1]
        self.proj = nn.Sequential(nn.Conv2d(c, 512, 1, bias=False), nn.BatchNorm2d(512), nn.GELU(),
                                  nn.Conv2d(512, D_TOK, 1))
        self.cls_proj = nn.Linear(D_TOK, D_TOK)
        self.W = nn.Parameter(torch.randn(n_lab_classes, D_TOK) * 0.02)   # lab class anchors
        self.kappa = 16.0

    def tokens(self, x):
        f = self.trunk(x)[-1]
        z = self.proj(f)                                  # B x 384 x 7 x 7
        return z.flatten(2).transpose(1, 2)               # B x 49 x 384

    def forward(self, x):
        z = self.tokens(x)
        g = self.cls_proj(z.mean(1))
        logits = self.kappa * F.normalize(g, dim=-1) @ F.normalize(self.W, dim=-1).t()
        return z, g, logits


def distill_loss(z, g, t_tok, t_cls, w_glob=2.0):
    """Token-level and global cosine distillation (Eq. 1)."""
    l_tok = (1 - F.cosine_similarity(z, t_tok, dim=-1)).mean()
    l_cls = (1 - F.cosine_similarity(g, t_cls, dim=-1)).mean()
    return l_tok + w_glob * l_cls


# ------------------------------------------------------------------ descriptors
def lesion_pool(z, healthy_bank, tau=0.05, mix=0.5):
    """Lesion-grounded pooling (Eqs. 4-5). z: B x M x d tokens; healthy_bank: H x d
    unit-norm healthy-tissue prototypes. Tokens far from every healthy prototype
    receive more weight; the descriptor mixes lesion-weighted and mean pooling."""
    zn = F.normalize(z, dim=-1)
    sim = zn @ healthy_bank.t()                           # B x M x H
    ell = 1 - sim.max(-1).values                          # lesion score per token
    a = torch.softmax(ell / tau, dim=1)                   # B x M
    les = (a.unsqueeze(-1) * zn).sum(1)
    h = mix * F.normalize(les, dim=-1) + (1 - mix) * F.normalize(zn.mean(1), dim=-1)
    return F.normalize(h, dim=-1), a


def mean_pool(z):
    return F.normalize(F.normalize(z, dim=-1).mean(1), dim=-1)


class HierPrototypes:
    """Hierarchical crop->disease prototypes with count-dependent shrinkage (Eqs. 6-8)."""
    def __init__(self, classes, rho=8.0, kappa=20.0):
        self.classes, self.rho, self.kappa = classes, rho, kappa
        self.crops = list(dict.fromkeys(c for c, _ in classes))
        self.crop_of = np.array([self.crops.index(c) for c, _ in classes])

    def fit(self, H, y):
        """H: n x d unit descriptors (numpy), y: class indices into self.classes."""
        K, d = len(self.classes), H.shape[1]
        mu = np.zeros((K, d)); n = np.zeros(K)
        for k in range(K):
            m = y == k; n[k] = m.sum()
            if n[k]: mu[k] = H[m].mean(0)
        P_crop = np.zeros((len(self.crops), d))
        for c in range(len(self.crops)):
            ks = np.where((self.crop_of == c) & (n > 0))[0]
            P_crop[c] = mu[ks].mean(0)
        shrink = n / (n + self.rho)
        P = shrink[:, None] * mu + (1 - shrink[:, None]) * P_crop[self.crop_of]
        self.P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        self.Pc = P_crop / (np.linalg.norm(P_crop, axis=1, keepdims=True) + 1e-8)
        self.n = n
        return self

    def proba(self, H, hierarchical=True):
        s = self.kappa * H @ self.P.T                     # n x K
        if not hierarchical:
            s = s - s.max(1, keepdims=True); p = np.exp(s); return p / p.sum(1, keepdims=True)
        sc = self.kappa * H @ self.Pc.T
        sc = sc - sc.max(1, keepdims=True); pc = np.exp(sc); pc /= pc.sum(1, keepdims=True)
        out = np.zeros_like(s)
        for c in range(len(self.crops)):
            ks = np.where(self.crop_of == c)[0]
            sk = s[:, ks] - s[:, ks].max(1, keepdims=True); pk = np.exp(sk); pk /= pk.sum(1, keepdims=True)
            out[:, ks] = pc[:, [c]] * pk
        return out


def uncertainty(p):
    """Confidence deficit (Eq. 8): 1 - max_y p_y."""
    return 1 - p.max(1)


# ------------------------------------------------------------------ capture risk
class CaptureRisk:
    """Monotone logistic model of misdiagnosis probability (Eqs. 13-15).
    Inputs: five standardised capture factors q, the prediction margin
    uncertainty u and the prototype distance o = 1 - max_k cos(h, p_k)."""
    def __init__(self, l2=1e-2, use_factors=True):
        self.l2 = l2; self.use_factors = use_factors

    def standardise(self, Q):
        return (Q - self.q_mu) / self.q_sd

    def fit_reference(self, Q_ref):
        self.q_mu = Q_ref.mean(0); self.q_sd = Q_ref.std(0) + 1e-6
        self.q_ref = np.median(self.standardise(Q_ref), 0)      # typical good capture
        return self

    def _design(self, Q, u, o):
        F_ = self.standardise(Q) if self.use_factors else np.zeros_like(Q)
        return np.column_stack([F_, u, o])

    def fit(self, Q, u, o, err):
        X = self._design(Q, u, o); n, p = X.shape

        def f(w):
            z = w[0] + X @ w[1:]
            nll = np.mean(np.logaddexp(0, z) - err * z) + self.l2 * np.sum(w[1:] ** 2)
            s = 1 / (1 + np.exp(-z))
            g = np.concatenate([[np.mean(s - err)], X.T @ (s - err) / n + 2 * self.l2 * w[1:]])
            return nll, g
        b = [(None, None)] + [(0, None)] * p                  # non-negative weights (monotone risk)
        r = minimize(f, np.zeros(p + 1), jac=True, method='L-BFGS-B', bounds=b)
        self.w = r.x
        return self

    def logit(self, Q, u, o):
        return self.w[0] + self._design(Q, u, o) @ self.w[1:]

    def risk(self, Q, u, o):
        return 1 / (1 + np.exp(-self.logit(Q, u, o)))

    def factor_contrib(self, Q):
        """Per-factor excess logit relative to a typical good capture (Eq. 15)."""
        if not self.use_factors: return np.zeros_like(Q)
        return self.w[1:6] * np.maximum(self.standardise(Q) - self.q_ref, 0)

    def expected_reduction(self, Q, u, o, action_factors):
        """Expected risk reduction of each recapture action (Eq. 16)."""
        z = self.logit(Q, u, o); r = 1 / (1 + np.exp(-z)); C = self.factor_contrib(Q)
        out = [r - 1 / (1 + np.exp(-(z - C[:, list(idx)].sum(1)))) for idx in action_factors]
        return np.stack(out, 1)


# ------------------------------------------------------------------ risk control
def selective_threshold(score, err, alpha, delta=0.1, grid=None):
    """Fixed-sequence Learn-then-Test threshold on the acceptance score (Eqs. 21-23).
    Accept when score <= lambda.  Hypotheses H_lambda: selective risk > alpha are
    tested from the most conservative lambda upward with a binomial p-value; the
    largest lambda reached before the first non-rejection is returned."""
    if grid is None:
        grid = np.quantile(score, np.linspace(0.02, 1.0, 50))
    lam_hat = None
    for lam in np.sort(grid):
        acc = score <= lam; n = int(acc.sum())
        if n == 0: continue
        k = int(err[acc].sum())
        pval = binom.cdf(k, n, alpha)
        if pval <= delta: lam_hat = lam
        else: break
    return lam_hat


# ------------------------------------------------------------------ guarded TTA
def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def bn_state(model):
    return [(m.running_mean.clone(), m.running_var.clone()) for m in bn_layers(model)]


def bn_load(model, st):
    for m, (a, b) in zip(bn_layers(model), st):
        m.running_mean.copy_(a); m.running_var.copy_(b)


@torch.no_grad()
def bn_update(model, x, eta):
    """Exponential update of BN statistics from a batch of accepted frames (Eq. 18)."""
    feats = {}
    hooks = []
    for i, m in enumerate(bn_layers(model)):
        def hk(mod, inp, out, i=i):
            a = inp[0]; feats[i] = (a.mean((0, 2, 3)), a.var((0, 2, 3), unbiased=False))
        hooks.append(m.register_forward_hook(hk))
    model.eval(); model.tokens(x)
    for h in hooks: h.remove()
    for i, m in enumerate(bn_layers(model)):
        mu, var = feats[i]
        m.running_mean.mul_(1 - eta).add_(eta * mu); m.running_var.mul_(1 - eta).add_(eta * var)


def norm_entropy(p):
    p = np.clip(p, 1e-12, 1); return float(-(p * np.log(p)).sum() / np.log(len(p)))

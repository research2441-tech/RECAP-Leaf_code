"""Distil the frozen DINOv2 teacher into the MobileNetV3 student on lab images only.
Lab images give token/global distillation plus a cross-entropy anchor; unlabeled
in-the-wild photographs of NON-target species (data/unlabeled.txt) give token and
global distillation only, which carries the teacher's field-domain geometry into
the student without any target-class field label.
Usage: python train_student.py SEED [EPOCHS] [--labonly]"""
import os, sys, time, json
import numpy as np, torch, torch.nn.functional as F
from common import manifest, label_space, load_rgb, to_tensor_np, stratified_split, lab_budget, WORK
from recap import Teacher, Student, distill_loss

torch.set_num_threads(2)
seed = int(sys.argv[1]); EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LABONLY = '--labonly' in sys.argv; TAG = 'labonly_' if LABONLY else ''
rows = manifest(); classes, crops = label_space(rows)
lab = [r for r in rows if r['domain'] == 'lab']
cidx = {c: i for i, c in enumerate(classes)}
n_lab = len({(r['crop'], r['disease']) for r in lab})
tr, va = stratified_split(lab, lambda r: (r['crop'], r['disease']), [0.9], seed=100)  # fixed lab split
tr = lab_budget(tr)

# ---- teacher cache (computed once, shared by all seeds)
cache = os.path.join(WORK, 'teacher_lab_train.pt')
if not os.path.exists(cache):
    T = Teacher(); cls_l, tok_l = [], []
    for i in range(0, len(tr), 32):
        x = torch.tensor(np.stack([to_tensor_np(load_rgb(r['path'])) for r in tr[i:i + 32]]))
        c, t = T(x); cls_l.append(c.half()); tok_l.append(t.half())
        if i % 640 == 0: print('teacher', i, len(tr), flush=True)
    torch.save(dict(paths=[r['path'] for r in tr], cls=torch.cat(cls_l), tok=torch.cat(tok_l)), cache)
C = torch.load(cache); assert C['paths'] == [r['path'] for r in tr]
from common import DATA
unl = open(os.path.join(DATA, 'unlabeled.txt')).read().split('\n')
cache_u = os.path.join(WORK, 'teacher_unlab.pt')
if not LABONLY and not os.path.exists(cache_u):
    T = Teacher(); cls_l, tok_l = [], []
    for i in range(0, len(unl), 32):
        x = torch.tensor(np.stack([to_tensor_np(load_rgb(p)) for p in unl[i:i + 32]]))
        c, t = T(x); cls_l.append(c.half()); tok_l.append(t.half())
    torch.save(dict(paths=unl, cls=torch.cat(cls_l), tok=torch.cat(tok_l)), cache_u)
if not LABONLY:
    CU = torch.load(cache_u); assert CU['paths'] == unl
    uimgs = np.stack([load_rgb(p) for p in unl]); BU = 32
y_tr = torch.tensor([cidx[(r['crop'], r['disease'])] for r in tr])
y_va = np.array([cidx[(r['crop'], r['disease'])] for r in va])

torch.manual_seed(seed); np.random.seed(seed)
S = Student(n_lab)
imgs = np.stack([load_rgb(r['path']) for r in tr])           # uint8 cache, ~0.8 GB
va_x = torch.tensor(np.stack([to_tensor_np(load_rgb(r['path'])) for r in va]))
bb = [p for n, p in S.named_parameters() if n.startswith('trunk')]
hd = [p for n, p in S.named_parameters() if not n.startswith('trunk')]
opt = torch.optim.AdamW([{'params': bb, 'lr': 4e-4}, {'params': hd, 'lr': 2e-3}], weight_decay=0.05)
B = 32; steps = EPOCHS * (len(tr) // B); sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=[4e-4, 2e-3], total_steps=steps, pct_start=0.1)
rng = np.random.default_rng(seed); log = []


def augment(batch, rng):
    out, flips = [], []
    for im in batch:
        im = im.astype(np.float32)
        f = rng.random() < 0.5
        if f: im = im[:, ::-1]
        a = rng.uniform(0.8, 1.2); b = rng.uniform(-20, 20); s = rng.uniform(0.8, 1.2)
        gray = im.mean(2, keepdims=True); im = (gray + s * (im - gray)) * a + b
        out.append(to_tensor_np(np.clip(im, 0, 255).astype(np.uint8))); flips.append(f)
    return torch.tensor(np.stack(out)), flips


t0 = time.time(); step = 0
for ep in range(EPOCHS):
    S.train(); perm = rng.permutation(len(tr)); tot = 0
    for i in range(0, len(perm) - B + 1, B):
        idx = perm[i:i + B]; x, flips = augment(imgs[idx], rng)
        tt = C['tok'][idx].float().view(B, 7, 7, -1)
        for j, f in enumerate(flips):
            if f: tt[j] = tt[j].flip(1)                          # align token grid with flipped image
        z, g, logits = S(x)
        loss = distill_loss(z, g, tt.view(B, 49, -1), C['cls'][idx].float()) + \
            0.5 * F.cross_entropy(logits, y_tr[idx], label_smoothing=0.1)
        if not LABONLY:                                              # unlabeled field-domain distillation
            ui = rng.integers(0, len(unl), BU); xu, fu = augment(uimgs[ui], rng)
            tu = CU['tok'][ui].float().view(BU, 7, 7, -1)
            for j, f in enumerate(fu):
                if f: tu[j] = tu[j].flip(1)
            zu, gu, _ = S(xu)
            loss = loss + distill_loss(zu, gu, tu.view(BU, 49, -1), CU['cls'][ui].float())
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item(); step += 1
    S.eval()
    with torch.no_grad():
        pv = torch.cat([S(va_x[k:k + 64])[2] for k in range(0, len(va_x), 64)]).argmax(1).numpy()
    acc = float((pv == y_va).mean())
    log.append(dict(epoch=ep, loss=tot / (len(perm) // B), lab_val_acc=acc, sec=time.time() - t0))
    print(log[-1], flush=True)
torch.save(S.state_dict(), os.path.join(WORK, f'student_{TAG}s{seed}.pt'))
json.dump(log, open(os.path.join(WORK, f'student_{TAG}s{seed}_log.json'), 'w'))

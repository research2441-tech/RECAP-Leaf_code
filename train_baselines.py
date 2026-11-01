"""Supervised lab-only baselines trained with cross-entropy.
Usage: python train_baselines.py ARCH SEED EPOCHS
ARCH in {source_cnn, densenet121, mobilenetv4_conv_small.e2400_r224_in1k, efficientnet_b0}"""
import os, sys, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import timm
from common import manifest, label_space, load_rgb, to_tensor_np, stratified_split, lab_budget, WORK

torch.set_num_threads(2)
arch, seed, EPOCHS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = manifest(); classes, _ = label_space(rows)
lab = [r for r in rows if r['domain'] == 'lab']; cidx = {c: i for i, c in enumerate(classes)}
n_lab = len({(r['crop'], r['disease']) for r in lab})
tr, va = stratified_split(lab, lambda r: (r['crop'], r['disease']), [0.9], seed=100)
tr = lab_budget(tr)


from train_baselines_models import SourceCNN


def build(arch):
    if arch == 'source_cnn': return SourceCNN(n_lab), 1e-3, 128
    m = timm.create_model(arch, pretrained=True, num_classes=n_lab)
    if arch == 'densenet121':                      # CPU budget: fine-tune the last dense block only
        for n, p in m.named_parameters():
            p.requires_grad_(n.startswith('features.denseblock4') or n.startswith('features.norm5') or n.startswith('classifier'))
    return m, 3e-4, 224


torch.manual_seed(seed); rng = np.random.default_rng(seed)
M, lr, res = build(arch)
imgs = np.stack([load_rgb(r['path'], res) for r in tr]); y = torch.tensor([cidx[(r['crop'], r['disease'])] for r in tr])
va_x = torch.tensor(np.stack([to_tensor_np(load_rgb(r['path'], res)) for r in va]))
y_va = np.array([cidx[(r['crop'], r['disease'])] for r in va])
params = [p for p in M.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05); B = 32
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=EPOCHS * (len(tr) // B), pct_start=0.1)
log = []; t0 = time.time()
for ep in range(EPOCHS):
    M.train(); perm = rng.permutation(len(tr))
    for i in range(0, len(perm) - B + 1, B):
        idx = perm[i:i + B]; xs = []
        for im in imgs[idx]:
            im = im.astype(np.float32)
            if rng.random() < 0.5: im = im[:, ::-1]
            a = rng.uniform(0.8, 1.2); b = rng.uniform(-20, 20); s = rng.uniform(0.8, 1.2)
            g = im.mean(2, keepdims=True); im = (g + s * (im - g)) * a + b
            xs.append(to_tensor_np(np.clip(im, 0, 255).astype(np.uint8)))
        loss = F.cross_entropy(M(torch.tensor(np.stack(xs))), y[idx], label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    M.eval()
    with torch.no_grad():
        pv = torch.cat([M(va_x[k:k + 64]) for k in range(0, len(va_x), 64)]).argmax(1).numpy()
    log.append(dict(epoch=ep, lab_val_acc=float((pv == y_va).mean()), sec=time.time() - t0)); print(log[-1], flush=True)
torch.save(M.state_dict(), os.path.join(WORK, f'base_{arch}_s{seed}.pt'))
json.dump(log, open(os.path.join(WORK, f'base_{arch}_s{seed}_log.json'), 'w'))

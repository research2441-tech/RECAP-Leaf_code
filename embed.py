"""Extract embeddings for the fixed evaluation sets.
Usage: python embed.py MODEL SEED   (MODEL in student, teacher, ce_<arch>)"""
import os, sys, json, numpy as np, torch, timm
from common import load_rgb, to_tensor_np, WORK
from recap import Student, Teacher

torch.set_num_threads(2)
model, seed = sys.argv[1], int(sys.argv[2])
meta = json.load(open(os.path.join(WORK, 'sets.json')))
N_LAB = 19


def batches(paths, res, cc):
    for i in range(0, len(paths), 32):
        yield torch.tensor(np.stack([to_tensor_np(load_rgb(p, res, center_crop=cc)) for p in paths[i:i + 32]]))


out = {}
with torch.no_grad():
    if model.startswith('student'):
        S = Student(N_LAB, pretrained=False); S.load_state_dict(torch.load(os.path.join(WORK, f'{model}_s{seed}.pt'))); S.eval()
        for k in ('lab', 'field', 'reg', 'regf'):
            out[f'tok_{k}'] = torch.cat([S.tokens(x).half() for x in batches(meta[k]['paths'], 224, k != 'regf')]).numpy()
    elif model == 'teacher':
        T = Teacher()
        for k in ('lab', 'field', 'reg', 'regf'):
            c, t = [], []
            for x in batches(meta[k]['paths'], 224, k != 'regf'):
                a, b = T(x); c.append(a.half()); t.append(b.half())
            out[f'cls_{k}'] = torch.cat(c).numpy(); out[f'tok_{k}'] = torch.cat(t).numpy()
    elif model.startswith('ce_'):
        arch = model[3:]
        if arch == 'source_cnn':
            from train_baselines_models import SourceCNN
            M = SourceCNN(N_LAB); res = 128
        else:
            M = timm.create_model(arch, pretrained=False, num_classes=N_LAB); res = 224
        M.load_state_dict(torch.load(os.path.join(WORK, f'base_{arch}_s{seed}.pt'))); M.eval()
        for k in ('lab', 'field', 'reg', 'regf'):
            f, l = [], []
            for x in batches(meta[k]['paths'], res, k != 'regf'):
                if arch == 'source_cnn':
                    h = M.h[:4](M.f(x)); f.append(h); l.append(M.h[4](h))
                else:
                    h = M.forward_head(M.forward_features(x), pre_logits=True); f.append(h); l.append(M.get_classifier()(h))
            out[f'feat_{k}'] = torch.cat(f).numpy(); out[f'logit_{k}'] = torch.cat(l).numpy()
np.savez(os.path.join(WORK, f'emb_{model}_s{seed}.npz'), **out)
print('saved', model, seed, {k: v.shape for k, v in out.items()})

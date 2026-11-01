"""E3 frame bank: every capture state reachable within two recaptures, rendered
once per backbone and seed.  Items are PlantDoc field scenes (framing defect =
whole scene instead of the diseased leaf) and a fixed subsample of regional
leaves composited into field backgrounds.  For each state the bank stores the
capture factors and the backbone's global descriptor.
Usage: python recapture_frames.py BACKBONE SEED   (BACKBONE in student, teacher)"""
import os, sys, json, hashlib
import numpy as np, torch
from PIL import Image
from common import WORK, scenes, load_rgb, to_tensor_np
from capture import new_state, render, capture_factors, _draw_blur, _draw_glare, _draw_occ, PRIOR
from recap import Student, Teacher
from evalkit import META, CLASSES, N_LAB, STATS, GlobalHead

torch.set_num_threads(2)
backbone, seed = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("student", 0)
REG_PER_CLASS = 50
TARGETED = ('steady', 'shade', 'clear_thumb', 'move_closer')


def krng(*key):
    h = hashlib.sha256(repr(key).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], 'little'))


def items():
    J = scenes(); cidx = {c: i for i, c in enumerate(CLASSES)}; out = []
    for i, s in enumerate(J['scenes']):
        out.append(dict(kind='scene', idx=i, path=s['path'], box=s['box'], y=cidx[(s['crop'], s['disease'])]))
    bgs = J['field_backgrounds']
    yr = np.array(META['reg']['y']); keep = set()
    for c in np.unique(yr):                                   # fixed subsample of regional leaves
        idx = np.where(yr == c)[0]; keep |= set(krng('regsub', int(c)).permutation(idx)[:REG_PER_CLASS].tolist())
    for i, p in enumerate(META['reg']['paths']):
        if i not in keep: continue
        r = krng('bg', i)
        out.append(dict(kind='reg', idx=i, path=p, bg=bgs[r.integers(len(bgs))], far=float(r.uniform(0.35, 0.6)), y=META['reg']['y'][i]))
    return out


def initial_state(it):
    rng = krng('init', it['kind'], it['idx'])
    if it['kind'] == 'scene':
        sc = np.asarray(Image.open(it['path']).convert('RGB'))
        return new_state(sc, it['box'], rng)
    leaf = load_rgb(it['path'], 256, center_crop=False)
    m = np.asarray(Image.open(it['path'].replace('.jpg', '_mask.png')).resize((256, 256))) > 127
    bg = np.asarray(Image.open(it['bg']).convert('RGB'))
    return new_state(leaf, None, rng, far_scale=it['far'], background=bg, leaf_mask=m)


def step(state, action, key):
    """Deterministic transition; returns None for an action that changes nothing."""
    s = dict(state); r = krng('step', key, action)
    if action == 'retake':
        s['blur'] = _draw_blur(r) if r.random() < PRIOR['p_blur'] else None
        s['glare'] = _draw_glare(r) if r.random() < PRIOR['p_glare'] else None
        s['occ'] = _draw_occ(r) if r.random() < PRIOR['p_occ'] else None
        return s
    field = {'steady': 'blur', 'shade': 'glare', 'clear_thumb': 'occ'}.get(action)
    if field is not None:
        if s[field] is None: return None
        s[field] = None
    else:
        if s['close']: return None
        s['close'] = True
    if s['blur'] is None and r.random() < PRIOR['p_shake_on_retake']:
        s['blur'] = _draw_blur(r)
    return s


def sig(s):
    return (s['blur'], s['glare'], s['occ'], s['close'])


def tree(it):
    """States keyed by action path; identical states share one frame."""
    s0 = initial_state(it); key = (it['kind'], it['idx'])
    nodes = {(): s0}
    for a in TARGETED:
        s1 = step(s0, a, key + (a,))
        if s1 is None: continue
        nodes[(a,)] = s1
        for b in TARGETED:
            s2 = step(s1, b, key + (a, b))
            if s2 is not None: nodes[(a, b)] = s2
    r1 = step(s0, 'retake', key + ('retake',)); nodes[('retake',)] = r1
    nodes[('retake', 'retake')] = step(r1, 'retake', key + ('retake', 'retake'))
    uniq, path2frame = {}, {}
    for p, s in nodes.items():
        g = sig(s)
        if g not in uniq: uniq[g] = (len(uniq), s)
        path2frame[p] = uniq[g][0]
    defects = dict(blur=s0['blur'] is not None, glare=s0['glare'] is not None, occ=s0['occ'] is not None, far=not s0['close'])
    return [s for _, s in sorted(uniq.values(), key=lambda t: t[0])], path2frame, defects


def main():
    if '--support-only' in sys.argv:
        pass
    if backbone == 'student':
        ck = os.path.join(WORK, f'student_s{seed}.pt')
        S = Student(N_LAB, pretrained=False); S.load_state_dict(torch.load(ck)); S.eval(); head = GlobalHead(ck)
        def encode(x):
            with torch.no_grad(): return head(S.tokens(x).mean(1)).numpy()
    else:
        T = Teacher()
        def encode(x):
            c, _ = T(x); return torch.nn.functional.normalize(c, dim=-1).numpy()
    its = items(); index = []; Q, D = [], []
    # clean close-up view of every scene: candidate field-support images (never a test frame of the same split)
    sup = []
    for it in its:
        if it['kind'] != 'scene': continue
        s0 = initial_state(it); s0.update(blur=None, glare=None, occ=None, close=True); sup.append(render(s0))
    Ds = np.concatenate([encode(torch.tensor(np.stack([to_tensor_np(im) for im in sup[i:i + 32]]))) for i in range(0, len(sup), 32)])
    np.save(os.path.join(WORK, f'recap_support_{backbone}_s{seed}.npy'), Ds.astype(np.float16))
    if '--support-only' in sys.argv: return
    buf_img, n_frames = [], 0

    def flush():
        if not buf_img: return
        x = torch.tensor(np.stack([to_tensor_np(im) for im in buf_img]))
        D.append(encode(x).astype(np.float16)); buf_img.clear()

    for n, it in enumerate(its):
        frames, p2f, defects = tree(it)
        index.append(dict(kind=it['kind'], idx=it['idx'], y=int(it['y']), first=n_frames, n=len(frames),
                          paths={'|'.join(p): f for p, f in p2f.items()}, defects=defects))
        for s in frames:
            im = render(s); buf_img.append(im); Q.append(capture_factors(im)); n_frames += 1
            if len(buf_img) == 32: flush()
        if n % 200 == 0: print(n, len(its), n_frames, flush=True)
    flush()
    np.savez(os.path.join(WORK, f'recap_bank_{backbone}_s{seed}.npz'), Q=np.stack(Q), D=np.concatenate(D))
    json.dump(index, open(os.path.join(WORK, f'recap_index_{backbone}_s{seed}.json'), 'w'))
    print('frames', n_frames)


if __name__ == '__main__':
    main()

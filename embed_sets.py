"""Fixed evaluation image sets shared by all experiments.
lab   : capped lab training images (prototype support for lab classes)
field : de-duplicated PlantDoc crops of the studied classes
reg   : regional subset (MangoLeafBD, lemon leaf) photographed on paper
regf  : the same regional leaves composited into field backgrounds with
        capture defects drawn once from the capture prior (seeded per image)."""
import os, json, numpy as np
from PIL import Image
from common import manifest, label_space, load_rgb, stratified_split, lab_budget, regional_subset, scenes, WORK
from capture import new_state, render, plant_cells, capture_factors

def build():
    rows = manifest(); classes, crops = label_space(rows); cidx = {c: i for i, c in enumerate(classes)}
    lab = [r for r in rows if r['domain'] == 'lab']
    tr, _ = stratified_split(lab, lambda r: (r['crop'], r['disease']), [0.9], seed=100); tr = lab_budget(tr)
    field = [r for r in rows if r['domain'] == 'field']
    sub = regional_subset(); reg = [r for r in rows if r['domain'] == 'regional' and r['path'] in sub]
    out = os.path.join(WORK, 'regf'); os.makedirs(out, exist_ok=True)
    bgs = scenes()['field_backgrounds']; regf = []
    for i, r in enumerate(reg):
        q = os.path.join(out, os.path.basename(r['path']))
        if not os.path.exists(q):
            rng = np.random.default_rng(10_000 + i)
            leaf = load_rgb(r['path'], 256, center_crop=False)
            m = np.asarray(Image.open(r['path'].replace('.jpg', '_mask.png')).resize((256, 256))) > 127
            bg = np.asarray(Image.open(bgs[rng.integers(len(bgs))]).convert('RGB'))
            st = new_state(leaf, None, rng, far_scale=float(rng.uniform(0.35, 0.6)), background=bg, leaf_mask=m)
            Image.fromarray(render(st)).save(q, quality=93)
        regf.append(dict(r, path=q))
    sets = dict(lab=tr, field=field, reg=reg, regf=regf)
    meta = {k: dict(paths=[r['path'] for r in v], y=[cidx[(r['crop'], r['disease'])] for r in v]) for k, v in sets.items()}
    meta['classes'] = classes; meta['crops'] = crops
    json.dump(meta, open(os.path.join(WORK, 'sets.json'), 'w'))
    # model-independent per-image quantities
    pc, qf = {}, {}
    for k, v in sets.items():
        imgs = [load_rgb(r['path'], 224, center_crop=(k != 'regf')) for r in v]
        pc[k] = np.stack([plant_cells(im) for im in imgs]); qf[k] = np.stack([capture_factors(im) for im in imgs])
    np.savez(os.path.join(WORK, 'image_stats.npz'), **{f'pc_{k}': v for k, v in pc.items()}, **{f'q_{k}': v for k, v in qf.items()})
    print({k: len(v) for k, v in sets.items()})

if __name__ == '__main__':
    build()

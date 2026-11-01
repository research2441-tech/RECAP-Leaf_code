"""Shared data utilities: label space, image loading, splits."""
import csv, json, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('RECAP_DATA', os.path.join(HERE, '..', 'data'))
WORK = os.environ.get('RECAP_WORK', os.path.join(HERE, '..', 'work'))
os.makedirs(WORK, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def manifest():
    return list(csv.DictReader(open(os.path.join(DATA, 'manifest.csv'))))


def scenes():
    return json.load(open(os.path.join(DATA, 'scenes.json')))


def label_space(rows):
    """Joint label space: (crop, disease) pairs from lab crops plus regional crops."""
    lab = sorted({(r['crop'], r['disease']) for r in rows if r['domain'] == 'lab'})
    reg = sorted({(r['crop'], r['disease']) for r in rows if r['domain'] == 'regional'})
    classes = lab + reg
    crops = sorted({c for c, _ in classes}, key=lambda c: [c for c, _ in classes].index(c))
    return classes, crops


def load_rgb(path, size=224, center_crop=True):
    im = Image.open(path).convert('RGB')
    if center_crop:
        w, h = im.size; s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        w, h = im.size; l, t = (w - size) // 2, (h - size) // 2
        im = im.crop((l, t, l + size, t + size))
    else:
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im)


def to_tensor_np(img):
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


def regional_subset():
    return set(open(os.path.join(DATA, 'regional_subset.txt')).read().split('\n'))


def stratified_split(items, key, fracs, seed):
    """Split a list into len(fracs)+1 parts per stratum; returns list of lists."""
    rng = np.random.default_rng(seed); groups = {}
    for it in items: groups.setdefault(key(it), []).append(it)
    parts = [[] for _ in range(len(fracs) + 1)]
    for k in sorted(groups):
        g = groups[k]; idx = rng.permutation(len(g)); n = len(g); start = 0
        for j, f in enumerate(fracs):
            m = int(round(f * n)) if f < 1 else int(f)
            parts[j] += [g[i] for i in idx[start:start + m]]; start += m
        parts[-1] += [g[i] for i in idx[start:]]
    return parts


def lab_budget(rows, per_class=200):
    """Deterministic per-class cap on lab training images (CPU training budget)."""
    out, seen = [], {}
    for r in rows:
        k = (r['crop'], r['disease']); seen[k] = seen.get(k, 0) + 1
        if seen[k] <= per_class: out.append(r)
    return out

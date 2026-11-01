"""PlantDoc detection scenes for the recapture benchmark.
Keeps scenes whose boxes all map to a single studied class, stores the largest
box, removes near-duplicates, and collects background scenes of species that are
not studied (used only as backgrounds when compositing regional leaves)."""
import os, sys, csv, json, collections
import numpy as np
from PIL import Image
from prepare_data import PD  # class mapping shared with the classification manifest

ROOT = sys.argv[1]
DET = os.path.join(ROOT, 'plantdoc_det')
OUT = os.path.abspath(os.path.join(ROOT, 'scenes')); os.makedirs(OUT, exist_ok=True)
BG_SPECIES = {'Blueberry leaf', 'Raspberry leaf', 'Soyabean leaf', 'Strawberry leaf', 'Peach leaf', 'Cherry leaf'}

def dhash(im):
    a = np.asarray(im.convert('L').resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()

boxes = collections.defaultdict(list)
for f in ['train_labels.csv', 'test_labels.csv']:
    sub = 'TRAIN' if f.startswith('train') else 'TEST'
    for r in csv.DictReader(open(os.path.join(DET, f))):
        boxes[(sub, r['filename'])].append((r['class'], [int(float(r[k])) for k in ('xmin', 'ymin', 'xmax', 'ymax')],
                                           int(float(r['width'])), int(float(r['height']))))
scenes, bgs, hashes = [], [], []
for (sub, fn), bl in sorted(boxes.items()):
    cls = {c for c, _, _, _ in bl}
    p = os.path.join(DET, sub, fn)
    if not os.path.exists(p): continue
    try:
        im = Image.open(p).convert('RGB')
    except Exception:
        continue
    h = dhash(im)
    if any(np.count_nonzero(h != g) <= 4 for g in hashes): continue
    hashes.append(h)
    W, H = im.size
    if cls <= BG_SPECIES:
        q = os.path.join(OUT, f'bg_{len(bgs)}.jpg'); im.resize((320, 320)).save(q, quality=90); bgs.append(q); continue
    if len(cls) != 1 or next(iter(cls)) not in PD: continue
    c = next(iter(cls)); crop, dis = PD[c]
    # boxes are given in the annotated width/height; rescale to the real image size
    sx, sy = W / max(1, bl[0][2]), H / max(1, bl[0][3])
    bx = max(bl, key=lambda b: (b[1][2] - b[1][0]) * (b[1][3] - b[1][1]))[1]
    bx = [bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy]
    s = 448 / max(W, H); im2 = im.resize((max(1, round(W * s)), max(1, round(H * s))))
    bx = [round(v * s) for v in bx]
    bx = [max(0, bx[0]), max(0, bx[1]), min(im2.size[0], bx[2]), min(im2.size[1], bx[3])]   # clamp to image
    if bx[2] - bx[0] < 16 or bx[3] - bx[1] < 16: continue
    q = os.path.join(OUT, f'scene_{len(scenes)}.jpg'); im2.save(q, quality=92)
    area = (bx[2] - bx[0]) * (bx[3] - bx[1]) / (im2.size[0] * im2.size[1])
    scenes.append(dict(path=q, crop=crop, disease=dis, box=bx, box_area=area, n_boxes=len(bl)))
# field backgrounds for compositing: vegetation-dominated scenes without white studio backgrounds
import cv2
from capture import _vegetation_mask
field_bgs = []
for b in bgs:
    a = np.asarray(Image.open(b).convert('RGB').resize((160, 160))); hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    if np.mean((hsv[..., 2] > 215) & (hsv[..., 1] < 30)) < 0.05 and _vegetation_mask(a).mean() > 0.45: field_bgs.append(b)
json.dump(dict(scenes=scenes, backgrounds=bgs, field_backgrounds=field_bgs), open(os.path.join(ROOT, 'scenes.json'), 'w'))
print(len(scenes), 'scenes;', len(bgs), 'backgrounds')
print(collections.Counter((s['crop'], s['disease']) for s in scenes))
print('median box area fraction', np.median([s['box_area'] for s in scenes]))

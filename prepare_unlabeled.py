"""Unlabeled in-the-wild distillation corpus of NON-target species only:
iNatAg-mini photographs of eight crops outside the studied six, PlantDoc crops of
non-target species that never appear as compositing backgrounds.
No image of tomato, potato, maize, bell pepper, mango or citrus is included."""
import os, sys, glob, zipfile, io, json
from PIL import Image
ROOT = sys.argv[1]; OUT = os.path.join(ROOT, 'unlab'); os.makedirs(OUT, exist_ok=True)
# species whose scenes serve as compositing backgrounds (blueberry, raspberry, soybean,
# strawberry, peach, cherry) are excluded so that no test background is seen in training
NON_TARGET_PD = ['Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Squash Powdery mildew leaf', 'grape leaf', 'grape leaf black rot']
paths = []
def save(im, name):
    im = im.convert('RGB'); w, h = im.size; s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
    p = os.path.abspath(os.path.join(OUT, name)); im.save(p, quality=90); paths.append(p)
for z in sorted(glob.glob(os.path.join(ROOT, 'inat', '*.zip'))):
    sp = os.path.basename(z)[:-4]
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if n.lower().endswith(('.jpg', '.jpeg', '.png')):
                try: save(Image.open(io.BytesIO(zf.read(n))), f'inat_{sp}_{len(paths)}.jpg')
                except Exception: pass
for sub in ('train', 'test'):
    for c in NON_TARGET_PD:
        for f in sorted(glob.glob(os.path.join(ROOT, 'plantdoc_cls', sub, c, '*'))):
            try: save(Image.open(f), f'pd_{len(paths)}.jpg')
            except Exception: pass
open(os.path.join(ROOT, 'unlabeled.txt'), 'w').write('\n'.join(paths))
print(len(paths))

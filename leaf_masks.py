"""GrabCut leaf masks for regional leaves photographed on paper (used only to
composite leaves into field backgrounds for the capture emulation)."""
import sys, csv, os, random
import numpy as np, cv2
from PIL import Image
man = sys.argv[1]; rows = list(csv.DictReader(open(man)))
rng = random.Random(0); sel = []
for crop, per in (('mango', 160), ('citrus', 10 ** 6)):
    by = {}
    for r in rows:
        if r['crop'] == crop: by.setdefault(r['disease'], []).append(r)
    for d, rs in sorted(by.items()):
        rng.shuffle(rs); sel += rs[:per]
for i, r in enumerate(sel):
    q = r['path'].replace('.jpg', '_mask.png')
    if os.path.exists(q): continue
    img = np.asarray(Image.open(r['path']).convert('RGB').resize((192, 192)))
    m = np.zeros(img.shape[:2], np.uint8); b = np.zeros((1, 65)); f = np.zeros((1, 65))
    cv2.grabCut(img, m, (6, 6, 180, 180), b, f, 4, cv2.GC_INIT_WITH_RECT)
    fg = ((m == 1) | (m == 3)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1: fg = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
    Image.fromarray(fg * 255).save(q)
with open(man.replace('manifest.csv', 'regional_subset.txt'), 'w') as fh:
    fh.write('\n'.join(r['path'] for r in sel))
print('masks', len(sel))

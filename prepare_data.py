"""Build a unified manifest of lab (PlantVillage), field (PlantDoc) and regional
(MangoLeafBD, lemon leaf) images, resized to 256 px on the short side.
Near-duplicate PlantDoc images are removed with a 64-bit difference hash."""
import os, io, sys, csv, glob, random, hashlib
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

ROOT = sys.argv[1] if __name__ == "__main__" else None

rng = random.Random(0)

PV = {
 'Pepper,_bell___healthy': ('pepper', 'healthy'),
 'Pepper,_bell___Bacterial_spot': ('pepper', 'bacterial_spot'),
 'Potato___healthy': ('potato', 'healthy'),
 'Potato___Early_blight': ('potato', 'early_blight'),
 'Potato___Late_blight': ('potato', 'late_blight'),
 'Corn_(maize)___healthy': ('maize', 'healthy'),
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': ('maize', 'gray_leaf_spot'),
 'Corn_(maize)___Common_rust_': ('maize', 'common_rust'),
 'Corn_(maize)___Northern_Leaf_Blight': ('maize', 'northern_leaf_blight'),
 'Tomato___healthy': ('tomato', 'healthy'),
 'Tomato___Bacterial_spot': ('tomato', 'bacterial_spot'),
 'Tomato___Early_blight': ('tomato', 'early_blight'),
 'Tomato___Late_blight': ('tomato', 'late_blight'),
 'Tomato___Leaf_Mold': ('tomato', 'leaf_mold'),
 'Tomato___Septoria_leaf_spot': ('tomato', 'septoria_leaf_spot'),
 'Tomato___Spider_mites Two-spotted_spider_mite': ('tomato', 'spider_mites'),
 'Tomato___Target_Spot': ('tomato', 'target_spot'),
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ('tomato', 'yellow_leaf_curl_virus'),
 'Tomato___Tomato_mosaic_virus': ('tomato', 'mosaic_virus'),
}
PD = {
 'Bell_pepper leaf': ('pepper', 'healthy'),
 'Bell_pepper leaf spot': ('pepper', 'bacterial_spot'),
 'Corn Gray leaf spot': ('maize', 'gray_leaf_spot'),
 'Corn leaf blight': ('maize', 'northern_leaf_blight'),
 'Corn rust leaf': ('maize', 'common_rust'),
 'Potato leaf early blight': ('potato', 'early_blight'),
 'Potato leaf late blight': ('potato', 'late_blight'),
 'Tomato Early blight leaf': ('tomato', 'early_blight'),
 'Tomato Septoria leaf spot': ('tomato', 'septoria_leaf_spot'),
 'Tomato leaf': ('tomato', 'healthy'),
 'Tomato leaf bacterial spot': ('tomato', 'bacterial_spot'),
 'Tomato leaf late blight': ('tomato', 'late_blight'),
 'Tomato leaf mosaic virus': ('tomato', 'mosaic_virus'),
 'Tomato leaf yellow virus': ('tomato', 'yellow_leaf_curl_virus'),
 'Tomato mold leaf': ('tomato', 'leaf_mold'),
}
def main():
    global OUT
    OUT = os.path.join(ROOT, 'img256')
    os.makedirs(OUT, exist_ok=True)
    PV_PER_CLASS = 300   # lab training budget per class (CPU-bound study)

    def save(im, name):
        im = im.convert('RGB'); w, h = im.size; s = 256 / min(w, h)
        im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
        p = os.path.abspath(os.path.join(OUT, name)); im.save(p, quality=92); return p

    def dhash(path):
        im = Image.open(path).convert('L').resize((9, 8), Image.BILINEAR)
        a = np.asarray(im, dtype=np.int16); return (a[:, 1:] > a[:, :-1]).flatten()

    rows = []
    for folder, (crop, dis) in PV.items():
        files = sorted(glob.glob(os.path.join(ROOT, 'pv/raw/color', folder, '*')))
        rng.shuffle(files)
        for i, f in enumerate(files[:PV_PER_CLASS]):
            p = save(Image.open(f), f'pv_{crop}_{dis}_{i}.jpg')
            rows.append(dict(path=p, domain='lab', source='PlantVillage', crop=crop, disease=dis))

    pd_rows, hashes = [], []
    for sub in ['train', 'test']:
        for folder, (crop, dis) in PD.items():
            for i, f in enumerate(sorted(glob.glob(os.path.join(ROOT, 'plantdoc_cls', sub, folder, '*')))):
                try: im = Image.open(f); im.load()
                except Exception: continue
                p = save(im, f'pd_{sub}_{crop}_{dis}_{i}.jpg')
                h = dhash(p)
                if any(np.count_nonzero(h != g) <= 4 for g in hashes):   # near duplicate
                    os.remove(p); continue
                hashes.append(h)
                pd_rows.append(dict(path=p, domain='field', source='PlantDoc', crop=crop, disease=dis))
    rows += pd_rows

    def from_parquet(fn, crop, source, names, filt=None):
        t = pq.read_table(os.path.join(ROOT, fn)).to_pandas()
        for i, r in t.iterrows():
            if filt is not None and not filt(r): continue
            im = Image.open(io.BytesIO(r['image']['bytes']))
            dis = names[int(r['label'])].lower().replace(' ', '_')
            p = save(im, f'{crop}_{dis}_{i}.jpg')
            rows.append(dict(path=p, domain='regional', source=source, crop=crop, disease=dis))

    from_parquet('mango.parquet', 'mango', 'MangoLeafBD',
                 ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'])
    from_parquet('lemon.parquet', 'citrus', 'LemonLeaf',
                 ['Anthracnose', 'Bacterial Blight', 'Citrus Canker', 'Curl Virus', 'Deficiency Leaf', 'Dry Leaf', 'Healthy', 'Sooty Mould', 'Spider Mites'])

    with open(os.path.join(ROOT, 'manifest.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['path', 'domain', 'source', 'crop', 'disease']); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(Counter((r['domain'], r['crop']) for r in rows))
    print('PlantDoc kept after dedup:', len(pd_rows))


if __name__ == '__main__':
    main()

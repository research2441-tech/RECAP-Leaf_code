"""E6: parameters, multiply-accumulates and CPU latency per frame (batch 1).
Latency is measured on the experiment host with 1 and 2 threads; it is a proxy
for, not a measurement on, single-board computers."""
import os, json, time, numpy as np, torch, timm
from torch.utils.flop_counter import FlopCounterMode
from common import WORK
from recap import Student, Teacher
from train_baselines_models import SourceCNN
from capture import capture_factors
import cv2

res = []
x224 = torch.randn(1, 3, 224, 224); x128 = torch.randn(1, 3, 128, 128)
models = {
    'RECAP-Leaf student (MobileNetV4-Conv-S + projection)': (lambda: Student(19, pretrained=False).eval(), x224, lambda m, x: m.tokens(x)),
    'DINOv2-S/14 teacher': (lambda: Teacher().eval(), x224, lambda m, x: m(x)),
    'DenseNet-121': (lambda: timm.create_model('densenet121', pretrained=False, num_classes=19).eval(), x224, lambda m, x: m(x)),
    'MobileNetV4-Conv-S (CE)': (lambda: timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=19).eval(), x224, lambda m, x: m(x)),
    'Compact three-block CNN': (lambda: SourceCNN(19).eval(), x128, lambda m, x: m(x)),
}
for name, (mk, x, fwd) in models.items():
    m = mk(); params = sum(p.numel() for p in m.parameters()) / 1e6
    with FlopCounterMode(display=False) as fc, torch.no_grad(): fwd(m, x)
    macs = fc.get_total_flops() / 2 / 1e9
    row = dict(model=name, params_M=params, GMACs=macs)
    for th in (1, 2):
        torch.set_num_threads(th)
        with torch.no_grad():
            for _ in range(5): fwd(m, x)
            ts = []
            for _ in range(30):
                t0 = time.perf_counter(); fwd(m, x); ts.append(time.perf_counter() - t0)
        row[f'ms_{th}t'] = 1000 * float(np.median(ts))
    res.append(row); print(row, flush=True)
img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
ts = []
for _ in range(50):
    t0 = time.perf_counter(); capture_factors(img); ts.append(time.perf_counter() - t0)
res.append(dict(model='capture factors (five measurements)', params_M=0.0, GMACs=float('nan'), ms_1t=1000 * float(np.median(ts)), ms_2t=float('nan')))
print(res[-1])
json.dump(res, open(os.path.join(WORK, 'results', 'efficiency.json'), 'w'))

"""Capture emulation and capture-factor measurement for RECAP-Leaf.

A capture state stores what the camera saw (scene + optional leaf box) and the
acquisition defects present in the frame (blur, specular glare, thumb occlusion,
distance).  Rendering a state yields the frame that the device would receive.
Recapture actions remove the defect they target; a generic retake redraws the
photometric defects from the prior and leaves the framing unchanged.
"""
import numpy as np
import cv2

PRIOR = dict(p_blur=0.35, p_glare=0.30, p_occ=0.20, p_shake_on_retake=0.10)
ACTIONS = ('steady', 'shade', 'clear_thumb', 'move_closer')
FACTORS = ('blur', 'glare', 'scale', 'occlusion', 'clutter')
ACTION_FACTORS = {'steady': (0,), 'shade': (1,), 'clear_thumb': (3,), 'move_closer': (2, 4)}


def _draw_blur(rng):
    if rng.random() < 0.5:
        return ('gauss', float(rng.uniform(1.8, 3.6)), 0.0)
    return ('motion', float(rng.integers(9, 19)), float(rng.uniform(0, 180)))


def _draw_glare(rng):
    return (float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.3, 0.7)),
            float(rng.uniform(0.14, 0.30)), float(rng.uniform(0.08, 0.20)),
            float(rng.uniform(0, 180)), float(rng.uniform(0.65, 0.92)))


def _draw_occ(rng):
    side = int(rng.integers(0, 4))
    return (side, float(rng.uniform(0.25, 0.75)), float(rng.uniform(0.16, 0.26)),
            float(rng.uniform(0.30, 0.45)), tuple(float(v) for v in rng.uniform(-18, 18, 3)))


def new_state(scene, box, rng, far_scale=None, background=None, leaf_mask=None):
    """scene: HxWx3 uint8 RGB. box: (x0,y0,x1,y1) of the diagnosed leaf or None.
    For regional crops photographed on white paper, `background` and `far_scale`
    composite the leaf into a field background to emulate a distant capture."""
    return dict(scene=scene, box=box, close=box is None and far_scale is None,
                far_scale=far_scale, background=background, leaf_mask=leaf_mask,
                blur=_draw_blur(rng) if rng.random() < PRIOR['p_blur'] else None,
                glare=_draw_glare(rng) if rng.random() < PRIOR['p_glare'] else None,
                occ=_draw_occ(rng) if rng.random() < PRIOR['p_occ'] else None)


def apply_action(state, action, rng):
    s = dict(state)
    if action == 'steady':
        s['blur'] = None
    elif action == 'shade':
        s['glare'] = None
    elif action == 'clear_thumb':
        s['occ'] = None
    elif action == 'move_closer':
        s['close'] = True
    elif action == 'retake':          # unguided retake: photometric defects redrawn
        s['blur'] = _draw_blur(rng) if rng.random() < PRIOR['p_blur'] else None
        s['glare'] = _draw_glare(rng) if rng.random() < PRIOR['p_glare'] else None
        s['occ'] = _draw_occ(rng) if rng.random() < PRIOR['p_occ'] else None
        return s
    if s['blur'] is None and rng.random() < PRIOR['p_shake_on_retake']:
        s['blur'] = _draw_blur(rng)
    return s


def _leaf_on_white_mask(img):
    """Leaf mask for leaves photographed on paper: pixels connected to the border
    whose colour is close to the median border colour are treated as paper."""
    f = img.astype(np.float32)
    ref = np.median(np.concatenate([f[0], f[-1], f[:, 0], f[:, -1]]), axis=0)
    near = (np.linalg.norm(f - ref, axis=2) < 38).astype(np.uint8)
    n, lab = cv2.connectedComponents(near, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    m = (~np.isin(lab, list(border))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(m.astype(np.float32), (5, 5), 0)[..., None]


def _geometry(state, size):
    sc = state['scene']
    if state['far_scale'] is not None:                       # composited regional leaf
        bg = cv2.resize(state['background'], (size, size), interpolation=cv2.INTER_AREA)
        scale = 0.92 if state['close'] else state['far_scale']
        k = max(8, int(size * scale))
        leaf = cv2.resize(sc, (k, k), interpolation=cv2.INTER_AREA)
        if state.get('leaf_mask') is not None:
            m = cv2.resize(state['leaf_mask'].astype(np.float32), (k, k), interpolation=cv2.INTER_LINEAR)
            m = cv2.GaussianBlur(m, (5, 5), 0)[..., None]
        else:
            m = _leaf_on_white_mask(leaf)
        y0 = (size - k) // 2; x0 = (size - k) // 2
        out = bg.astype(np.float32).copy()
        out[y0:y0 + k, x0:x0 + k] = m * leaf + (1 - m) * out[y0:y0 + k, x0:x0 + k]
        return out.clip(0, 255).astype(np.uint8)
    if state['box'] is not None and state['close']:
        x0, y0, x1, y1 = state['box']; H, W = sc.shape[:2]
        mx, my = 0.12 * (x1 - x0), 0.12 * (y1 - y0)
        x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        x1, y1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
        sc = sc[y0:y1, x0:x1]
    return cv2.resize(sc, (size, size), interpolation=cv2.INTER_AREA)


def render(state, size=224):
    img = _geometry(state, size).astype(np.float32)
    if state['occ'] is not None:
        side, pos, rad_a, rad_b, jit = state['occ']
        c = np.array([224, 172, 140], np.float32) + np.array(jit, np.float32)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
        cx, cy = [(pos, 0.0), (1.0, pos), (pos, 1.0), (0.0, pos)][side]
        d = ((xx - cx) / rad_a) ** 2 + ((yy - cy) / rad_b) ** 2 if side % 2 == 0 else \
            ((xx - cx) / rad_b) ** 2 + ((yy - cy) / rad_a) ** 2
        m = np.clip((1.0 - d) * 6.0, 0, 1)[..., None]
        shade = (0.82 + 0.18 * np.clip(1 - d, 0, 1))[..., None]
        img = m * (c * shade) + (1 - m) * img
    if state['glare'] is not None:
        gx, gy, ra, rb, ang, a = state['glare']
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
        t = np.deg2rad(ang); u = (xx - gx) * np.cos(t) + (yy - gy) * np.sin(t)
        v = -(xx - gx) * np.sin(t) + (yy - gy) * np.cos(t)
        g = np.exp(-((u / ra) ** 2 + (v / rb) ** 2) * 2.0)[..., None] * a
        img = img * (1 - g) + 255.0 * g
    img = img.clip(0, 255).astype(np.uint8)
    if state['blur'] is not None:
        kind, p, ang = state['blur']
        if kind == 'gauss':
            img = cv2.GaussianBlur(img, (0, 0), p)
        else:
            L = int(p); k = np.zeros((L, L), np.float32); k[L // 2, :] = 1.0
            R = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), ang, 1.0)
            k = cv2.warpAffine(k, R, (L, L)); k /= k.sum() + 1e-8
            img = cv2.filter2D(img, -1, k)
    return img


# ---------------------------------------------------------------- factor measurement
def _vegetation_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    m = ((h > 25) & (h < 160) & (s > 0.18) & (v > 0.12)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def capture_factors(img):
    """Raw capture factors of a 224x224 RGB frame: blur, glare, scale, occlusion, clutter."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    q_blur = -np.log10(cv2.Laplacian(g, cv2.CV_64F).var() + 1.0)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    q_glare = float(np.mean((hsv[..., 2] > 0.93) & (hsv[..., 1] < 0.14)))
    veg = _vegetation_mask(img)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(veg, connectivity=8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])); leaf = (lab == k).astype(np.uint8)
    else:
        leaf = np.zeros_like(veg)
    q_scale = 1.0 - leaf.mean()
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    skin = (ycc[..., 1] > 138) & (ycc[..., 1] < 175) & (ycc[..., 2] > 92) & (ycc[..., 2] < 128) & (ycc[..., 0] > 120)
    q_occ = float(np.mean(skin & (veg == 0)))
    edges = cv2.Canny(g, 60, 150) > 0
    outside = cv2.dilate(leaf, np.ones((11, 11), np.uint8)) == 0
    q_clutter = float(edges[outside].mean()) if outside.any() else 0.0
    return np.array([q_blur, q_glare, q_scale, q_occ, q_clutter], np.float32)


def plant_cells(img, grid=7):
    """Fraction of plant tissue (green/yellow tissue with enclosed lesions filled)
    in each cell of a grid x grid partition of a 224x224 frame."""
    veg = _vegetation_mask(img)
    filled = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full = np.zeros_like(filled); cv2.drawContours(full, cnts, -1, 1, thickness=-1)
    return cv2.resize(full.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA).flatten()

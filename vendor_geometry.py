"""PA / FL / MT from aponeurosis and fascicle probability maps, in millimetres.

Host protocol (forum answers by the organizers):
* MT: three *vertical* image lines (left / middle / right) between the aponeuroses, averaged.
* PA: angle between fascicle and the slope of the *deep* aponeurosis, at fascicle insertions.
* FL: three fascicles chosen to minimise extrapolation, extended linearly to both aponeuroses.

All coordinates are physical: X = x * mx, Y = y * my (mm per probability-map pixel), so
non-square pixels are handled. Every choice that is not fixed by the protocol is a key in
`DEFAULT`, so variants can be compared on the expert-labelled OSF benchmark.
"""
import cv2
import numpy as np

DEFAULT = dict(
    apo_thr=0.5,
    mt_mode="inner_vertical",   
    mt_xs=(0.2, 0.5, 0.8),      
    pa_mode="depth_fit",        
    pa_at=0.9,                  
    fl_mode="extend_inner",     
    fl_select="weighted",       
    fl_factor=1.0,
)
FALLBACK = dict(pa_deg=15.0, fl_mm=70.0, mt_mm=20.0)


def _column_profile(mask_comp, prob, cols):
    """Per column of one component: weighted centre, top edge and bottom edge (map pixels)."""
    ys, xs = np.nonzero(mask_comp)
    cy, top, bot = [], [], []
    for c in cols:
        yc = ys[xs == c]
        cy.append(np.average(yc, weights=prob[yc, c]))
        top.append(yc.min() - 0.5)
        bot.append(yc.max() + 0.5)
    return np.array(cy), np.array(top), np.array(bot)


def aponeuroses(p_apo, mx, my, thr=0.5, merge_mm=4.0, min_gap_mm=6.0, min_rel_extent=0.35):
    """Superficial and deep aponeurosis as dicts of quadratic fits (Y in mm as a function of X in mm).

    Each band has fits for its centreline ("c"), upper edge ("top") and lower edge ("bot").
    Components less than `merge_mm` apart in depth are merged (fragmented / double lines).
    """
    h, w = p_apo.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((p_apo > thr).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        if st[i, 2] < 0.03 * w:
            continue
        m = lab == i
        cols = np.unique(np.nonzero(m)[1])
        cy, top, bot = _column_profile(m, p_apo, cols)
        comps.append(dict(X=cols * mx, c=cy * my, top=top * my, bot=bot * my))
    if not comps:
        return None
    xc = np.median(np.concatenate([c["X"] for c in comps]))
    depth = [np.polyval(np.polyfit(c["X"], c["c"], 1), xc) if len(c["X"]) > 1 else c["c"].mean() for c in comps]
    bands = []
    for k in np.argsort(depth):
        if bands and depth[k] - bands[-1]["d"] < merge_mm:
            b = bands[-1]
            for key in ("X", "c", "top", "bot"):
                b[key] = np.concatenate([b[key], comps[k][key]])
        else:
            bands.append(dict(d=depth[k], **comps[k]))
    ext = np.array([b["X"].max() - b["X"].min() for b in bands])
    bands = [b for b, e in zip(bands, ext) if e >= min_rel_extent * ext.max()]
    out = []
    for b in bands:
        if out and b["d"] - out[-1]["d"] < min_gap_mm:
            continue
        deg = 2 if b["X"].max() - b["X"].min() > 15 else 1
        b["fit"] = {k: np.polyfit(b["X"], b[k], deg) for k in ("c", "top", "bot")}
        b["lo"], b["hi"] = b["X"].min(), b["X"].max()
        out.append(b)
        if len(out) == 2:
            break
    return out if len(out) == 2 else None


def _shared_span(sup, deep):
    lo, hi = max(sup["lo"], deep["lo"]), min(sup["hi"], deep["hi"])
    if hi - lo < 3:
        lo, hi = min(sup["lo"], deep["lo"]), max(sup["hi"], deep["hi"])
    return lo, hi


def muscle_thickness(sup, deep, mode, xs_rel):
    lo, hi = _shared_span(sup, deep)
    X = lo + np.array(xs_rel) * (hi - lo)
    if mode == "inner_vertical":
        d = np.polyval(deep["fit"]["top"], X) - np.polyval(sup["fit"]["bot"], X)
    elif mode == "centre_vertical":
        d = np.polyval(deep["fit"]["c"], X) - np.polyval(sup["fit"]["c"], X)
    elif mode == "centre_minus_half":
        th = lambda b: np.median(b["bot"] - b["top"])
        d = np.polyval(deep["fit"]["c"], X) - np.polyval(sup["fit"]["c"], X) - (th(sup) + th(deep)) / 2
    else:
        raise ValueError(mode)
    return float(np.mean(d))


def fascicle_segments(p_fasc, mx, my, thr=0.5, min_len_mm=4.0):
    """Long, thin connected components -> (cx, cy, angle_deg, length_mm) in physical units."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((p_fasc > thr).astype(np.uint8), connectivity=8)
    segs = []
    for i in range(1, n):
        if st[i, 4] < 6:
            continue
        ys, xs = np.nonzero(lab == i)
        P = np.stack([xs * mx, ys * my], 1)
        c = P.mean(0)
        _, s, vt = np.linalg.svd(P - c, full_matrices=False)
        proj = (P - c) @ vt[0]
        L = proj.max() - proj.min()
        if L < min_len_mm or s[0] < 3 * (s[1] + 1e-6):
            continue
        ang = (np.degrees(np.arctan2(vt[0][1], vt[0][0])) + 90) % 180 - 90
        segs.append((c[0], c[1], ang, L))
    return segs


def _slope_deg(poly, X):
    return float(np.degrees(np.arctan(np.polyval(np.polyder(poly), X))))


def _intersect(cx, cy, ang, poly):
    """Intersection of the line through (cx, cy) at `ang` with the polynomial curve nearest to cx."""
    k = np.tan(np.radians(ang))
    q = np.array(poly, dtype=float).copy()
    q[-1] -= cy - k * cx
    q[-2] -= k
    r = np.roots(q)
    r = r[np.isreal(r)].real
    if len(r) == 0:
        return None
    X = r[np.argmin(np.abs(r - cx))]
    return np.array([X, cy + k * (X - cx)])


def _wmedian(v, w):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, cw[-1] / 2)])


def measure(p_apo, p_fasc, mx, my, **kw):
    """Measure one image. Returns dict with pa_deg, fl_mm, mt_mm, ok and per-segment data."""
    P = {**DEFAULT, **kw}
    out = dict(FALLBACK, ok=False, segs=np.zeros((0, 4)))
    apo = aponeuroses(p_apo, mx, my, thr=P["apo_thr"])
    if apo is None:
        return out
    sup, deep = apo
    out["apo"] = apo
    out["mt_mm"] = muscle_thickness(sup, deep, P["mt_mode"], P["mt_xs"])
    inner = P["fl_mode"] != "extend_centre"
    sup_line = sup["fit"]["bot"] if inner else sup["fit"]["c"]
    deep_line = deep["fit"]["top"] if inner else deep["fit"]["c"]

    rows = []  
    for cx, cy, ang, L in fascicle_segments(p_fasc, mx, my):
        ys, yd = np.polyval(sup["fit"]["c"], cx), np.polyval(deep["fit"]["c"], cx)
        if not (ys - 2 < cy < yd + 2):
            continue
        pa = abs(((ang - _slope_deg(deep["fit"]["c"], np.clip(cx, deep["lo"], deep["hi"]))) + 90) % 180 - 90)
        if not (2 < pa < 60):
            continue
        p1, p2 = _intersect(cx, cy, ang, sup_line), _intersect(cx, cy, ang, deep_line)
        fl = np.linalg.norm(p1 - p2) if p1 is not None and p2 is not None else np.nan
        rows.append(((cy - ys) / max(yd - ys, 1e-6), pa, L, fl))
    if not rows:
        return out
    S = np.array(rows)
    out["segs"] = S
    out["ok"] = True

    pa = None
    if P["pa_mode"] == "depth_fit":
        pa = pa_depth_fit(S, P["pa_at"])
    elif P["pa_mode"] == "deep_half":
        deep_rows = S[S[:, 0] > 0.5]
        if len(deep_rows) >= 2:
            pa = _wmedian(deep_rows[:, 1], deep_rows[:, 2])
    out["pa_deg"] = pa if pa is not None else _wmedian(S[:, 1], S[:, 2])

    fl_sin = out["mt_mm"] / np.sin(np.radians(max(out["pa_deg"], 3.0)))
    good = np.isfinite(S[:, 3]) & (S[:, 3] > 10) & (S[:, 3] < 300)
    if not good.any():
        fl_ext = fl_sin
    elif P["fl_select"] == "top3":
        g = S[good]
        fl_ext = float(np.mean(g[np.argsort(-g[:, 2])[:3], 3]))
    elif P["fl_select"] == "visible":
        fl_ext = _wmedian(S[good, 3], S[good, 2] ** 2 / S[good, 3])
    else:
        fl_ext = _wmedian(S[good, 3], S[good, 2])
    if P["fl_mode"] == "mt_over_sin":
        fl = fl_sin
    elif P["fl_mode"] == "avg":
        fl = 0.5 * (fl_ext + fl_sin)
    else:
        fl = fl_ext
    out["fl_mm"] = fl * P["fl_factor"]
    return out


def pa_depth_fit(S, at, min_n=5):
    """Fit PA against depth and evaluate only within observed segment support."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[1] != 4:
        raise ValueError("Expected segment columns: relative_depth, PA, visible_length, extended_length")
    if not np.isfinite(S[:, :3]).all() or (S[:, 2] <= 0).any():
        raise ValueError("Invalid depth/angle/visible-length measurement")
    if len(S) < min_n or np.ptp(S[:, 0]) < 0.2:
        return None
    # Conservative change: use the observed minimum and maximum, keeping at=0.9.
    # Applied identically to each image and each content-similar frame group.
    at = float(np.clip(at, S[:, 0].min(), S[:, 0].max()))
    return float(np.polyval(np.polyfit(S[:, 0], S[:, 1], 1, w=np.sqrt(S[:, 2])), at))
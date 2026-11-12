"""Pennation angle from raw image texture, independent of the fascicle segmentation model.

The image is resampled to isotropic pixels (r px/mm) so angles are physical. Inside the deep part of
the muscle belly (between the inner aponeurosis edges found by the aponeurosis model), a high-pass
filtered image is projected onto candidate directions: when the direction matches the fascicles,
the bright fascicle lines fall into the same projection bins and the variance of the bin means peaks
(a discrete Radon-transform maximum). Near-horizontal directions are excluded because speckle and
residual aponeurosis brightness are elongated laterally.
"""
import cv2
import numpy as np

from vendor_geometry import _slope_deg


def texture_angle(gray, sx, sy, apo, r=4.0, depth_lo=0.45, depth_hi=0.92, margin_mm=1.0,
                  angles=np.arange(-50, 50.5, 0.5), min_abs=4.0):
    """Return (fascicle angle in image coords, PA vs deep aponeurosis) in degrees, or (nan, nan)."""
    sup, deep = apo
    H0, W0 = gray.shape
    iso = cv2.resize(gray, (max(int(W0 * r / sx), 8), max(int(H0 * r / sy), 8)), interpolation=cv2.INTER_AREA)
    iso = iso.astype(np.float32)
    hp = iso - cv2.GaussianBlur(iso, (0, 0), 2.0 * r)
    h, w = hp.shape
    X = (np.arange(w) + 0.5) / r
    lo, hi = max(sup["lo"], deep["lo"]), min(sup["hi"], deep["hi"])
    ytop = np.polyval(sup["fit"]["bot"], X) + margin_mm
    ybot = np.polyval(deep["fit"]["top"], X) - margin_mm
    y0 = ytop + depth_lo * (ybot - ytop)
    y1 = ytop + depth_hi * (ybot - ytop)
    Y = (np.arange(h)[:, None] + 0.5) / r
    roi = (Y >= y0[None]) & (Y <= y1[None]) & (X[None] >= lo) & (X[None] <= hi) & (ybot > ytop)[None]
    if roi.sum() < 400:
        return np.nan, np.nan
    yy, xx = np.nonzero(roi)
    v = hp[yy, xx]
    v = (v - v.mean()) / (v.std() + 1e-6)
    best, best_a = -1, np.nan
    for a in angles:
        if abs(a) < min_abs:
            continue
        t = np.radians(a)
        b = np.round(-xx * np.sin(t) + yy * np.cos(t)).astype(np.int64)
        b -= b.min()
        n = np.bincount(b)
        s = np.bincount(b, weights=v)
        ok = n >= 15
        if ok.sum() < 5:
            continue
        mu = s[ok] / n[ok]
        score = float(np.average(mu ** 2, weights=n[ok]))
        if score > best:
            best, best_a = score, a
    if not np.isfinite(best_a):
        return np.nan, np.nan
    xc = np.clip((lo + hi) / 2, deep["lo"], deep["hi"])
    pa = abs(((best_a - _slope_deg(deep["fit"]["c"], xc)) + 90) % 180 - 90)
    return float(best_a), float(pa)
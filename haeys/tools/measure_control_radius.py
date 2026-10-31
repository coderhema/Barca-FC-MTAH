"""Where CONTROL_RADIUS comes from: ball-to-carrier distance in the ATD during clear carries.

A carry is a frame where, over +-0.5 s, the same player stays nearest to the ball, moves at least
1.5 m, and the ball's displacement matches theirs within 1 m. Uses the tracking data only, no labels.

    python tools/measure_control_radius.py
"""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")

from possession import data  # noqa: E402

atd = data.load_atd(os.path.join(ROOT, "out", "cache"))
P = atd["players"][..., :2].astype(np.float64)
B = atd["ball"][:, :2].astype(np.float64)
n = len(B)
graded = np.zeros(n, bool)
for s, e in data.HALVES:
    graded[s:e] = True

d = np.linalg.norm(P - B[:, None], axis=2)
d = np.where(np.isnan(d), np.inf, d)
near = d.argmin(1)
dmin = d.min(1)
ok = graded & ~np.isnan(B[:, 0]) & np.isfinite(dmin)

W = 12  # +-0.5 s
idx = np.flatnonzero(ok)
idx = idx[(idx >= W) & (idx < n - W)]
carry = []
for t in idx:
    a, b = t - W, t + W
    if not (ok[a] and ok[b]): continue
    s = near[t]
    if near[a] != s or near[b] != s: continue
    dp = P[b, s] - P[a, s]
    db = B[b] - B[a]
    if np.any(np.isnan(dp)): continue
    if np.linalg.norm(dp) < 1.5: continue            # player actually moving (>= 1.5 m/s)
    if np.linalg.norm(db - dp) > 1.0: continue       # ball moves with them
    carry.append(t)
carry = np.array(carry)
print(f"carry frames: {len(carry)} ({len(carry)/ok.sum():.1%} of graded frames with a ball)")

dist = dmin[carry]
print("ball-to-carrier distance (m): " + "  ".join(f"p{q} {np.percentile(dist, q):.2f}" for q in (25, 50, 75, 90, 95)))
print(f"share within 1.5 m: {np.mean(dist <= 1.5):.1%}   within 2.5 m: {np.mean(dist <= 2.5):.1%}   within 3 m: {np.mean(dist <= 3):.1%}")

off = B[carry] - P[carry, near[carry]]
print(f"mean offset ball - carrier: dx {off[:,0].mean():+.2f} m  dy {off[:,1].mean():+.2f} m   (median dx {np.median(off[:,0]):+.2f}, dy {np.median(off[:,1]):+.2f})")
# offset along the carrier's direction of travel: positive = ball ahead of the player (lag would make it negative)
v = P[carry + W, near[carry]] - P[carry - W, near[carry]]
u = v / np.linalg.norm(v, axis=1, keepdims=True)
along = (off * u).sum(1)
across = off[:, 0] * -u[:, 1] + off[:, 1] * u[:, 0]
print(f"along direction of travel: median {np.median(along):+.2f} m   across: median {np.median(across):+.2f} m  |across| median {np.median(np.abs(across)):.2f}")

# time lag: ball(t) vs carrier(t+L)
best = []
for L in range(-15, 16, 3):
    pp = P[np.clip(carry + L, 0, n - 1), near[carry]]
    best.append((L, np.nanmedian(np.linalg.norm(B[carry] - pp, axis=1))))
print("median distance ball(t) to carrier(t+L): " + "  ".join(f"L{L:+d}:{v:.2f}" for L, v in best))

# by half and by pitch side (camera side y=0?)
for h, (s, e) in enumerate(data.HALVES, 1):
    m = (carry >= s) & (carry < e)
    print(f"half {h}: median dist {np.median(dist[m]):.2f}  mean dy {off[m,1].mean():+.2f}")
ys = B[carry, 1]
for lo, hi in ((0, 23), (23, 45), (45, 68.1)):
    m = (ys >= lo) & (ys < hi)
    print(f"ball y in [{lo},{hi}): n {m.sum()}  median dist {np.median(dist[m]):.2f}  mean dy {off[m,1].mean():+.2f}")

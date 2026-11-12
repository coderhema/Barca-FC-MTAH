"""CONTROL_RADIUS from EXTERNAL open data only: SkillCorner open data (broadcast tracking, 10 fps).

https://github.com/SkillCorner/opendata -- data/matches/<id>/<id>_tracking_extrapolated.jsonl
(Git LFS; download via media.githubusercontent.com). Coordinates are metres, pitch-centred.
No competition data (BAR-ATH tracking, tagging, video or ground truth) is read.

Two estimates:
  * carry heuristic (same definition as before): over +-0.5 s the same player stays nearest the
    ball, moves >= 1.5 m, and the ball's displacement matches theirs within 1 m;
  * SkillCorner's own per-frame possession label: distance from the ball to the labelled player.

    python haeys/tools/measure_control_radius_skillcorner.py FILE.jsonl [FILE.jsonl ...]
"""
import json
import sys

import numpy as np

W = 5  # +-0.5 s at 10 fps


def load(path):
    frames = [json.loads(line) for line in open(path)]
    ids = sorted({p["player_id"] for f in frames for p in f["player_data"]})
    col = {pid: i for i, pid in enumerate(ids)}
    n = len(frames)
    P = np.full((n, len(ids), 2), np.nan)
    B = np.full((n, 2), np.nan)
    owner = np.full(n, -1)
    for t, f in enumerate(frames):
        b = f["ball_data"]
        if b.get("is_detected") and b["x"] is not None:
            B[t] = b["x"], b["y"]
        for p in f["player_data"]:
            if p.get("is_detected", True):
                P[t, col[p["player_id"]]] = p["x"], p["y"]
        pid = (f.get("possession") or {}).get("player_id")
        if pid in col:
            owner[t] = col[pid]
    return P, B, owner


def measure(P, B, owner):
    n = len(B)
    d = np.linalg.norm(P - B[:, None], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    near, dmin = d.argmin(1), d.min(1)
    ok = ~np.isnan(B[:, 0]) & np.isfinite(dmin)
    carry = []
    for t in np.flatnonzero(ok):
        a, b = t - W, t + W
        if a < 0 or b >= n or not (ok[a] and ok[b]):
            continue
        s = near[t]
        if near[a] != s or near[b] != s:
            continue
        dp, db = P[b, s] - P[a, s], B[b] - B[a]
        if np.any(np.isnan(dp)) or np.linalg.norm(dp) < 1.5 or np.linalg.norm(db - dp) > 1.0:
            continue
        carry.append(dmin[t])
    lab = np.flatnonzero(ok & (owner >= 0))
    lab_d = d[lab, owner[lab]]
    return np.array(carry), lab_d[np.isfinite(lab_d)]


def show(name, x):
    print("{:<22} n {:>6}  ".format(name, len(x)) + "  ".join(
        "p{} {:.2f}".format(q, np.percentile(x, q)) for q in (50, 75, 90, 95)) +
        "   <=1.5 m {:.0%}  <=2.0 m {:.0%}  <=2.5 m {:.0%}".format(
            np.mean(x <= 1.5), np.mean(x <= 2.0), np.mean(x <= 2.5)))


if __name__ == "__main__":
    carries, labelled = [], []
    for path in sys.argv[1:]:
        c, l = measure(*load(path))
        show(path.split("/")[-1][:20] + " carry", c)
        show(path.split("/")[-1][:20] + " labelled", l)
        carries.append(c); labelled.append(l)
    show("ALL carry", np.concatenate(carries))
    show("ALL labelled", np.concatenate(labelled))

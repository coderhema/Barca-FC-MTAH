"""Cavalry-friendly diagnostics export (same schema as baseline.py --diagnostics).

Coordinates are normalised 0-1 pitch coords (matching the ATD export and baseline viz),
even though haeys works internally in metres. Distances stay in metres.
"""

import csv
import os

import numpy as np

from .data import AWAY, FPS, HOME, PITCH


def _norm_xy(xy):
    """Metres -> normalised pitch coords; NaN stays NaN."""
    out = np.asarray(xy, dtype=np.float64).copy()
    out[..., 0] = out[..., 0] / PITCH[0]
    out[..., 1] = out[..., 1] / PITCH[1]
    return out


def nearest_details(players, teams, ball_xy_m):
    """Nearest slot/distance/position per team, distances in m, positions in metres."""
    n = len(ball_xy_m)
    home_mask = teams == HOME
    away_mask = teams == AWAY
    dx = players[..., 0] - ball_xy_m[:, None, 0]
    dy = players[..., 1] - ball_xy_m[:, None, 1]
    d = np.hypot(dx, dy)
    d = np.where(np.isnan(d), np.inf, d)

    def pack(mask):
        dd = np.where(mask[None, :], d, np.inf)
        slot = np.argmin(dd, axis=1)
        dist = dd[np.arange(n), slot]
        valid = np.isfinite(dist)
        slot = np.where(valid, slot, -1)
        dist = np.where(valid, dist, np.nan)
        x = np.where(valid, players[np.arange(n), np.maximum(slot, 0), 0], np.nan)
        y = np.where(valid, players[np.arange(n), np.maximum(slot, 0), 1], np.nan)
        return slot, dist, x, y

    return pack(home_mask) + pack(away_mask)


def control_codes(control):
    """Map continuous (n,2) control scores to baseline-style -1/0/1 codes."""
    h, a = control[:, HOME], control[:, AWAY]
    codes = np.full(len(control), -1, np.int8)
    both = (h > 0) | (a > 0)
    codes[both & (h >= a)] = HOME
    codes[both & (a > h)] = AWAY
    return codes


def nearest_codes(dist):
    """-1/0/1 nearest-team codes from (n,2) distances (inf = missing)."""
    h, a = dist[:, HOME], dist[:, AWAY]
    codes = np.full(len(dist), -1, np.int8)
    known = np.isfinite(h) | np.isfinite(a)
    codes[known & (h <= a)] = HOME
    codes[known & (a < h)] = AWAY
    return codes


def write_diagnostics(folder, atd, signals, labels):
    """Write viz/frames.csv and viz/players.csv for Cavalry (or any timeline tool)."""
    import warnings

    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    players = atd["players"]
    teams = atd["teams"]
    raw_ball = atd["ball"][:, :2].astype(np.float64)  # metres, NaN where missing
    # Prefer raw (seen) ball for display; fall back is already NaN.
    ball_m = raw_ball
    seen = signals["seen"]
    control = signals["control"]
    dead_score = signals["dead"]
    dist = signals["dist"]

    details = nearest_details(players, teams, np.where(seen[:, None], ball_m, np.nan))
    (home_slot, home_d, home_x, home_y,
     away_slot, away_d, away_x, away_y) = details

    # Normalised coords for Cavalry / baseline schema compatibility.
    bx = np.where(seen, ball_m[:, 0] / PITCH[0], np.nan)
    by = np.where(seen, ball_m[:, 1] / PITCH[1], np.nan)
    home_xn = home_x / PITCH[0]
    home_yn = home_y / PITCH[1]
    away_xn = away_x / PITCH[0]
    away_yn = away_y / PITCH[1]

    ctrl = control_codes(control)
    nearest = nearest_codes(dist)
    # dead flag for viz: positive dead evidence or final label D
    dead_flag = (dead_score > 0) | (labels == "D")

    frames_path = os.path.join(folder, "frames.csv")
    with open(frames_path, "w", newline="\n") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "frame", "time_s", "ball_x", "ball_y", "ball_tracked",
            "nearest_home_slot", "nearest_home_distance_m", "nearest_home_x", "nearest_home_y",
            "nearest_away_slot", "nearest_away_distance_m", "nearest_away_x", "nearest_away_y",
            "control_team", "nearest_team", "dead_ball", "label",
            # haeys extras (Cavalry can ignore unknown columns)
            "score_H", "score_A", "score_D",
        ])
        n = len(labels)
        # scores not in signals -- caller can pass via signals if present
        score_h = signals.get("score_H")
        score_a = signals.get("score_A")
        score_d = signals.get("score_D", dead_score)
        if score_h is None:
            score_h = control[:, HOME]
            score_a = control[:, AWAY]
        for i in range(n):
            writer.writerow([
                i, "{:.3f}".format(i / FPS),
                bx[i], by[i], bool(seen[i]),
                int(home_slot[i]), home_d[i], home_xn[i], home_yn[i],
                int(away_slot[i]), away_d[i], away_xn[i], away_yn[i],
                int(ctrl[i]), int(nearest[i]), bool(dead_flag[i]), labels[i],
                float(score_h[i]), float(score_a[i]), float(score_d[i]),
            ])
    print("wrote", frames_path)

    players_path = os.path.join(folder, "players.csv")
    px = players[..., 0] / PITCH[0]
    py = players[..., 1] / PITCH[1]
    ps = players[..., 2]
    with open(players_path, "w", newline="\n") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "time_s", "slot", "team", "x", "y", "speed_m_s"])
        n_frames, n_slots = px.shape
        team_letter = np.where(teams == HOME, "H", "A")
        for frame in range(n_frames):
            slots = np.flatnonzero(~np.isnan(px[frame]) & ~np.isnan(py[frame]))
            t = "{:.3f}".format(frame / FPS)
            for slot in slots:
                writer.writerow([
                    frame, t, int(slot), team_letter[slot],
                    px[frame, slot], py[frame, slot], ps[frame, slot],
                ])
    print("wrote", players_path)
    return frames_path, players_path

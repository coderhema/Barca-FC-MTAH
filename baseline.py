"""possession_v1 -- frame-level ball possession for More than a Hack 2027 (BAR-ATH).

Regenerate the submission (data unpacked into ./data, or MTAH_DATA_DIR set):

    python baseline.py                        # writes <TEAM_NAME>.csv
    python baseline.py --out other_name.csv   # or choose the output path
    python baseline.py --diagnostics viz      # also writes Cavalry-friendly CSVs to viz/

Method (all constants are football priors, fixed without reference to any ground truth):

  1. CONTROL. On frames where the ball is tracked, a team is "in control" when its nearest player
     is within CONTROL_RADIUS_M of the ball AND clearly closer than the nearest opponent.
  2. HYSTERESIS. Possession only changes hands when the other team keeps control for
     CONFIRM_FRAMES in a row (a touch / deflection is not a change -- CHALLENGE.md). Between
     confirmations the last team keeps the ball (loose / contested ball stays with the last team).
  3. DEAD BALL. D when: the ball is tracked outside the pitch; or the game is "frozen" (ball not
     tracked or not moving AND players slow) for at least DEAD_MIN_S; or before the 2nd-half
     kickoff (frame 113742, published in FAQ.md). Smart Tagging SET PIECES give extra evidence:
     the frames just before a tagged restart are treated as dead if the game was slow there, and
     the tagged team (when not N/A) is used to seed who restarts.
  4. CLEAN-UP. Possessions shorter than MIN_SPELL_S are merged into their neighbours, because the
     ranking metric (spellF1) punishes invented short possessions hard.
  Ball missing inside play -> keep the last team (same as X for spellF1, better for macroF1).
  Ungraded frames (before kickoff, half-time, after the whistle) -> X.
"""

import argparse
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "starter"))
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402

# Team name used for the default output file name, <TEAM_NAME>.csv.
TEAM_NAME = "Vector"

# ---------------- parameters (football priors, NOT tuned on ground truth) ----------------
FPS = 25
CONTROL_RADIUS_M = 2.0      # a player this close to the ball can be playing it
MARGIN_M = 1.0              # ...and must be this much closer than the nearest opponent
CONFIRM_FRAMES = 12         # ~0.5 s of continuous control to confirm a change of possession
OUT_MARGIN = 0.005          # ball this far outside the 0-1 pitch box counts as out (~0.5 m)
SLOW_PLAYER_MS = 1.0        # median player speed below this = game looks stopped
BALL_STILL_MS = 1.0         # ball moving slower than this = not being played
DEAD_MIN_S = 3.0            # a "frozen" stretch must last this long to be called dead
MIN_SPELL_S = 2.0           # possessions shorter than this are merged away
BALL_GAP_RESET_S = 1.0      # a longer missing-ball gap invalidates a pending change
SECOND_HALF_KICKOFF = 113742  # FAQ.md: graded from 110808, all D until this kickoff
TEAM_TAGS = {"barcelona": 0, "atletic club": 1, "athletic club": 1}

HOME, AWAY, NONE = 0, 1, -1


def track_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) < 2:
        raise ValueError("tracking metadata must contain two real teams")
    first, second = team_ids[:2]
    return np.array([
        HOME if p.get("teamId") == first else AWAY if p.get("teamId") == second else NONE
        for p in root.iter("Player")
    ], dtype=np.int8)


def read_atd(raw_path, n_frames, n_slots):
    """Arrays (n_frames, n_slots) of x, y (0-1) and speed; ball x, y. NaN where missing."""
    px = np.full((n_frames, n_slots), np.nan, np.float32)
    py = np.full((n_frames, n_slots), np.nan, np.float32)
    ps = np.full((n_frames, n_slots), np.nan, np.float32)
    bx = np.full(n_frames, np.nan, np.float32)
    by = np.full(n_frames, np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1                      # ATD frame_id counts from 1
            if not 0 <= f < n_frames:
                continue
            for s, g in enumerate(parts[1].split(";")[:n_slots]):
                if g[:1] == "N" or not g:
                    continue
                v = g.split(",")
                px[f, s], py[f, s] = float(v[0]), float(v[1])
                if len(v) > 2 and v[2][:1] != "N":
                    ps[f, s] = float(v[2])
            b = parts[2].split(",")
            if b[0][:1] != "N" and b[0]:
                bx[f], by[f] = float(b[0]), float(b[1])
    return px, py, ps, bx, by


def load_set_pieces(n_frames):
    """[(start_frame, end_frame, team_code)] for SET PIECES tags; [] if the file is missing."""
    try:
        import load_smart_tagging as lst
        events = lst.load()
    except Exception as e:  # tagging is optional evidence
        print("  (no Smart Tagging used: {})".format(e))
        return []
    out = []
    for ev in events:
        if (ev["code"] or "").strip().lower() != "set pieces":
            continue
        team = TEAM_TAGS.get((ev["tags"].get("Team") or "").strip().lower(), NONE)
        out.append((max(0, ev["start_frame"]), min(n_frames, ev["end_frame"]), team))
    return out


def run_lengths(mask):
    """[(start, stop)] runs of True."""
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def control_per_frame(px, py, bx, by, teams, pitch):
    L, W = pitch
    dx = (px - bx[:, None]) * L
    dy = (py - by[:, None]) * W
    d = np.sqrt(dx * dx + dy * dy)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows = nobody tracked
        dh = np.nanmin(np.where(teams[None, :] == HOME, d, np.nan), axis=1)
        da = np.nanmin(np.where(teams[None, :] == AWAY, d, np.nan), axis=1)
    dh = np.where(np.isnan(dh), np.inf, dh)
    da = np.where(np.isnan(da), np.inf, da)
    ctrl = np.full(len(bx), NONE, np.int8)
    ctrl[(dh <= CONTROL_RADIUS_M) & (dh + MARGIN_M <= da)] = HOME
    ctrl[(da <= CONTROL_RADIUS_M) & (da + MARGIN_M <= dh)] = AWAY
    nearest = np.where(dh <= da, HOME, AWAY).astype(np.int8)
    nearest[np.isinf(dh) & np.isinf(da)] = NONE
    return ctrl, nearest


def nearest_details(px, py, bx, by, teams, pitch):
    """Return nearest slot, distance and position for each team on every frame."""
    L, W = pitch
    dx = (px - bx[:, None]) * L
    dy = (py - by[:, None]) * W
    d = np.sqrt(dx * dx + dy * dy)
    home_mask = teams == HOME
    away_mask = teams == AWAY
    home_d = np.where(home_mask[None, :] & np.isfinite(d), d, np.inf)
    away_d = np.where(away_mask[None, :] & np.isfinite(d), d, np.inf)
    home_slot = np.argmin(home_d, axis=1)
    away_slot = np.argmin(away_d, axis=1)
    home_distance = home_d[np.arange(len(d)), home_slot]
    away_distance = away_d[np.arange(len(d)), away_slot]
    home_valid = np.isfinite(home_distance)
    away_valid = np.isfinite(away_distance)
    home_slot = np.where(home_valid, home_slot, -1)
    away_slot = np.where(away_valid, away_slot, -1)
    home_distance = np.where(home_valid, home_distance, np.nan)
    away_distance = np.where(away_valid, away_distance, np.nan)
    home_x = np.where(home_valid, px[np.arange(len(px)), np.maximum(home_slot, 0)], np.nan)
    home_y = np.where(home_valid, py[np.arange(len(py)), np.maximum(home_slot, 0)], np.nan)
    away_x = np.where(away_valid, px[np.arange(len(px)), np.maximum(away_slot, 0)], np.nan)
    away_y = np.where(away_valid, py[np.arange(len(py)), np.maximum(away_slot, 0)], np.nan)
    return (home_slot, home_distance, home_x, home_y,
            away_slot, away_distance, away_x, away_y)


def dead_mask(px, py, ps, bx, by, ctrl, set_pieces, n):
    L, W = GAMES[GAME]["pitch"]
    ball_seen = ~np.isnan(bx)
    out = ball_seen & ((bx < -OUT_MARGIN) | (bx > 1 + OUT_MARGIN) |
                       (by < -OUT_MARGIN) | (by > 1 + OUT_MARGIN))
    # ball speed from positions (m/s), over 5 frames to beat jitter
    k = 5
    bsp = np.full(n, np.nan, np.float32)
    bsp[k:] = np.hypot((bx[k:] - bx[:-k]) * L, (by[k:] - by[:-k]) * W) * FPS / k
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med_speed = np.nanmedian(ps, axis=1)
    slow = np.nan_to_num(med_speed, nan=0.0) < SLOW_PLAYER_MS
    ball_idle = ball_seen & (np.nan_to_num(bsp, nan=0.0) < BALL_STILL_MS)
    frozen = slow & ball_idle
    dead = out.copy()
    for a, b in run_lengths(frozen):
        if b - a >= DEAD_MIN_S * FPS:
            dead[a:b] = True
    # just before a tagged restart the game is stopped: if it was slow there, mark dead
    for s, e, _ in set_pieces:
        lo = max(0, s - int(2 * FPS))
        seg = slice(lo, s)
        dead[seg] |= slow[seg]
    # Once the ball has gone out, remain dead only until tracked play plausibly restarts.
    # A missing ball is not itself evidence that play is frozen.
    plausible_restart = ball_seen & ~out & (
        (np.nan_to_num(bsp, nan=0.0) >= BALL_STILL_MS) |
        (np.nan_to_num(med_speed, nan=0.0) >= SLOW_PLAYER_MS) |
        (ctrl != NONE)
    )
    for a, b in run_lengths(out):
        j = b
        while j < n and slow[j] and not plausible_restart[j]:
            j += 1
        dead[a:j] = True
    return dead


def label_half(start, stop, ctrl, nearest, dead, sp_team, ball_seen):
    """Hysteresis state machine over one graded half. Returns codes 0/1/2 (H/A/D)."""
    out = np.full(stop - start, 2, np.int8)
    current, cand, streak, candidate_start = NONE, NONE, 0, NONE
    pending_start = NONE
    missing_ball = 0
    max_gap = int(BALL_GAP_RESET_S * FPS)
    for i in range(start, stop):
        if dead[i]:
            out[i - start] = 2
            current, cand, streak, candidate_start = NONE, NONE, 0, NONE
            pending_start = NONE
            missing_ball = 0
            continue
        if not ball_seen[i]:
            missing_ball += 1
            if missing_ball >= max_gap:
                cand, streak, candidate_start = NONE, 0, NONE
        else:
            missing_ball = 0
        if current == NONE:  # restart: seed from tag, else first control, else nearest
            seed = sp_team[i] if sp_team[i] != NONE else (ctrl[i] if ctrl[i] != NONE else nearest[i])
            if seed != NONE:
                current = seed
                if pending_start != NONE:
                    out[pending_start - start:i + 1 - start] = current
                    pending_start = NONE
            elif pending_start == NONE:
                pending_start = i
        c = ctrl[i]
        if c != NONE and c != current:
            streak = streak + 1 if c == cand else 1
            cand = c
            if streak == 1:
                candidate_start = i
            if streak >= CONFIRM_FRAMES:
                # change confirmed: back-date it to when the new team first took control
                out[candidate_start - start:i - start] = c
                current, cand, streak, candidate_start = c, NONE, 0, NONE
        elif c == current:
            cand, streak, candidate_start = NONE, 0, NONE
        out[i - start] = current if current != NONE else 2
    return out


def merge_short(codes, min_len):
    """Absorb H/A runs shorter than min_len into the previous possession (or the next one)."""
    e = np.flatnonzero(np.diff(codes)) + 1
    runs = [[int(codes[a]), int(a), int(b)] for a, b in
            zip(np.concatenate(([0], e)), np.concatenate((e, [len(codes)])))]
    merged = []
    for k, (code, a, b) in enumerate(runs):
        if code != 2 and b - a < min_len:
            prev = merged[-1][0] if merged else 2
            nxt = runs[k + 1][0] if k + 1 < len(runs) else 2
            fill = prev if prev != 2 else nxt
            if fill != 2:
                code = fill
        if merged and merged[-1][0] == code:
            merged[-1][2] = b
        else:
            merged.append([code, a, b])
    out = codes.copy()
    for code, a, b in merged:
        out[a:b] = code
    return out


def write_diagnostics(path, bx, by, dead, ctrl, nearest, labels, details):
    """Write compact per-frame evidence for external visualization tools."""
    import csv

    folder = os.path.abspath(path)
    os.makedirs(folder, exist_ok=True)
    (home_slot, home_distance, home_x, home_y,
     away_slot, away_distance, away_x, away_y) = details
    frames_path = os.path.join(folder, "frames.csv")
    players_path = os.path.join(folder, "players.csv")
    with open(frames_path, "w", newline="\n") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "frame", "time_s", "ball_x", "ball_y", "ball_tracked",
            "nearest_home_slot", "nearest_home_distance_m", "nearest_home_x", "nearest_home_y",
            "nearest_away_slot", "nearest_away_distance_m", "nearest_away_x", "nearest_away_y",
            "control_team", "nearest_team", "dead_ball", "label",
        ])
        for i in range(len(labels)):
            writer.writerow([
                i, "{:.3f}".format(i / FPS),
                bx[i], by[i], not np.isnan(bx[i]),
                home_slot[i], home_distance[i], home_x[i], home_y[i],
                away_slot[i], away_distance[i], away_x[i], away_y[i],
                ctrl[i], nearest[i], bool(dead[i]), labels[i],
            ])
    print("wrote", frames_path)
    return frames_path, players_path


def write_player_diagnostics(path, px, py, ps, teams):
    """Write tracked player positions in long form for Cavalry-style animation."""
    import csv

    players_path = os.path.join(os.path.abspath(path), "players.csv")
    with open(players_path, "w", newline="\n") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "time_s", "slot", "team", "x", "y", "speed_m_s"])
        for frame in range(px.shape[0]):
            slots = np.flatnonzero(~np.isnan(px[frame]) & ~np.isnan(py[frame]))
            for slot in slots:
                if teams[slot] == NONE:
                    continue
                writer.writerow([
                    frame, "{:.3f}".format(frame / FPS), int(slot),
                    "H" if teams[slot] == HOME else "A",
                    px[frame, slot], py[frame, slot], ps[frame, slot],
                ])
    print("wrote", players_path)
    return players_path


def main():
    parser = argparse.ArgumentParser(description="Write the possession submission CSV.")
    parser.add_argument("--out", default=None,
                        help="output CSV path (default: <TEAM_NAME>.csv next to this script)")
    parser.add_argument("--diagnostics", default=None, metavar="DIR",
                        help="also export frames.csv and players.csv for visualization")
    args = parser.parse_args()
    n = source_frames("tactical", GAME)
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    teams = track_teams(meta_path)
    print("reading tracking ({} slots) ...".format(len(teams)))
    px, py, ps, bx, by = read_atd(raw_path, n, len(teams))
    print("  ball tracked on {:.1f}% of frames".format(100 * np.mean(~np.isnan(bx))))

    set_pieces = load_set_pieces(n)
    sp_team = np.full(n, NONE, np.int8)
    for s, e, t in set_pieces:
        sp_team[s:e] = t

    ctrl, nearest = control_per_frame(px, py, bx, by, teams, GAMES[GAME]["pitch"])
    details = nearest_details(px, py, bx, by, teams, GAMES[GAME]["pitch"])
    dead = dead_mask(px, py, ps, bx, by, ctrl, set_pieces, n)

    labels = np.full(n, "X", dtype="<U1")
    letters = np.array(["H", "A", "D"])
    for half, first, last in half_video_blocks(GAME):
        if half == 2:
            dead[first:min(last + 1, SECOND_HALF_KICKOFF)] = True
        codes = label_half(
            first, last + 1, ctrl, nearest, dead, sp_team, ~np.isnan(bx)
        )
        codes = merge_short(codes, int(MIN_SPELL_S * FPS))
        labels[first:last + 1] = letters[codes]

    path = args.out or os.path.join(HERE, "{}.csv".format(TEAM_NAME))
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    vals, cnts = np.unique(labels, return_counts=True)
    print("wrote", path, dict(zip(vals.tolist(), cnts.tolist())))
    if args.diagnostics:
        write_diagnostics(args.diagnostics, bx, by, dead, ctrl, nearest, labels, details)
        write_player_diagnostics(args.diagnostics, px, py, ps, teams)
    print("next: python evaluation/validate_submission.py", path)


if __name__ == "__main__":
    main()

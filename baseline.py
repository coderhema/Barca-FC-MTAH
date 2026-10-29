"""possession_v1 -- frame-level ball possession for More than a Hack 2027 (BAR-ATH).

Regenerate the submission (data unpacked into ./data, or MTAH_DATA_DIR set):

    python baseline.py                        # writes <TEAM_NAME>.csv
    python baseline.py --out other_name.csv   # or choose the output path

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
# TODO: replace with the team name every member registered with.
TEAM_NAME = "Vector"

# ---------------- parameters (football priors, NOT tuned on ground truth) ----------------
FPS = 25
CONTROL_RADIUS_M = 2.0      # a player this close to the ball can be playing it
MARGIN_M = 1.0              # ...and must be this much closer than the nearest opponent
CONFIRM_FRAMES = 12         # ~0.5 s of continuous control to confirm a change of possession
OUT_MARGIN = 0.005          # ball this far outside the 0-1 pitch box counts as out (~0.5 m)
SLOW_PLAYER_MS = 2.0        # median player speed below this = game looks stopped
BALL_STILL_MS = 1.0         # ball moving slower than this = not being played
DEAD_MIN_S = 3.0            # a "frozen" stretch must last this long to be called dead
MIN_SPELL_S = 2.0           # possessions shorter than this are merged away
SECOND_HALF_KICKOFF = 113742  # FAQ.md: graded from 110808, all D until this kickoff
TEAM_TAGS = {"barcelona": 0, "atletic club": 1, "athletic club": 1}

HOME, AWAY, NONE = 0, 1, -1


def track_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


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


def dead_mask(px, py, ps, bx, by, set_pieces, n):
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
    ball_idle = ~ball_seen | (np.nan_to_num(bsp, nan=0.0) < BALL_STILL_MS)
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
    # once the ball has gone out, the game stays dead while the players are still slow
    for a, b in run_lengths(out):
        j = b
        while j < n and slow[j]:
            j += 1
        dead[a:j] = True
    return dead


def label_half(start, stop, ctrl, nearest, dead, sp_team):
    """Hysteresis state machine over one graded half. Returns codes 0/1/2 (H/A/D)."""
    out = np.full(stop - start, 2, np.int8)
    current, cand, streak = NONE, NONE, 0
    for i in range(start, stop):
        if dead[i]:
            out[i - start] = 2
            current, cand, streak = NONE, NONE, 0   # new possession starts after the restart
            continue
        if current == NONE:  # restart: seed from tag, else first control, else nearest
            seed = sp_team[i] if sp_team[i] != NONE else (ctrl[i] if ctrl[i] != NONE else nearest[i])
            if seed != NONE:
                current = seed
        c = ctrl[i]
        if c != NONE and c != current:
            streak = streak + 1 if c == cand else 1
            cand = c
            if streak >= CONFIRM_FRAMES:
                # change confirmed: back-date it to when the new team first took control
                j0 = i - start - streak + 1
                out[j0:i - start] = c
                current, cand, streak = c, NONE, 0
        elif c == current:
            cand, streak = NONE, 0
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


def main():
    parser = argparse.ArgumentParser(description="Write the possession submission CSV.")
    parser.add_argument("--out", default=None,
                        help="output CSV path (default: <TEAM_NAME>.csv next to this script)")
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
    dead = dead_mask(px, py, ps, bx, by, set_pieces, n)

    labels = np.full(n, "X", dtype="<U1")
    letters = np.array(["H", "A", "D"])
    for half, first, last in half_video_blocks(GAME):
        codes = label_half(first, last + 1, ctrl, nearest, dead, sp_team)
        if half == 2:
            codes[:max(0, SECOND_HALF_KICKOFF - first)] = 2
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
    print("next: python evaluation/validate_submission.py", path)


if __name__ == "__main__":
    main()

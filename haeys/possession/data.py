"""Load the tactical ATD, its metadata and the Smart Tagging into numpy arrays.

Everything is on the tactical video timeline: index i is video frame i (ATD frame_id - 1).
Coordinates are converted from the normalised 0-1 pitch to metres.

The raw ATD is ~200k lines, so the parsed arrays are cached as an .npz next to the outputs.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

N_FRAMES = 199319
FPS = 25
PITCH = (105.3, 68.0)  # metres, as measured by the organisers (games.py)
STEM = "BAR-ATH_tactical"

# Graded halves in video frames, half-open. Published in DATA.md; only used to write X outside
# them, never as labels.
HALVES = ((11185, 90652), (110808, 190198))

# Kickoff of each half in video frames. Published in FAQ.md: the first half is graded from its
# kickoff, the second is graded from 110808 but kicks off at 113742, and the frames between are D.
KICKOFFS = (11185, 113742)

HOME, AWAY = 0, 1


def data_dir():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default = os.path.join(os.path.dirname(here), "data")   # the repository's ./data, like games.py
    return os.environ.get("MTAH_DATA_DIR") or default


def path(suffix):
    full = os.path.join(data_dir(), "tactical", STEM + suffix)
    if not os.path.exists(full):
        raise FileNotFoundError("{} not found -- set MTAH_DATA_DIR to the unpacked bundle".format(full))
    return full


def track_teams(meta_path):
    """Team code per track slot. The first Team in the EPTS metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams, found {}".format(len(team_ids)))
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def team_names(meta_path):
    root = ET.parse(meta_path).getroot()
    return [team.findtext("Name") for team in root.iter("Team")]


def _parse_raw(raw_path, n_slots, n_frames):
    """players (n_frames, n_slots, 3) and ball (n_frames, k) arrays of x, y, speed (+ extras).

    Rows the file never mentions stay NaN. The ball group's width is read off the file rather
    than assumed, because extra ball channels (z, state, owner) would be the most valuable
    thing in the export and must not be silently dropped.
    """
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = None
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) < 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                if group and group[0] != "N":
                    players[frame, slot] = [float(v) for v in group.split(",")[:3]]
            values = parts[2].split(",")
            if ball is None:
                ball = np.full((n_frames, len(values)), np.nan, dtype=np.float32)
            ball[frame, :len(values)] = [float(v) if v and v[0] != "N" else np.nan
                                         for v in values[:ball.shape[1]]]
    players[..., 0] *= PITCH[0]
    players[..., 1] *= PITCH[1]
    ball[:, 0] *= PITCH[0]
    ball[:, 1] *= PITCH[1]
    return players, ball


def load_atd(cache_dir, n_frames=N_FRAMES):
    """dict(players, ball, teams, names), cached."""
    meta_path, raw_path = path("_atd.xml"), path("_atd.txt")
    teams = track_teams(meta_path)
    cache = os.path.join(cache_dir, "atd_cache.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        stored = np.load(cache)
        players, ball = stored["players"], stored["ball"]
    else:
        print("parsing {} (one-off, cached afterwards) ...".format(os.path.basename(raw_path)))
        players, ball = _parse_raw(raw_path, len(teams), n_frames)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, players=players, ball=ball)
    return {"players": players, "ball": ball, "teams": teams, "names": team_names(meta_path)}


def load_tagging(fps=FPS):
    """[{code, start, end, tags}] with start/end as video frames (end exclusive)."""
    out = []
    for instance in ET.parse(path("_smart_tagging.xml")).getroot().iter("instance"):
        out.append({
            "code": (instance.findtext("code") or "").strip().lower(),
            "start": int(round(float(instance.findtext("start")) * fps)),
            "end": int(round(float(instance.findtext("end")) * fps)),
            "tags": {(lab.findtext("group") or "").strip().lower():
                     (lab.findtext("text") or "").strip().lower()
                     for lab in instance.findall("label")},
        })
    return out

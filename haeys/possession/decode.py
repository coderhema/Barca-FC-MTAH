"""Turn per-frame scores into possession spells with a three-state Viterbi.

Smoothing with a majority vote (the baseline) trades short possessions for fewer flickers with
one knob. A segmentation with switching costs does better on the metric that ranks: a change of
state is accepted only when the evidence summed since the change outweighs its cost, so a single
deflection cannot split a possession, but a short possession with sustained control survives.
"""

import numpy as np

from .data import FPS

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# Params -- costs in the same "frames of evidence" units as features.W_*.
SWITCH_TEAM = 12.0    # H <-> A: ~0.48 s of clean control — same prior as baseline.py
                        # CONFIRM_FRAMES=12; CHALLENGE.md: a touch without control is not a change
ENTER_DEAD = 30.0     # alive -> D: stoppages last seconds, a blip of calm is not one
LEAVE_DEAD = 10.0     # D -> alive: restarts are sharp (a kick), so cheaper to leave

MIN_SPELL_S = 2.0     # possessions shorter than this are merged into neighbours
                        # (spellF1 punishes invented short possessions hard)


def transition_costs():
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = SWITCH_TEAM
    cost[H, D] = cost[A, D] = ENTER_DEAD
    cost[D, H] = cost[D, A] = LEAVE_DEAD
    return cost


def viterbi(scores, cost=None, start=D):
    """Best state path maximising sum(score) - sum(transition cost). scores is (n, 3).

    start fixes the state before the first frame; None leaves it free (the evidence decides).
    """
    cost = transition_costs() if cost is None else cost
    n = len(scores)
    back = np.zeros((n, 3), dtype=np.int8)
    value = np.zeros(3) if start is None else np.full(3, -1e9)
    if start is not None:
        value[start] = 0.0
    value = value + scores[0]
    for t in range(1, n):
        candidates = value[:, None] - cost          # [from, to]
        back[t] = candidates.argmax(axis=0)
        value = candidates.max(axis=0) + scores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(value.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def merge_short(codes, min_len):
    """Absorb H/A runs shorter than min_len into the previous (or next) possession.

    codes is an int8 array with H=0/A=1/D=2. Only H/A runs are merged; D runs are left alone.
    Mirrors baseline.py's merge_short so short flickers don't inflate spellF1.
    """
    e = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], e))
    stops = np.concatenate((e, [len(codes)]))
    runs = [(int(codes[a]), int(a), int(b)) for a, b in zip(starts, stops)]
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


def decode(scores, halves, n_frames, kickoffs=None, min_spell_s=MIN_SPELL_S):
    """Labels for the whole video: X outside the graded halves, Viterbi inside each.

    With kickoffs, graded frames before a half's kickoff are D and decoding starts at the kickoff
    with a free initial state: the kickoff frame is already live play.
    Short H/A possessions (below min_spell_s) are merged into neighbours afterwards, per half.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    min_len = int(min_spell_s * FPS)
    for i, (start, stop) in enumerate(halves):
        kick = start if kickoffs is None else min(max(start, kickoffs[i]), stop)
        codes = np.full(stop - start, D, np.int8)
        first = D if kickoffs is None else None
        codes[kick - start:stop - start] = viterbi(scores[kick:stop], start=first)
        codes = merge_short(codes, min_len)
        labels[start:stop] = LABELS[codes]
    return labels

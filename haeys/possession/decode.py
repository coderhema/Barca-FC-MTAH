"""Turn per-frame scores into possession spells with a three-state Viterbi.

Smoothing with a majority vote (the baseline) trades short possessions for fewer flickers with
one knob. A segmentation with switching costs does better on the metric that ranks: a change of
state is accepted only when the evidence summed since the change outweighs its cost, so a single
deflection cannot split a possession, but a short possession with sustained control survives.
"""

import numpy as np

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# Params -- costs in the same "frames of evidence" units as features.W_*.
SWITCH_TEAM = 8.0     # H <-> A: about a third of a second of clean control by the new team
ENTER_DEAD = 30.0     # alive -> D: stoppages last seconds, a blip of calm is not one
LEAVE_DEAD = 10.0     # D -> alive: restarts are sharp (a kick), so cheaper to leave


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


def decode(scores, halves, n_frames, kickoffs=None):
    """Labels for the whole video: X outside the graded halves, Viterbi inside each.

    With kickoffs, graded frames before a half's kickoff are D and decoding starts at the kickoff
    with a free initial state: the kickoff frame is already live play.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    for i, (start, stop) in enumerate(halves):
        kick = start if kickoffs is None else min(max(start, kickoffs[i]), stop)
        labels[start:kick] = "D"
        first = D if kickoffs is None else None
        labels[kick:stop] = LABELS[viterbi(scores[kick:stop], start=first)]
    return labels

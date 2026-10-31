"""Per-frame evidence for the three states, from tracking geometry and Smart Tagging.

The decoder works on scores, not probabilities: for every frame a score for H, A and D, where
positive means "this frame looks like that state" and 0 means "no opinion". Most frames carry no
opinion at all -- the ball is in flight, or missing -- and the decoder's switching costs carry
the last state through them, which is exactly the "possession stays with the last team that had
it" rule in CHALLENGE.md.

Every constant here is a football prior (metres, m/s, seconds), chosen before looking at any
ground truth. None of them is fitted to the public five minutes.
"""

import warnings

import numpy as np

from .data import AWAY, FPS, HOME, PITCH

# Params
MAX_BALL_GAP = 2 * FPS      # interpolate the ball across gaps up to this many frames
CONTROL_RADIUS = 2.5        # m; a player this close to the ball is treated as touching it.
                            # 95th percentile of the ball-to-carrier distance in the ATD while a
                            # player and the ball move together for 1 s (1.5 m covered only 84%)
CONTROL_MARGIN = 2.0        # m; full confidence once the other team is this much further away
INTERP_WEIGHT = 0.4         # control seen on an interpolated ball counts this much
OUT_MARGIN = 0.5            # m beyond the line before the ball counts as out
CALM_SPEED = 1.6            # m/s mean player speed at which play starts to look stopped
CALM_SCALE = 0.6            # m/s over which calmness ramps from 0 to 1
KICK_SPEED = 6.0            # m/s ball speed that marks a restart being taken
SMOOTH = FPS                # frames; window for the slow signals (player speed)

# Weights, in "frames of evidence": the decoder compares their sums against switching costs.
W_CONTROL = 1.0
W_OUT = 3.0
W_CALM = 1.0
W_BALL_FAST = 1.0
W_PRE_RESTART = 2.0
W_PRE_KICKOFF = 5.0
W_SHAPE = 0.15


def moving_mean(x, window):
    """Centred mean over +/- window frames, ignoring NaN. NaN where the window is empty."""
    valid = ~np.isnan(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(x))
    low = np.maximum(0, index - window)
    high = np.minimum(len(x), index + window + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def interpolate_ball(ball_xy, max_gap=MAX_BALL_GAP):
    """(xy, interpolated mask). Linear fill of gaps no longer than max_gap with both ends seen."""
    xy = ball_xy.copy()
    seen = ~np.isnan(xy[:, 0])
    filled = np.zeros(len(xy), dtype=bool)
    idx = np.flatnonzero(seen)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            t = np.arange(1, gap + 1) / float(gap + 1)
            xy[a + 1:b] = xy[a] + t[:, None] * (xy[b] - xy[a])
            filled[a + 1:b] = True
    return xy, filled


def ball_speed(ball_xy):
    """m/s between consecutive observed frames, lightly smoothed. NaN where unknown."""
    step = np.full(len(ball_xy), np.nan)
    step[1:] = np.hypot(*(ball_xy[1:] - ball_xy[:-1]).T) * FPS
    return moving_mean(step, 2)


def team_distances(players, teams, ball_xy):
    """(n, 2) nearest distance from each team to the ball, inf where nobody/no ball."""
    dx = players[..., 0] - ball_xy[:, None, 0]
    dy = players[..., 1] - ball_xy[:, None, 1]
    d = np.hypot(dx, dy)
    d = np.where(np.isnan(d), np.inf, d)
    return np.stack([d[:, teams == HOME].min(axis=1), d[:, teams == AWAY].min(axis=1)], axis=1)


def control_evidence(dist, weight):
    """(n, 2) per-team score: someone is on the ball and the other team is clearly not."""
    out = np.zeros_like(dist)
    for team in (HOME, AWAY):
        own, other = dist[:, team], dist[:, 1 - team]
        close = own <= CONTROL_RADIUS
        with np.errstate(invalid="ignore"):
            margin = np.clip((other - own) / CONTROL_MARGIN, 0.0, 1.0)
        out[:, team] = np.where(close, margin * weight, 0.0)
    return out


def restart_moment(speed, start, stop):
    """First frame in [start, stop) where the ball is kicked, or None."""
    window = speed[start:stop]
    hits = np.flatnonzero(window > KICK_SPEED)
    return start + int(hits[0]) if len(hits) else None


def tagging_dead_evidence(tags, speed, n):
    """Dead-ball score from set-piece tags: dead up to the kick that restarts play."""
    score = np.zeros(n)
    for tag in tags:
        if tag["code"] != "set pieces":
            continue
        start, stop = max(0, tag["start"]), min(n, tag["end"])
        kick = restart_moment(speed, start, stop)
        kickoff = tag["tags"].get("type", "").startswith("kick off (start)")
        weight = W_PRE_KICKOFF if kickoff else W_PRE_RESTART
        if kick is None:
            score[start:(start + stop) // 2] += 0.25 * weight
        else:
            score[start:kick] += weight
            score[kick:kick + FPS] -= W_BALL_FAST
    return score


def pre_kickoff_evidence(tags, speed, halves, n):
    """Graded frames of a half before its tagged kickoff is taken are dead."""
    score = np.zeros(n)
    kickoffs = [t for t in tags if t["code"] == "set pieces"
                and t["tags"].get("type", "").startswith("kick off (start)")]
    for start, stop in halves:
        inside = [t for t in kickoffs if start - 10 * FPS <= t["start"] < stop]
        if not inside:
            continue
        first = min(inside, key=lambda t: t["start"])
        kick = restart_moment(speed, max(start, first["start"]), min(stop, first["end"]))
        end = kick if kick is not None else first["start"]
        score[start:end] += W_PRE_KICKOFF
    return score


def shape_features(players, teams, halves, n):
    """Collective team-shape features, oriented so home always attacks towards +x."""
    x, y, s = players[..., 0], players[..., 1], players[..., 2]
    feats = []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames: no players tracked
        cols = {}
        for team, name in ((HOME, "h"), (AWAY, "a")):
            m = teams == team
            cols[name + "_cx"] = np.nanmean(x[:, m], axis=1)
            cols[name + "_sx"] = np.nanstd(x[:, m], axis=1)
            cols[name + "_sy"] = np.nanstd(y[:, m], axis=1)
            cols[name + "_v"] = np.nanmean(s[:, m], axis=1)
    flip = np.zeros(n, dtype=bool)
    for start, stop in halves:
        flip[start:stop] = np.nanmean(cols["h_cx"][start:stop]) > PITCH[0] / 2
    for name in ("h_cx", "a_cx"):
        cols[name] = np.where(flip, PITCH[0] - cols[name], cols[name])
    for name in ("h_cx", "a_cx"):
        smoothed = moving_mean(cols[name], SMOOTH)
        velocity = np.full(n, np.nan)
        velocity[SMOOTH:] = (smoothed[SMOOTH:] - smoothed[:-SMOOTH]) / (SMOOTH / float(FPS))
        cols[name + "_vel"] = velocity
    for key in sorted(cols):
        feats.append(moving_mean(cols[key], SMOOTH // 2))
    return np.stack(feats, axis=1)


def fit_logistic(X, y, steps=300, lr=0.5, l2=1e-3):
    """Plain gradient-descent logistic regression on standardised features. Deterministic."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w)
    return lambda Xn: 1.0 / (1.0 + np.exp(-np.hstack([(Xn - mu) / sd, np.ones((len(Xn), 1))]) @ w))


def shape_evidence(players, teams, halves, control, n):
    """Self-supervised H/A score for frames with no ball evidence.

    Pseudo-labels come from frames where the ball-control rule is confident -- produced by an
    algorithm, no human decided any of them (RULES.md) -- and a logistic model learns which team
    shape goes with which team on the ball. It only speaks where control is silent.
    """
    X = shape_features(players, teams, halves, n)
    usable = ~np.isnan(X).any(axis=1)
    graded = np.zeros(n, dtype=bool)
    for start, stop in halves:
        graded[start:stop] = True
    confident = usable & graded & (np.abs(control[:, HOME] - control[:, AWAY]) > 0.5)
    if confident.sum() < 1000:
        return np.zeros((n, 2))
    model = fit_logistic(X[confident], (control[confident, HOME] > control[confident, AWAY]) * 1.0)
    p_home = np.full(n, 0.5)
    p_home[usable] = model(X[usable])
    silent = (control.max(axis=1) == 0) & usable
    score = np.zeros((n, 2))
    score[silent, HOME] = W_SHAPE * (2 * p_home[silent] - 1)
    score[silent, AWAY] = -score[silent, HOME]
    return score


def build_scores(atd, tags, halves, use_shape=False):
    """(n, 3) scores for H, A, D, and a dict of the intermediate signals for inspection."""
    players, teams = atd["players"], atd["teams"]
    raw_ball = atd["ball"][:, :2].astype(np.float64)
    n = len(raw_ball)

    seen = ~np.isnan(raw_ball[:, 0])
    ball, interpolated = interpolate_ball(raw_ball)
    speed = ball_speed(raw_ball)
    dist = team_distances(players, teams, ball)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        player_speed = moving_mean(np.nanmean(players[..., 2], axis=1).astype(np.float64), SMOOTH)
    calm = np.clip((CALM_SPEED - np.nan_to_num(player_speed, nan=0.0)) / CALM_SCALE, -1.0, 1.0)

    # On the ball: full weight on a seen ball, partial on an interpolated one, and fading out when
    # play looks stopped -- a thrower holding the ball is not a possession yet, it is a dead ball.
    weight = np.where(seen, W_CONTROL, np.where(interpolated, W_CONTROL * INTERP_WEIGHT, 0.0))
    weight = weight * (1.0 - np.clip(calm, 0.0, 1.0))
    control = control_evidence(dist, weight)

    known = seen | interpolated
    out = known & ((ball[:, 0] < -OUT_MARGIN) | (ball[:, 0] > PITCH[0] + OUT_MARGIN) |
                   (ball[:, 1] < -OUT_MARGIN) | (ball[:, 1] > PITCH[1] + OUT_MARGIN))
    fast = np.nan_to_num(speed, nan=0.0) > 3.0

    dead = (W_OUT * out + W_CALM * calm - W_BALL_FAST * fast
            + tagging_dead_evidence(tags, speed, n)
            + pre_kickoff_evidence(tags, speed, halves, n))

    team = control
    if use_shape:
        team = team + shape_evidence(players, teams, halves, control, n)

    scores = np.column_stack([team[:, HOME], team[:, AWAY], dead])
    signals = {"seen": seen, "interpolated": interpolated, "dist": dist, "calm": calm,
               "out": out, "control": control, "dead": dead}
    return scores, signals

"""numpy-only re-implementation of macroF1 and spellF1, for environments without pandas.

The official scorer (repo/evaluation/run_evaluation.py) is the authority; this mirrors
possession_metrics.py closely enough to sanity-check synthetic runs.

    python tests/score.py truth.csv prediction.csv
"""

import sys

import numpy as np

IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def read(path):
    with open(path) as handle:
        next(handle)
        return np.array([line.rstrip().split(",")[1].strip().upper() for line in handle])


def runs(gt):
    graded = np.concatenate(([0], (gt != "X").astype(int), [0]))
    edges = np.diff(graded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def filled(pred, spans):
    out = pred.copy()
    for a, b in spans:
        w = out[a:b]
        known = np.flatnonzero(w != "X")
        if len(known):
            carry = np.maximum.accumulate(np.where(w != "X", np.arange(len(w)), -1))
            carry[carry < 0] = known[0]
            out[a:b] = w[carry]
    return out


def spells(codes, spans):
    out = []
    for a, b in spans:
        w = codes[a:b]
        bounds = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        out += [(a + s, a + e, w[s]) for s, e in zip(bounds[:-1], bounds[1:]) if w[s] in "HA"]
    return out


def spell_f1(gt, pred, spans):
    g, p = spells(gt, spans), spells(filled(pred, spans), spans)
    pairs = []
    for i, (gs, ge, gk) in enumerate(g):
        for j, (ps, pe, pk) in enumerate(p):
            if pk == gk and ps < ge and gs < pe:
                pairs.append(((min(ge, pe) - max(gs, ps)) / (max(ge, pe) - min(gs, ps)), i, j))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    f1s = []
    for threshold in IOUS:
        used_g, used_p = set(), set()
        for iou, i, j in pairs:
            if iou < threshold:
                break
            if i not in used_g and j not in used_p:
                used_g.add(i)
                used_p.add(j)
        tp = len(used_g)
        prec, rec = tp / max(len(p), 1), tp / max(len(g), 1)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)), len(g), len(p)


def macro_f1(gt, pred):
    graded = gt != "X"
    g, p = gt[graded], pred[graded]
    f1 = []
    for k in "HAD":
        tp = np.sum((g == k) & (p == k))
        denominator = np.sum(g == k) + np.sum(p == k)
        f1.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(f1))


if __name__ == "__main__":
    gt = read(sys.argv[1])
    spans = runs(gt)
    for path in sys.argv[2:]:
        pred = read(path)
        sf, n_gt, n_pred = spell_f1(gt, pred, spans)
        print("{:<40} macroF1 {:.4f}  spellF1 {:.4f}  poss {} (truth {})".format(
            path.split("/")[-1], macro_f1(gt, pred), sf, n_pred, n_gt))

"""Data bundle in, possession CSV out.

    python run.py                       # writes out/<team>.csv
    python run.py --shape               # also use the self-supervised team-shape model
    python run.py --inspect             # what the ATD export actually contains

Deterministic: the same data gives the same file, byte for byte.
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

from possession import data
from possession.decode import decode
from possession.features import build_scores

HERE = os.path.dirname(os.path.abspath(__file__))


def write_csv(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def check(labels, n_frames):
    """The structural checks validate_submission.py runs, so a bad file fails here first."""
    assert len(labels) == n_frames, "row count {} != {}".format(len(labels), n_frames)
    assert set(np.unique(labels)) <= set("HADX"), "bad labels"


def summary(labels):
    """Possession counts per half: the first thing to look at, since spellF1 punishes flicker."""
    for half, (start, stop) in enumerate(data.HALVES, 1):
        window = labels[start:stop]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        runs = np.split(window, breaks)
        spells = [len(r) for r in runs if r[0] in "HA"]
        counts = {k: int((window == k).sum()) for k in "HAD"}
        print("half {}: {} possessions, mean {:.1f} s, median {:.1f} s   H {H}  A {A}  D {D}".format(
            half, len(spells), np.mean(spells) / data.FPS, np.median(spells) / data.FPS, **counts))


def inspect():
    """Print the ATD data specification and the shape of a few raw lines."""
    root = ET.parse(data.path("_atd.xml")).getroot()
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        if any(key in tag for key in ("Channel", "Register", "DataSpecification")):
            print(tag, dict(element.attrib))
    with open(data.path("_atd.txt")) as handle:
        for _, line in zip(range(3), handle):
            parts = line.rstrip("\n").split(":")
            groups = parts[1].split(";") if len(parts) > 1 else []
            print("fields {}  slots {}  player group '{}'  ball '{}'".format(
                len(parts), len(groups), groups[0] if groups else "", parts[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="submission", help="names the output file")
    parser.add_argument("--out", default=os.path.join(HERE, "out"))
    parser.add_argument("--shape", action="store_true", help="add the team-shape model")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    if args.inspect:
        inspect()
        return

    atd = data.load_atd(os.path.join(args.out, "cache"))
    print("teams: H = {}, A = {}".format(*atd["names"]))
    tags = data.load_tagging()
    scores, signals = build_scores(atd, tags, data.HALVES, use_shape=args.shape)
    print("ball seen {:.1f}%, interpolated {:.1f}%".format(
        100 * signals["seen"].mean(), 100 * signals["interpolated"].mean()))

    labels = decode(scores, data.HALVES, len(scores), kickoffs=data.KICKOFFS)
    check(labels, data.N_FRAMES)
    summary(labels)
    target = os.path.join(args.out, args.team + ".csv")
    write_csv(target, labels)
    print("wrote", target)


if __name__ == "__main__":
    main()

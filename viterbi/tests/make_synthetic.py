"""Write a fake data bundle in the real file formats, plus its true labels.

Only for checking that the pipeline runs end to end and that its logic does what it claims
before the real data arrives. Scores on it mean nothing about the real match.

    python tests/make_synthetic.py /tmp/fake_bundle
    MTAH_DATA_DIR=/tmp/fake_bundle python run.py --team synthetic
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession.data import FPS, HALVES, N_FRAMES, PITCH, STEM  # noqa: E402

N_SLOTS = 22


def simulate(rng):
    labels = np.full(N_FRAMES, "X", dtype="<U1")
    players = np.full((N_FRAMES, N_SLOTS, 3), np.nan)
    ball = np.full((N_FRAMES, 2), np.nan)
    tags = []
    pos = np.column_stack([rng.uniform(10, 95, N_SLOTS), rng.uniform(5, 63, N_SLOTS)])

    for start, stop in HALVES:
        t, state = start, "D"
        bxy = np.array([PITCH[0] / 2, PITCH[1] / 2])
        while t < stop:
            if state == "D":
                length = int(rng.uniform(5, 20) * FPS)
                tags.append((t, t + length + 2 * FPS, rng.integers(2)))
                end = min(stop, t + length)
                for f in range(t, end):
                    pos += rng.normal(0, 0.03, pos.shape)
                    players[f, :, :2], players[f, :, 2] = pos, 0.8
                    ball[f] = bxy
                labels[t:end] = "D"
                state = "H" if rng.random() < 0.5 else "A"
                t = end
                continue
            length = int(rng.uniform(0.8, 40) * FPS)
            end = min(stop, t + length)
            team = 0 if state == "H" else 1
            carrier = team * 11 + rng.integers(11)
            for f in range(t, end):
                pos += rng.normal(0, 0.15, pos.shape)
                pos = np.clip(pos, [1, 1], [PITCH[0] - 1, PITCH[1] - 1])
                if rng.random() < 0.01:  # pass: ball travels for a few frames, nobody on it
                    carrier = team * 11 + rng.integers(11)
                target = pos[carrier] + [0.5, 0.3]
                step = target - bxy
                bxy = bxy + step * min(1.0, 3.0 / (np.linalg.norm(step) + 1e-9))
                players[f, :, :2], players[f, :, 2] = pos, rng.uniform(2, 5, N_SLOTS)
                ball[f] = bxy
            labels[t:end] = state
            t = end
            r = rng.random()
            if r < 0.25:
                state = "D"
                bxy = np.array([bxy[0], -2.0])  # out over the touchline
                ball[end - 3:end] = bxy
            else:
                state = "A" if state == "H" else "H"

    # Lose the ball in chunks, ~40% of frames, like the real export.
    f = 0
    while f < N_FRAMES:
        if rng.random() < 0.4:
            ball[f:f + int(rng.uniform(5, 80))] = np.nan
        f += int(rng.uniform(5, 80))
    return labels, players, ball, tags


def write(bundle, labels, players, ball, tags):
    folder = os.path.join(bundle, "tactical")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, STEM + "_atd.xml"), "w") as handle:
        handle.write('<Main><Teams><Team id="T1"><Name>Home</Name></Team>'
                     '<Team id="T2"><Name>Away</Name></Team></Teams><Players>')
        for slot in range(N_SLOTS):
            handle.write('<Player id="P{}" teamId="{}"/>'.format(slot, "T1" if slot < 11 else "T2"))
        handle.write("</Players></Main>")

    def group(x, y, s):
        return "NaN,NaN,NaN" if np.isnan(x) else "{:.5f},{:.5f},{:.2f}".format(x, y, s)

    with open(os.path.join(folder, STEM + "_atd.txt"), "w") as handle:
        for f in range(N_FRAMES):
            p = players[f]
            groups = ";".join(group(p[i, 0] / PITCH[0], p[i, 1] / PITCH[1], p[i, 2])
                              for i in range(N_SLOTS))
            b = ball[f]
            handle.write("{}:{}:{}\n".format(f + 1, groups,
                                             group(b[0] / PITCH[0], b[1] / PITCH[1], 0.0)))

    with open(os.path.join(folder, STEM + "_smart_tagging.xml"), "w") as handle:
        handle.write("<file><ALL_INSTANCES>")
        for i, (start, stop, team) in enumerate(tags):
            handle.write("<instance><ID>{}</ID><start>{}</start><end>{}</end><code>SET PIECES</code>"
                         "<label><group>Team</group><text>{}</text></label></instance>".format(
                             i, start / FPS, stop / FPS, "Home" if team == 0 else "Away"))
        handle.write("</ALL_INSTANCES></file>")

    with open(os.path.join(bundle, "truth.csv"), "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


if __name__ == "__main__":
    bundle = sys.argv[1]
    write(bundle, *simulate(np.random.default_rng(0)))
    print("wrote", bundle)

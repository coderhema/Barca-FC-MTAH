# Haeys — possession by evidence scoring + Viterbi decoding

_Haeys is pronounced "eyes"._

A second pipeline, next to `baseline.py`. Tracking + Smart Tagging in, `frame,label` possession
CSV out. numpy only.

## Reproduce

Tested with Python 3.14.7 and `numpy==2.5.3` (`requirements.txt` in this folder). The match data
goes in the repository's `./data/` as described in the main README, or anywhere `MTAH_DATA_DIR`
points; only the ATD `.txt`/`.xml` and the Smart Tagging `.xml` are read.

From the repository root:

```bash
pip install -r haeys/requirements.txt
python haeys/run.py --team Vector --shape             # -> haeys/out/Vector.csv
python evaluation/validate_submission.py haeys/out/Vector.csv
md5sum haeys/out/Vector.csv                           # c6b27c75947357d87becd893bc5ebb30
```

The run is deterministic and the file name does not affect the content. The first run parses the
ATD (~25 s) and caches it in `haeys/out/cache/`; later runs take a few seconds.
`python haeys/run.py --inspect` prints what the ATD export contains.

## How it works

1. **Load** (`possession/data.py`) — ATD to metres on the tactical timeline, Smart Tagging to frames.
2. **Evidence** (`possession/features.py`) — per-frame scores for H, A and D:
   - *control*: a player within 2.5 m of the ball with the other team clearly further away;
     partial weight on a ball interpolated across gaps of up to 2 s; faded out when play is calm
   - *dead*: ball beyond the lines, players' mean speed low, set-piece tags up to the detected
     restart kick, graded frames before each half's kickoff
   - *shape* (`--shape`): logistic model on collective team shape, trained on algorithmic
     pseudo-labels from confident control frames, used only where control is silent
3. **Decode** (`possession/decode.py`) — three-state Viterbi with switching costs, so a change
   needs sustained evidence: no flicker, but short real possessions survive.

Every constant is either a football prior (metres, m/s, seconds) or measured from the tracking
data itself. None is fitted to the public ground truth, and no labels of any kind are used.

`CONTROL_RADIUS` is one of the measured ones: the 95th percentile of the ball-to-carrier distance
while a player and the ball move together for a second (`python haeys/tools/measure_control_radius.py`).

## Tests

End to end on a synthetic bundle in the real file formats, with a known truth:

```bash
python haeys/tests/make_synthetic.py /tmp/fake
MTAH_DATA_DIR=/tmp/fake python haeys/run.py --team synthetic --out /tmp/fake/out
python haeys/tests/score.py /tmp/fake/truth.csv /tmp/fake/out/synthetic.csv
```

The match data and everything derived from it (`out/`, caches) are never committed.

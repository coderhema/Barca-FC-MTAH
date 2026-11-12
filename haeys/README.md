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
md5sum haeys/out/Vector.csv                           # d43f442be2aecfce7940f4b9fd4cb9c2
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
   - *restart*: after a set piece with a tagged team, while the ball is still on its first
     flight, the other team's proximity counts for the taker (a defender under a corner is not
     in control of it)
   - *shape* (`--shape`): logistic model on collective team shape, trained on algorithmic
     pseudo-labels from confident control frames, used only where control is silent
3. **Decode** (`possession/decode.py`) — three-state Viterbi with switching costs, so a change
   needs sustained evidence: no flicker, but short real possessions survive.

Every constant is either a football prior (metres, m/s, seconds) or measured on **external open
data**. None is fitted, tuned or calibrated on the competition's tracking data, its video or its
ground truth, following Talent Arena's rules clarification (parameters may only be tuned on
external data).

`CONTROL_RADIUS` (2.0 m) is the measured one: the 95th percentile of the ball-to-carrier distance
while a player and the ball move together for a second, measured on SkillCorner's public open data
(<https://github.com/SkillCorner/opendata>, matches 1874553 and 1886347):

```bash
python haeys/tools/measure_control_radius_skillcorner.py /path/to/*_tracking_extrapolated.jsonl
```

An earlier version measured this value (2.5 m) on the competition tracking data; that tool has been
removed. The public 5-minute ground truth is used only once, to report the final submission's score,
never to choose between versions or settings.

## Tests

End to end on a synthetic bundle in the real file formats, with a known truth:

```bash
python haeys/tests/make_synthetic.py /tmp/fake
MTAH_DATA_DIR=/tmp/fake python haeys/run.py --team synthetic --out /tmp/fake/out
python haeys/tests/score.py /tmp/fake/truth.csv /tmp/fake/out/synthetic.csv
```

The match data and everything derived from it (`out/`, caches) are never committed.

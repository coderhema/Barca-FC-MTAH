# Barca-FC-MTAH

Frame-level ball possession for **More than a Hack 2027 — Phase 1: Ball Possession**
(FC Barcelona vs Athletic Club, LaLiga 2024-25 matchday 2), organised by Metrica Sports,
FC Barcelona and Mobile World Capital Barcelona.
Official challenge repository: https://github.com/metrica-sports/more-than-a-hack-2027

For every one of the 199,319 frames of the tactical video, `baseline.py` writes one label:
`H` (home in possession, ball alive), `A` (away in possession, ball alive), `D` (dead ball) or
`X` (ungraded frame / no answer), as a `frame,label` CSV.

## Requirements

- Python **3.13.5** (the version this was run and tested with; see `.python-version`)
- `numpy==2.5.4`, `pandas==3.0.6` (pinned in `requirements.txt`)

## Setup

```bash
git clone https://github.com/coderhema/Barca-FC-MTAH.git
cd Barca-FC-MTAH
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the data

The match data is gated and is **not** in this repository. Get it as described in the official
[DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md): enrol, accept
the terms, and use the download link you receive. Unpack it into `./data`, keeping the original
names:

```
data/
└─ tactical/
   ├─ BAR-ATH_tactical_atd.xml
   ├─ BAR-ATH_tactical_atd.txt
   └─ BAR-ATH_tactical_smart_tagging.xml
```

Only these three tactical files are read. Alternatively, keep the data elsewhere and point
`MTAH_DATA_DIR` at the folder that contains `tactical/`:

```bash
export MTAH_DATA_DIR=/path/to/mtah2027        # Windows PowerShell: $env:MTAH_DATA_DIR = "D:\mtah2027"
```

`data/` and every match asset are listed in `.gitignore`.

## Regenerate the submission

```bash
python baseline.py
```

This writes `<TEAM_NAME>.csv` next to `baseline.py` (set `TEAM_NAME` at the top of `baseline.py`),
or any path with `python baseline.py --out path/to/file.csv`. The algorithm is deterministic: the
same data and pinned dependencies produce the same file.

## Validate

```bash
python evaluation/validate_submission.py TEAM_NAME.csv
```

This is the organisers' validator (same checks as the official scorer): header, exactly 199,319
rows, frames 0…199318 in order, labels in `H`/`A`/`D`/`X`.

## Approach (summary)

Input: the tactical automatic tracking (players and ball) and the Smart Tagging set pieces. No
video, homography, broadcast data, ground truth or hand labels are used. All constants are fixed
football priors (in metres, seconds and m/s), not tuned against any ground truth.

1. **Control.** On frames where the ball is tracked, a team controls the ball when its nearest
   player is within 2 m of it and at least 1 m closer than the nearest opponent.
2. **Hysteresis.** Possession changes only when the other team keeps control for 12 consecutive
   frames (0.5 s). The change is dated back to the first frame of that control. Touches,
   deflections and loose balls leave possession with the last team.
3. **Missing ball.** While play is live and the ball is not tracked, the last team keeps possession.
4. **Dead ball (`D`).** The ball tracked outside the pitch (and the slow stretch that follows), any
   stretch of at least 3 s where median player speed is below 2 m/s and the ball is missing or
   still, the 2 s before a tagged set piece when play is slow, and the second half before its
   kickoff (frame 113,742).
5. **Restarts.** After a dead ball, the team named on the set-piece tag (when not `N/A`) starts
   with the ball. Otherwise it is the first team in control, falling back to the team nearest the ball.
6. **Clean-up.** Possessions shorter than 2 s are merged into the neighbouring possession.
7. Frames outside the two graded halves are written as `X`.

## Repository layout

```
baseline.py                          the algorithm; writes the submission CSV
games.py                             match constants and data paths (Metrica Sports, MIT)
starter/load_smart_tagging.py        Smart Tagging reader (Metrica Sports, MIT)
evaluation/possession_metrics.py     metric library used by the validator (Metrica Sports, MIT)
evaluation/validate_submission.py    official submission validator (Metrica Sports, MIT)
requirements.txt                     pinned dependencies
```

## Licence

Our code is MIT-licensed, © 2026 Olugbemi Tolulope. Files copied from the official repository
remain © Metrica Sports under MIT (see `NOTICE` and `LICENSE`). The match data is not included and
is not covered by either licence.

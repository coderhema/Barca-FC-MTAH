# Barca-FC-MTAH

Ball possession for every frame of a real football match, built for
**More than a Hack 2027 (Phase 1: Ball Possession)**, organised by Metrica Sports, FC Barcelona and
Mobile World Capital Barcelona.
Official challenge repository: <https://github.com/metrica-sports/more-than-a-hack-2027>

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [Quick start](#2-quick-start)
3. [Step-by-step setup](#3-step-by-step-setup)
4. [Getting the data](#4-getting-the-data)
5. [Running it](#5-running-it)
6. [Validating the output](#6-validating-the-output)
7. [How the method works](#7-how-the-method-works)
8. [Reproducibility](#8-reproducibility)
9. [Troubleshooting / FAQ](#9-troubleshooting--faq)
10. [Repository layout](#10-repository-layout)
11. [Licence and credits](#11-licence-and-credits)

---

## 1. What this is

**The challenge.** The organisers give every registered team one full LaLiga match, **FC Barcelona vs
Athletic Club** (2024-25, matchday 2). It comes with video, automatic player and ball tracking, and
automatic tactical tags. The task is to say, for **every frame** of the tactical video (199,319
frames at 25 frames per second), which team has the ball.

**The output** is a CSV file with one row per frame:

```
frame,label
0,X
1,X
...
11185,H
...
```

| label | meaning |
|---|---|
| `H` | **Home** team (Barcelona) has the ball, ball in play |
| `A` | **Away** team (Athletic Club) has the ball, ball in play |
| `D` | **Dead ball**: the ball is not in play (out, foul, stoppage, …) |
| `X` | **No answer**. Used here for frames that are not graded (before kickoff, half-time, after the final whistle) |

**What this repo does.** `baseline.py` reads the provided tracking data and tags, decides a label
for every frame using simple football rules (see [section 7](#7-how-the-method-works)), and writes the
CSV. `evaluation/validate_submission.py` is the organisers' own checker, included so you can confirm
the file is well-formed before submitting it.

---

## 2. Quick start

Already have Python 3.13 and the data? Run this from the repo folder (macOS / Linux):

```bash
git clone https://github.com/coderhema/Barca-FC-MTAH.git
cd Barca-FC-MTAH
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# put the match data in ./data/tactical/ (see section 4)
python baseline.py
python evaluation/validate_submission.py Vector.csv
```

On Windows, use `py -3.13 -m venv .venv` and `.venv\Scripts\activate` instead (see section 3).

---

## 3. Step-by-step setup

### 3.1 Install Python 3.13

This project was run and tested with **Python 3.13.5** (also recorded in `.python-version`). The pinned
library versions in `requirements.txt` need a recent Python, so use 3.13.

| OS | How to install |
|---|---|
| **Windows** | Download the 3.13 installer from <https://www.python.org/downloads/windows/>. In the installer, tick **"Add python.exe to PATH"**. |
| **macOS** | Download the 3.13 installer from <https://www.python.org/downloads/macos/>, or use Homebrew: `brew install python@3.13` |
| **Linux** | Use your package manager (e.g. `sudo apt install python3.13 python3.13-venv` where available) or <https://www.python.org/downloads/source/> |

Check it worked:

```bash
python --version        # Windows (or: py -3.13 --version)
python3 --version       # macOS / Linux
```

It should print `Python 3.13.x`.

### 3.2 Get the code

```bash
git clone https://github.com/coderhema/Barca-FC-MTAH.git
cd Barca-FC-MTAH
```

(No git? On the GitHub page click **Code → Download ZIP**, unzip it, and open a terminal in that folder.)

### 3.3 Create and activate a virtual environment

A virtual environment ("venv") keeps this project's libraries separate from everything else on your computer.

**Windows (PowerShell)**
```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell refuses with "running scripts is disabled", run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Windows (Command Prompt)**
```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Once it's active, your prompt starts with `(.venv)`. Inside the venv, `python` and `pip` refer to
the venv's copies on every OS. Run `deactivate` to leave it.

### 3.4 Install the dependencies

```bash
pip install -r requirements.txt
```

This installs exactly `numpy==2.5.4` and `pandas==3.0.6`. Nothing else is needed.

---

## 4. Getting the data

### 4.1 Where it comes from

The match data is **gated**. Per the official
[DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md), it is released
only to participants who **enrol** in More than a Hack 2027 and **accept the terms of data use**. Once
you do, you receive a download link. It is **not** in this repository and never will be.

### 4.2 Where to put it

Unpack the bundle into a folder called `data` inside this repo, **keeping the original file names**.
The bundle contains more than this, but `baseline.py` only reads the three files marked ◀:

```
Barca-FC-MTAH/
└─ data/
   ├─ tactical/
   │  ├─ BAR-ATH_tactical.mp4                  (not used)
   │  ├─ BAR-ATH_tactical_atd.xml              ◀ tracking metadata (which track belongs to which team)
   │  ├─ BAR-ATH_tactical_atd.txt              ◀ tracking: players and ball, one line per frame
   │  ├─ BAR-ATH_tactical_homography.csv       (not used)
   │  └─ BAR-ATH_tactical_smart_tagging.xml    ◀ automatic tactical tags (set pieces are used)
   └─ broadcast/                               (not used)
      ├─ BAR-ATH_broadcast.mp4
      ├─ BAR-ATH_broadcast_atd.xml
      ├─ BAR-ATH_broadcast_atd.txt
      └─ BAR-ATH_broadcast_homography.csv
```

These names come from `games.py`, so `games.py` is the one place to change them if the bundle ever differs.

### 4.3 Or keep the data somewhere else (`MTAH_DATA_DIR`)

Point the `MTAH_DATA_DIR` environment variable at the folder that **contains** `tactical/`:

```powershell
# Windows PowerShell (current window only)
$env:MTAH_DATA_DIR = "D:\mtah2027"
```
```bat
:: Windows Command Prompt (current window only)
set MTAH_DATA_DIR=D:\mtah2027
```
```bash
# macOS / Linux (current terminal only)
export MTAH_DATA_DIR=/path/to/mtah2027
```

### 4.4 ⚠️ Rules about the data

- **Never commit the data to git or upload it anywhere.** This repo is meant to be public, and the data
  is licensed for the competition only. `data/` and all match file types are already in `.gitignore`.
  Run `git status` before every commit to make sure.
- **Never paste or upload the data (or any extract of it) into online AI tools** such as ChatGPT,
  Claude or Gemini. The competition terms forbid making the data available to any third-party hosted
  or cloud-based LLM. Using AI to help with *code* is fine.
- Labels must come from an algorithm. Hand-labelling or hand-correcting the provided match is not
  allowed (official RULES.md).

---

## 5. Running it

### 5.1 Set your team name

The team name is set near the top of `baseline.py`. It must match the name **every member registered with**, which for us is "Vector":

```python
TEAM_NAME = "Vector"
```

The organisers ask for the CSV to be named after the team (`<team_name>.csv`), and this is used as the
default file name.

### 5.2 Run

With the venv active, from the repo folder:

```bash
python baseline.py
```

Expected console output (numbers are illustrative):

```
reading tracking (46 slots) ...
  ball tracked on 56.9% of frames
wrote /…/Barca-FC-MTAH/YourTeam.csv {'A': …, 'D': …, 'H': …, 'X': …}
next: python evaluation/validate_submission.py /…/Barca-FC-MTAH/YourTeam.csv
```

- **"46 slots"**: the tracking has 46 track slots (DATA.md).
- **"ball tracked on … %"**: DATA.md says the ball is present on 56.9% of all frames
  (113,488 / 199,319), so you should see about **57%**. A very different number means the data
  didn't load correctly.
- If the Smart Tagging file is missing, you'll also see a line starting `(no Smart Tagging used: …`
  and the run continues without it. **The result then differs**, so make sure the file is in place before producing the
  file you submit.

### 5.3 Where the CSV goes

By default it is written **next to `baseline.py`** as `<TEAM_NAME>.csv`. To write somewhere else:

```bash
python baseline.py --out results/my_run.csv
```

Missing folders in the `--out` path are created automatically.

### 5.4 How long it takes

The script reads the 199,319-line tracking file once in plain Python, then does the rest with numpy.
Expect a few seconds to a couple of minutes depending on your computer. It needs a few hundred MB of
free memory.

---

## 6. Validating the output

**Always validate the exact file you are about to submit:**

```bash
python evaluation/validate_submission.py YourTeam.csv
```

This is the organisers' validator. It runs the same format checks as the official scorer:
the header is `frame,label`, there are exactly 199,319 rows, frames run 0…199318 in order, there are no blank cells,
and the labels are `H`/`A`/`D`/`X`. A passing file looks like this (counts will differ):

```
OK    YourTeam.csv
      199319 rows, frame 0..199318

      label counts
        H    …
        A    …
        D    …
        X    40462   20.3%

      reports … possessions of mean … s
      ...

Valid. Note this checks the format only -- it says nothing about whether the labels are right.
```

`X` is always 40,462 rows: these are the ungraded frames before kickoff, at half-time and after the final whistle.
A failing file prints `FAIL` and one line explaining why (see [section 9](#9-troubleshooting--faq)).

### Optional: check against the public 5-minute window

The official repo ships the true labels for the **first 5 minutes of play only**, plus its scorer. To
check the pipeline end-to-end:

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git
cp YourTeam.csv more-than-a-hack-2027/evaluation/results/     # Windows: copy YourTeam.csv more-than-a-hack-2027\evaluation\results\
cd more-than-a-hack-2027
python evaluation/run_evaluation.py
```

It prints a scoreboard with `spellF1` (the ranking metric) and `macroF1` for every CSV in
`evaluation/results/`, including the organisers' four baselines.

> **Use this only to check the pipeline works** (frames aligned, labels meaning what you think).
> The official rules forbid training or tuning on these 5 minutes. They are also too short to compare
> versions. Final ranking uses the private full-match ground truth.

---

## 7. How the method works

In plain words: **a team gets the ball when one of its players is clearly closest to it and stays
there for half a second. It keeps the ball until the other team does the same, or the ball goes dead.**

All values are football common sense. They were fixed without reference to any ground truth.

| # | Rule | What it does | Value | Constant in `baseline.py` |
|---|---|---|---|---|
| 1 | **Control** | On frames where the ball is tracked, a team controls it if its nearest player is within this distance… | 2.0 m | `CONTROL_RADIUS_M` |
| | | …and at least this much closer than the nearest opponent | 1.0 m | `MARGIN_M` |
| 2 | **Hysteresis** | Possession changes only after the other team controls the ball this many frames in a row. The change is then dated back to when their control began. Touches, deflections and loose balls leave the ball with the last team. | 12 frames (0.48 s) | `CONFIRM_FRAMES` |
| 3 | **Ball not tracked** | While play is live, the last team keeps the ball | – | – |
| 4a | **Dead ball: out** | Ball tracked outside the pitch by more than this (pitch coordinates run 0–1), plus the slow stretch right after it | 0.005 (≈0.5 m) | `OUT_MARGIN` |
| 4b | **Dead ball: frozen game** | Median player speed below this… | 2.0 m/s | `SLOW_PLAYER_MS` |
| | | …and the ball missing or moving slower than this… | 1.0 m/s | `BALL_STILL_MS` |
| | | …for at least this long | 3.0 s | `DEAD_MIN_S` |
| 4c | **Dead ball: before a set piece** | Frames in the 2 s before a tagged `SET PIECES` instance, if players are slow | 2 s | in `dead_mask()` |
| 4d | **Dead ball: 2nd-half start** | The second half is graded from frame 110,808 but kicks off at 113,742 (official FAQ). Everything before kickoff is `D`. | frame 113742 | `SECOND_HALF_KICKOFF` |
| 5 | **Restarts** | After a dead ball, the team on the set-piece tag (`Barcelona` → H, `Atletic Club` → A; `N/A` ignored) starts with the ball. Otherwise it's the first team in control, falling back to the team nearest the ball. | – | `TEAM_TAGS`, `label_half()` |
| 6 | **Minimum possession** | Possessions shorter than this are merged into the neighbouring one | 2.0 s | `MIN_SPELL_S` |
| 7 | **Ungraded frames** | Outside the two graded halves (frames 11,185–90,651 and 110,808–190,197) → `X` | – | `half_video_blocks()` in `games.py` |

**Where to change them:** all constants are in one block at the top of `baseline.py`, under
`# ---------------- parameters`. The logic lives in these functions:

| function | job |
|---|---|
| `read_atd()` | reads the tracking file into arrays |
| `load_set_pieces()` | reads `SET PIECES` tags from the Smart Tagging file |
| `control_per_frame()` | rule 1 |
| `dead_mask()` | rule 4 |
| `label_half()` | rules 2, 3 and 5, one half at a time |
| `merge_short()` | rule 6 |
| `main()` | ties it together and writes the CSV |

Inputs used: the tactical tracking and the Smart Tagging set pieces. **Not used:** the videos, the
homography, the broadcast data, any ground truth and any hand labels.

---

## 8. Reproducibility

- The algorithm has **no randomness**. With the same data and the pinned versions, `python baseline.py`
  produces a **byte-identical file** every run. You can check this with a checksum:
  - macOS / Linux: `shasum -a 256 YourTeam.csv`
  - Windows PowerShell: `Get-FileHash YourTeam.csv`
- Versions are pinned: Python 3.13.5, `numpy==2.5.4`, `pandas==3.0.6`.
- The output depends on all three input files in section 4.2. Make sure the Smart Tagging file is present.

---

## 9. Troubleshooting / FAQ

**`OSError: match data not found: …/tactical/BAR-ATH_tactical_atd.xml`**
The script couldn't find the data. The message lists the exact paths it tried. Either unpack the bundle
so those paths exist (section 4.2, with `tactical/` directly inside `data/`, not `data/<something>/tactical/`),
or set `MTAH_DATA_DIR` to the folder that contains `tactical/` (section 4.3).

**`pip install` fails, or `No matching distribution found for numpy==2.5.4`**
You're probably on an older Python. Check `python --version` inside the venv. If it isn't 3.13, delete
the `.venv` folder and recreate it with Python 3.13 (section 3.3).

**`python` is not recognised (Windows)**
Use `py -3.13` instead, or reinstall Python with **"Add python.exe to PATH"** ticked. Inside an active
venv, `python` always works.

**`python: command not found` (macOS / Linux)**
Use `python3` outside the venv. Inside an active venv, `python` works.

**Validator says `row count 164488 != 199319` (or any number near 164k)**
That's the broadcast video's frame count, so a broadcast file was used instead of the tactical one. The
submission must be indexed to the **tactical** video (199,319 frames). Check that the files in
`data/tactical/` are the `BAR-ATH_tactical_*` ones.

**Validator says `FAIL`, other reasons**
- `row count … != 199319`: the file is truncated or not from this script. Re-run `python baseline.py`.
- `'frame' column is not 0..199318 in order` or `empty label in … rows`: the file was edited
  by hand or by a spreadsheet program. Don't open and re-save it in Excel. Regenerate it.
- `could not read: …`: wrong file, or not a CSV.

**"ball tracked on …%" is far from 57%**
The wrong tracking file was loaded, or the file is incomplete or corrupted. Re-download and unpack it.

**Windows path problems**
- Use quotes around paths with spaces: `$env:MTAH_DATA_DIR = "C:\Users\Me\My Data\mtah2027"`.
- Keep the folder names exactly `data\tactical\` with the original file names (`BAR-ATH_tactical_atd.txt`, etc.).
  Windows may hide extensions, so make sure a file isn't really `…_atd.txt.txt`.
- Unzip the bundle fully. Don't run from inside the ZIP preview.
- `MTAH_DATA_DIR` set with `$env:` or `set` lasts only for that terminal window.

**Can I tune the parameters on the public 5 minutes?**
No. The official rules say those 5 minutes are for pipeline checking only, not training or tuning.

---

## 10. Repository layout

| path | what it is | author |
|---|---|---|
| `baseline.py` | the algorithm. Writes the submission CSV | this repo |
| `games.py` | match constants, frame ranges of each half, data file paths, `MTAH_DATA_DIR` | Metrica Sports (MIT) |
| `starter/load_smart_tagging.py` | reads the Smart Tagging XML into frame intervals | Metrica Sports (MIT) |
| `evaluation/validate_submission.py` | official submission validator | Metrica Sports (MIT) |
| `evaluation/possession_metrics.py` | metric and validation library used by the validator | Metrica Sports (MIT) |
| `requirements.txt` | pinned dependencies | this repo |
| `.python-version` | Python version used (3.13.5) | this repo |
| `.gitignore` | keeps data, outputs, venvs and caches out of git | this repo |
| `LICENSE`, `NOTICE` | licences and third-party credits | – |

---

## 11. Licence and credits

- Code in this repository: **MIT**, © 2026 Olugbemi Tolulope (see `LICENSE`).
- `games.py`, `starter/load_smart_tagging.py`, `evaluation/validate_submission.py` and
  `evaluation/possession_metrics.py` are copied from the official repository
  <https://github.com/metrica-sports/more-than-a-hack-2027> (commit `f8f2629`). They are
  **© 2026-2027 Metrica Sports, MIT License**, unmodified apart from a header naming their origin (see `NOTICE`).
- The **match data** is not included and is **not** covered by any licence here. It is licensed by the
  organisers for the competition only.
- Thanks to **Metrica Sports**, **FC Barcelona / Barça Innovation Hub** and **Mobile World Capital
  Barcelona** for the challenge, the data and the tooling.

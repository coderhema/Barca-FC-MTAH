# Technical document — Team Vector
## More than a Hack 2027, Phase 1: Ball Possession

**Team name:** Vector  
**Repository:** https://github.com/coderhema/Barca-FC-MTAH  
**Submission file:** `Vector.csv` (regenerate with the commands in §7)  
**Language note:** This document uses Simplified Technical English (ASD-STE100 style): short sentences, one idea per sentence, fixed terms.

---

## 1. Purpose

This document describes the method that makes the possession labels.

The method is an algorithm. No person writes labels for this match. No person corrects the output by hand.

The ranked metric is `spellF1`. The method is built to recover full possessions, not only frame labels.

---

## 2. Approach — what we built, and why

### 2.1 Name

The submission pipeline is **haeys** (said like “eyes”).

Code path: `haeys/`.

Entry point: `haeys/run.py`.

### 2.2 Design choice

We do **not** use the video as an input to the algorithm.

We use:

- tactical Automatic Tracking Data (ATD)
- Smart Tagging set-piece instances

**Why this choice**

1. The challenge is not a tracking challenge. The organisers give tracking and tags so teams can answer possession.
2. The ball is missing on many graded frames. A method must keep possession across gaps. Geometry plus temporal decoding does that.
3. A three-state decoder with switch costs fits the ranking metric better than a short majority window. One deflection must not split a possession. Short real control must still survive.

We keep a second pipeline, `baseline.py`, as a simple reference. It is **not** the submission.

### 2.3 Pipeline stages

The system has three stages.

| Stage | Module | Function |
|---|---|---|
| 1. Load | `haeys/possession/data.py` | Read ATD and Smart Tagging. Map all times to tactical video frames. Convert pitch positions to metres. |
| 2. Evidence | `haeys/possession/features.py` | For each frame, make three scores: home (H), away (A), dead (D). |
| 3. Decode | `haeys/possession/decode.py` | Find the best state path with Viterbi. Then merge very short H/A spells. |

Optional stage (diagnostics only, not scored):

| Stage | Module | Function |
|---|---|---|
| Export | `haeys/possession/diagnostics.py` | Write Cavalry CSVs under `viz/`. |

### 2.4 Why Viterbi, not only rules

`baseline.py` uses control radius, hysteresis, and a short-spell merge.

haeys uses **scores** and **switch costs**:

- A change H↔A costs sustained control evidence (about 0.48 s of clean control).
- Entry to dead costs more. A short calm period is not a stoppage.
- Exit from dead costs less. A restart kick is sharp.

This matches CHALLENGE.md: a touch without control is not a change. Possession stays with the last team while the ball is loose or contested.

---

## 3. Assets we use and assets we do not use

### 3.1 Used

| Asset | How we use it |
|---|---|
| `BAR-ATH_tactical_atd.txt` | Player and ball positions and speeds, one line per frame |
| `BAR-ATH_tactical_atd.xml` | Team id for each track slot |
| `BAR-ATH_tactical_smart_tagging.xml` | Set-piece intervals, team tags, kick-off types |

### 3.2 Not used (tried or available, then dropped for the submission)

| Asset | Reason |
|---|---|
| Tactical video | Not required. Labels come from tracking geometry and tags. |
| Broadcast video | Not synchronised to the tactical timeline. |
| Broadcast ATD / homography | Not on the submission timeline. |
| Tactical homography | ATD already gives pitch coordinates. |
| Public 5-minute ground truth | Pipeline check and final report only. **Not** used to train or to select parameters. |
| Private full-match ground truth | Forbidden. Not available. |
| Human event data for this match | Forbidden. |

### 3.3 External data (other games)

One constant uses **external open data**:

- **Source:** SkillCorner open tracking  
  https://github.com/SkillCorner/opendata  
  matches `1874553` and `1886347`
- **Measure:** 95th percentile of ball-to-carrier distance in clear carries
- **Result:** `CONTROL_RADIUS = 2.0` m
- **Tool:** `haeys/tools/measure_control_radius_skillcorner.py`

No labels from the provided match train any model.

---

## 4. Training data

| Type | Used? | Detail |
|---|---|---|
| Hand labels on the provided match | **No** | Forbidden by RULES.md |
| Hand labels on other matches | **No** | Not used in this submission |
| Supervised learning on this match | **No** | |
| Self-supervised shape model | **Yes (optional flag)** | Logistic model on team-shape features. Pseudo-labels come only from frames where the control rule is already confident. No person selects those frames. The model speaks only where control is silent (`--shape`). |

---

## 5. Possession rules we implement

These rules follow CHALLENGE.md. Constants are football priors or external measures. They are **not** fit on the public 5-minute ground truth.

### 5.1 Labels

| Label | Meaning in our system |
|---|---|
| `H` | Home (Barcelona) has controlled possession. Ball is alive. |
| `A` | Away (Athletic Club) has controlled possession. Ball is alive. |
| `D` | Ball is not in play (stoppage). |
| `X` | Outside graded halves (before kick-off, half-time, after full time). Never scored. |

### 5.2 Controlled possession (H / A)

- A player near the ball supports control.
- Full weight inside **2.0 m**.
- Weight falls in a straight line over the next **1.0 m**.
- The other team must be clearly farther (margin scale **2.0 m**).
- If the ball is missing, we may fill short gaps (up to **2 s**) by linear interpolation. Weight is lower in the middle of a gap.
- If play looks calm (players slow), control weight falls. A player who holds the ball for a throw-in is not live possession yet.

### 5.3 What is not a change

- One touch without sustained control
- A deflection without interception
- An aerial duel without later control

**Mechanism:** H↔A switch cost = **12** evidence units ≈ **0.48 s** of clean control (same prior as `CONFIRM_FRAMES` in `baseline.py`).

### 5.4 Loose and contested ball

If neither team has clear control, scores stay near zero. The decoder **keeps the last live state**. This is the CHALLENGE rule: while play is live, one team has possession; loose ball stays with the last team.

### 5.5 Dead ball (D)

We add dead evidence when:

1. The ball is tracked **out of pitch** (margin **0.5 m**).
2. After the ball goes out, players stay **slow** (out-hold). The stoppage continues during the regroup.
3. A **frozen** stretch lasts at least **3.0 s**: players slow and ball idle or missing.
4. Smart Tagging marks a **set piece**. Dead holds until the detected restart kick.
5. Graded frames **before a half kick-off** (FAQ: second half is graded before the kick).

We reduce dead evidence when the ball moves fast (play is alive again).

### 5.6 Restarts

- After a tagged set piece with a known team, the **taker** owns the first flight of the ball.
- Near defenders under a corner do not steal possession during that flight.
- Throw-ins tagged with team `N/A` are skipped for team seed.

### 5.7 Short possessions

After Viterbi, H/A runs shorter than **2.0 s** merge into the neighbour possession.

**Why:** `spellF1` punishes invented short spells.

### 5.8 Ungraded frames

Frames outside the two graded halves are `X`.

---

## 6. Main parameters (priors)

| Parameter | Value | Role |
|---|---|---|
| `CONTROL_RADIUS` | 2.0 m | Full control distance (SkillCorner external measure) |
| `CONTROL_SOFT` | 1.0 m | Soft falloff beyond the radius |
| `CONTROL_MARGIN` | 2.0 m | Opponent clearance scale |
| `MAX_BALL_GAP` | 2.0 s | Max interpolation gap |
| `DEAD_MIN_S` | 3.0 s | Min frozen length for strong dead |
| `SWITCH_TEAM` | 12 | H↔A cost (~0.48 s) |
| `ENTER_DEAD` | 30 | Alive → D cost |
| `LEAVE_DEAD` | 10 | D → alive cost |
| `MIN_SPELL_S` | 2.0 s | Post merge of short H/A spells |
| `W_SHAPE` | 0.15 | Weight of optional shape model |

All of the above are fixed before any look at private labels. The public 5-minute file is not a tuning set.

---

## 7. Reproduction

### 7.1 Requirements

- Python 3.9+ (developed with 3.12 / 3.13-class environments)
- `numpy==2.5.3` from `haeys/requirements.txt`
- Match data under `./data/tactical/` or `MTAH_DATA_DIR` (see repository README)

Do **not** put match data in git.

### 7.2 Commands

From the repository root:

```bash
pip install -r haeys/requirements.txt
python haeys/run.py --team Vector --shape
python evaluation/validate_submission.py haeys/out/Vector.csv
```

Output path: `haeys/out/Vector.csv`.

Copy or rename to the team file name for the form if required: `Vector.csv`.

The run is **deterministic**. Same data and same pins give the same file.

First run parses ATD and writes a cache under `haeys/out/cache/`. Later runs are faster.

### 7.3 Optional diagnostics (not submitted)

```bash
python haeys/run.py --team Vector --shape --diagnostics viz
```

Writes `viz/frames.csv` and `viz/players.csv` for timeline tools (for example Cavalry). These files are not the scored submission.

### 7.4 Synthetic self-check (no competition GT)

```bash
python haeys/tests/make_synthetic.py /tmp/fake
MTAH_DATA_DIR=/tmp/fake python haeys/run.py --team synthetic --out /tmp/fake/out
python haeys/tests/score.py /tmp/fake/truth.csv /tmp/fake/out/synthetic.csv
```

---

## 8. Validation and public-window report

### 8.1 Format

`evaluation/validate_submission.py` reports:

- 199,319 rows
- frames 0…199318
- labels in `H`/`A`/`D`/`X`
- `X` count 40,462 on ungraded structure

### 8.2 Full-match structure (no private GT)

Approximate structure of the regenerated file:

- about **242** possessions
- mean length about **14.2 s**
- label mix near: H 27.5%, A 15.6%, D 36.6%, X 20.3%

### 8.3 Public 5-minute ground truth (pipeline check only)

We run the official scorer **once** on the published 5 minutes to check alignment. RULES.md forbids training and tuning on this window.

Illustrative board position (public window only — **cannot rank the competition**):

| File | macroF1 | spellF1 |
|---|---:|---:|
| haeys (`--shape`) | ~0.87 | ~0.59 |
| Our `baseline.py` | ~0.71 | ~0.47 |
| Organiser `baseline_smooth125` | ~0.58 | ~0.42 |

Bootstrap bands on 17 possessions are wide. This table is a **sanity check**, not a claim of private rank.

Organisers’ published full-match baseline to beat: `spellF1` **0.2433** (`baseline_smooth50`). We cannot measure our private score.

---

## 9. Limitations

1. **No ball on many frames.** Interpolation and last-state carry help, but long gaps stay hard.
2. **No player identities.** ATD is team-level slots. Identity switches can move “nearest player”.
3. **Smart Tagging is automatic.** False set pieces and missed tags exist. We treat tags as a prior, not as truth.
4. **Dead vs live at the margin.** Calm live play can look like a stoppage. Frozen length and kick speed reduce this, but errors remain.
5. **Shape model is weak and local.** It only fills silence. It can be wrong in unusual shapes.
6. **We do not use video.** A future detector on video could improve ball and contact cues if it stays fully automatic and shareable.
7. **Public window is small.** We must not overfit it. Private full-match rank is unknown until the organisers score.

---

## 10. What we would improve with more time

1. Better ball recovery from video **by algorithm only**, with code in the public repo.
2. Stronger external calibration of switch costs on open multi-match data.
3. Explicit multi-pass control proxies (CHALLENGE.md) on other-game labels if needed.
4. Joint use of more Smart Tagging phases (build-up, transition) as soft priors, with care for noise.

---

## 11. Rules compliance summary

| Rule | Our compliance |
|---|---|
| Labels from an algorithm | Yes — haeys only |
| No hand labels on this match | Yes |
| No private GT | Yes |
| Public 5 min not for train/tune | Yes — report only |
| No human event data for this match | Yes |
| Reproducible from public repo | Yes — §7 |
| Organisers can inspect the method | Yes — full source |
| Match data not in the repo | Yes — gitignored |

---

## 12. Document control

| Item | Value |
|---|---|
| Document type | Competition technical document (SUBMISSION.md) |
| Style | ASD-STE100-oriented Simplified Technical English |
| Pipeline version | `haeys` on branch `feature/haeys-improvements` |
| Primary command | `python haeys/run.py --team Vector --shape` |
| Scored artifact | `haeys/out/Vector.csv` → upload as `Vector.csv` |

---

*End of technical document.*

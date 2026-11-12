# pumpfun-top5-forecast-01 — top-5 new Solana coins, 24 hours out

Entry for the make.fun competition "Predict the top 5 new Solana coins by
market cap, 24 hours out" (pumpfun-top5-forecast-01). One program,
`main.py` (Python 3.13, standard library only, no network), reads the
house's 10-minute market-cap snapshots from `$MAKEFUN_INPUT_DIR`
(default `/input`) and writes one forecast to `$MAKEFUN_OUTPUT_DIR`
(default `/output`): for each of the top 5 places by market cap at an
answer read, a probability for each coin named being at that place, the
top 5 in order, and a probability for each place's market cap in 82
log10 bins. Deterministic: two runs on the same history write
byte-identical output (fixed-seed splitmix64, sorted orders, no clock).

## How it works

A survival-aware Monte Carlo over each coin's terminal log market cap at
each answer horizon (1,000 paths per horizon, top 150 coins by cap at the
cut, pool coins first):

- **Common market factor.** One draw per path per horizon, shared by every
  live coin: money flows into the day's coins together, so the top coins
  mostly rise and fall together and their relative order is steadier than
  independent draws would say. Mean-reverting: its mean saturates at
  +0.8 log and its sd at 0.32, because the final's answer window is
  18–24 hours after the opening and the inflow has decayed.
- **Pool (graduated) coins.** Small excess drift from their own recent
  climb, a 2% crash hazard per 10 minutes, idiosyncratic noise that is
  smaller for the big coins (0.09 log per 10 minutes at the top). The
  idiosyncratic variance saturates after a few hours: a coin's log-cap
  does not keep diffusing like a Brownian motion for a whole day.
- **Curve (bonding) coins.** Most collapse to the dead floor (1.45) within
  an hour unless they graduate (up to 25%, with a jump in either
  direction); a hot curve coin carries a 20% crash hazard per 10 minutes.
- **Newcomers.** A synthetic field of coins launched after the cut,
  counted in proportion to the horizon (about 80 enter the chart per 10
  minutes, measured). Their caps come from the empirical first-appearance
  distribution the rehearsal history shows (454 coins, more than half
  first appear at the floor, top 1% 3.10–3.47), with a Pareto tail past
  the largest cap seen. Each one is a curve coin, so a newcomer that
  arrived hours ago has almost certainly crashed by the read; only the
  recent rockets and the few that graduated still matter. Only the
  field's top few terminal caps are drawn, exactly as the top order
  statistics of that many uniforms, so the share of each place left
  unnamed is the honest probability that a coin nobody has seen yet
  takes it.

Place probabilities come from the simulation's counts, floored at 0.002
per coin and capped at 0.98 named per place. Size distributions are the
simulation's histogram mixed 70/30 with a normal fit to it (sd floored
by horizon, capped at 0.45), named to 0.92 per place. Every parameter is
documented where it is set in `main.py`; the numbers were chosen from the
rehearsal history's own statistics and from how pump.fun coins behave,
not tuned to any answer.

## Running it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local tests (`devtools/`)

- `score.py` — scores a forecast against held-out marks, following
  RULES.txt exactly (unnamed order share divided by the chart's unnamed
  coins; unnamed size share divided by the place's unnamed bins; floor
  0.0001; top-5 agreement). Verified against the kit's grader source
  (`kit/harness-src/grader/forecast/ranked.ts`).
- `baseline.py` — a Python port of the kit starter's house baseline
  (persistence: 0.30 on the coin then at each place, 0.10 on its
  neighbours; normal sizes, sd 0.35).
- `backtest.py` — truncates the rehearsal history at a cut, forecasts,
  scores against the held-out marks.

Backtest on the rehearsal history (the practice answer's own reads are
21:00–22:00 UTC, which the rehearsal input does not hold, so these are
10–30-minute-ahead rolling cuts instead):

| cut | reads | this entry | house baseline |
|-----|-------|-----------|----------------|
| 20:30 | 20:40, 20:50 | −27.23 | −33.48 |
| 20:20 | 20:30, 20:40, 20:50 | −39.15 | −49.96 |
| 20:40 | 20:50 | −23.20 | −29.87 |

The entry beats the baseline at every cut, by 6–11 log points, on both
order and sizes. (The house baseline scores −40.41 on the practice
forecast over its seven reads; this entry's practice score cannot be
computed locally because the practice answer reads are not in the
rehearsal input.)

Full-history run: ~6 s, ~11 KB forecast, byte-identical across runs.
Format validated: every named mint is in the history, per-place sums
≤ 1, no duplicate rows.

## What is not verified

- **Long horizons.** The final is scored at reads 7–13 hours after the
  cut; the rehearsal only supports 10–50-minute-ahead backtests. The
  long-horizon behaviour (newcomer field, crash hazards, saturating
  volatilities) was checked only by a smoke test: sizes peak at
  4.1–5.2 log SOL and the extreme tail (≥10^6 SOL) carries ≤2% per
  place. The 7–13-hour dynamics are a model, not a measurement.
- **The house AI model forecast.** The prize bar is the higher of the
  house baseline and the house model forecast; the model forecast's
  score is unknown.
- **Newcomer tail and market growth.** Both are extrapolated: the
  Pareto tail from 6 coins above 3.10 log, and the market factor from a
  first-hour growth rate decayed to an 11-hour-old market. If the
  market grows much faster or slower through the answer window, the
  size distributions will be mis-centred (the order ranking is barely
  affected, because the common factor moves all coins together).

## Deliberately not done

The rules allow naming a coin at p 0, which takes it out of a place's
unnamed share without spending mass. That would raise the score when a
newcomer takes a place, but it costs the floor (−9.21) if that coin ever
holds the place, and the model's per-coin place probabilities for small
coins are not precise enough to use it safely. It is left out.

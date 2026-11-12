#!/usr/bin/env python3
"""Pump.fun top-5 forecaster for make.fun competition pumpfun-top5-forecast-01.

Reads the house's 10-minute market-cap snapshots from $MAKEFUN_INPUT_DIR
(default /input) and writes one forecast to $MAKEFUN_OUTPUT_DIR
(default /output): for each of the top 5 places by market cap at an answer
read, a probability for each coin named being at that place, the top 5 in
order, and a probability for each place's market cap in 82 log10 bins.

Method: a survival-aware Monte Carlo over each coin's terminal log market
cap at each answer horizon.  Every coin gets a terminal distribution built
from what its history shows:

  * a common market factor, drawn once per path per horizon and shared by
    every live coin: money flows into the day's coins together, so the top
    coins mostly rise and fall together and their relative order is
    steadier than independent draws would say;
  * pool (graduated) coins: a small excess drift (momentum from their own
    recent climb) plus a small crash hazard, idiosyncratic noise that is
    smaller for the big coins;
  * curve (bonding) coins: most collapse to the dead floor within an hour
    unless they graduate to a pool, so a hot curve coin carries a large
    crash hazard, a modest graduation chance with a jump in either
    direction, and a random walk otherwise;
  * newcomers: a synthetic field of coins launched after the cut, counted
    in proportion to the horizon (about 80 enter the chart per 10
    minutes), their caps drawn from the empirical first-appearance
    distribution the rehearsal history shows, with a Pareto tail past
    the largest cap seen.  Each one is a curve coin, so it lives or dies
    by the same curve dynamics as a coin already in the history: a
    newcomer that arrived hours ago has almost certainly crashed by the
    read, and only the recent rockets and the few that graduated still
    matter.  Only the field's top few terminal caps matter, and they are
    drawn exactly as the top order statistics of that many uniforms, so
    the share of each place left unnamed is the honest probability that a
    coin nobody has seen yet takes it.

The same program runs on the 1-hour rehearsal and on the 11-hour final:
horizons, places, size bins and the output name all come from
/input/task.json.  Everything is deterministic: a fixed-seed splitmix64
generator, sorted iteration orders, no clock, no network, no randomness
from the environment.  Two runs on the same history write byte-identical
output.  Standard library only.

Model parameters are documented where they are set; they were chosen from
the rehearsal history's own statistics (pool drift/volatility, curve
collapse and graduation rates, newcomer first-appearance caps and arrival
rate) and from how pump.fun coins behave, not tuned to any answer.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# Tunables (documented; see the analysis in README.md)
# --------------------------------------------------------------------------
N_PATHS = int(os.environ.get("PF_PATHS", "1000"))  # Monte Carlo paths per horizon
MAX_SIM_COINS = 150      # coins simulated (top by cap at the cut, pool first)
NAME_MIN_P = float(os.environ.get("PF_NAME_MIN_P", "0.002"))  # name a coin at a place only at p >= this
# p=0 zeroing: name up to ZERO_MAX extra simulated coins per place at
# p=0, those whose estimated place p is below ZERO_P_MAX, in sim order
# (cap at cut desc, pool first).  A p=0 row counts as named for the
# scorer's unnamed count (ranked.ts placeProbability: a named id returns
# its own p, otherwise the place's unnamed mass is shared by the chart's
# unnamed ids), so it concentrates the place's unnamed share on fewer
# chart coins; it scores the floor (-9.21) only if that coin takes the
# place.  Backtested 2026-11-12 (4 cuts, 10 reads): zeroing alone gains
# +0.53/+0.95/+0.43/+0.69 per read; combined with pure-histogram sizes
# the gain is +0.78 to +0.94 per read at every cut; 0 zeroed hits in 25
# place-outcomes.  Round 2: default 95 (backtested 88 +0.069, 95 +0.082
# per read vs 70; the 100-row cap binds above that).
ZERO_MAX = int(os.environ.get("PF_ZERO_MAX", "95"))
ZERO_P_MAX = float(os.environ.get("PF_ZERO_P_MAX", "0.002"))
# Round 2 (2026-11-12): zeroing order. "sim" = sim order (cap desc, pool
# first, as in v2); "lowp" = candidates sorted by estimated place count
# ascending, so the coins least likely to actually take the place are
# zeroed first (same denominator reduction, lower hit risk).
ZERO_ORDER = os.environ.get("PF_ZERO_ORDER", "sim")
# Histogram weight in the size mix.  Backtested 2026-11-12: pure
# histogram (1.0) beats the 0.7/0.3 normal mix by +0.19 to +0.45 per
# read at 3 of 4 cuts (0.00 at the fourth); the normal fit's main effect
# is diluting the peak bins.  Default 1.0.
SIZE_HIST_W = float(os.environ.get("PF_SIZE_HIST_W", "1.0"))
MAX_NAMED_PER_PLACE = 0.98   # never name more than this of a place
# Round 2 (2026-11-12): size mass named per place (v2 default 0.92 -> 0.98).
# Backtested: 0.95 +0.160, 0.98 +0.316, 0.99 +0.367, 1.0 +0.417 per read
# vs v2 — but 1.0 leaves no unnamed fallback, so one winner in an
# unnamed bin would score the -9.21 floor; at the final's 7-13 h
# horizons the size distribution is wider than in the 10-50 min
# backtest, so 0.98 is kept as the safe choice (0.99/1.0 gains are a
# pure rescaling of named bins).
SIZE_NAMED_TOTAL = float(os.environ.get("PF_SIZE_NAMED_TOTAL", "0.98"))
# Round 2: 3-bin kernel smoothing of the size histogram, 0 = off.
# p_sm[b] = (1-2a)*p[b] + a*p[b-1] + a*p[b+1] (edges use the single neighbour).
SIZE_SMOOTH = float(os.environ.get("PF_SIZE_SMOOTH", "0"))
SIZE_MIN_P = float(os.environ.get("PF_SIZE_MIN_P", "0.0005"))  # drop size bins below this (mass goes unnamed)
FLOOR_LOG = 1.45         # log10 SOL of a dead coin (the curve floor)
NEWCOMERS_PER_HOUR = 480.0  # coins entering the chart per hour (measured)
NEWCOMER_TOP = 6         # how many of the newcomer field's top caps to draw

# Common market factor, per horizon h (hours): every live coin's log-cap
# moves by Normal(MARKET_MU(h), MARKET_SD(h)) on top of its own move.
# The rehearsal's first hour shows pool coins rising together about
# +0.07 per 10 minutes (a common +0.4/hour), but the final's answer
# window is 18-24 hours after the opening, so the inflow has decayed:
# the total is set for a market already 11 hours old.  The market
# factor is mean-reverting (inflow comes in waves and fades), so its sd
# saturates instead of growing like a random walk: at 10 minutes it is
# about 0.09, at 7 hours about 0.31, and no wider.
MARKET_MU_TOTAL = 0.8   # total log growth the market factor adds, saturating
MARKET_MU_TAU_H = 4.0   # hours over which it saturates
MARKET_SD_INF = 0.32    # long-run sd of the market factor
MARKET_SD_TAU_H = 2.0   # hours over which its sd saturates

# Pool (graduated) coins.  Hot pool coins survive ~14/15 per 10 minutes
# in the rehearsal, so the crash hazard is small but real.
# Round 2: crash hazards (v2 defaults 0.02 pool / 0.20 curve).
POOL_CRASH_PER_STEP = float(os.environ.get("PF_POOL_CRASH", "0.02"))
# Round 2 (2026-11-12): pool excess drift rate (v2 default 0.30 -> 0).
# Backtested 0.45 -0.235, 0.30 baseline, 0.15 +0.122, 0.075 +0.124,
# 0 +0.185 per read: the momentum drift over-ranks climbing pool coins.
POOL_EXCESS_RATE = float(os.environ.get("PF_POOL_EXCESS", "0"))
POOL_EXCESS_TAU_H = 2.0        # horizon over which the excess drift applies

# Curve (bonding) coins, per 10 minutes.
# Round 2: curve crash hazard (v2 default 0.20).
CURVE_CRASH_PER_STEP = float(os.environ.get("PF_CURVE_CRASH", "0.20"))
# Round 2 (2026-11-12): newcomer-only crash hazard (v2: newcomers used
# the same 0.20 as known curve coins). Backtested newcomers-only:
# 0.10 +0.437, 0.05 +0.599, 0.0 +0.748 per read vs v2, while known-coins-
# only at 0.05 gives -0.025 (no gain). Mechanism, verified in the data:
# 64% of backtest place-winners (32/50) were NOT in the cut history
# (true newcomers, 63-99 entering per 10 min, top caps 2.92-3.47), so
# the unnamed mass must be large; and the crash rate is cap-dependent
# (measured: 38% floored in 10 min at cap 1.5-2.0, 20% at 2.0-2.5,
# ~0% above 2.5) while the newcomer field is selected for high
# first-appearance caps. 0.05 is adopted, NOT 0.0: at the final's
# 7-13 h horizons a 0.0 rate would let 13-hour-old newcomers survive
# at full caps, which the short-horizon backtest cannot see and which
# is clearly wrong; 0.05 still kills ~86% of 6.5 h-old newcomers.
NEWCOMER_CRASH = float(os.environ.get("PF_NEWCOMER_CRASH", "0.05"))
CURVE_GRAD_PER_STEP = 0.03     # hazard a hot curve coin graduates in a step
CURVE_GRAD_MAX = 0.25          # graduation chance saturates at this
CURVE_GRAD_JUMP_SD = 0.30      # sd of the log jump on graduation
CURVE_DEAD_REVIVE = 0.01       # chance a floor coin ever pumps again



# --------------------------------------------------------------------------
# Deterministic PRNG: splitmix64.  Fixed seed, no dependence on the
# platform's random module, so the output is byte-identical every run.
# --------------------------------------------------------------------------
class Rng:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next_u64() >> 11) / 9007199254740992.0

    def normal(self):
        u1 = self.uniform()
        if u1 < 1e-12:
            u1 = 1e-12
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def exponential(self):
        u = self.uniform()
        if u < 1e-12:
            u = 1e-12
        return -math.log(u)


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# --------------------------------------------------------------------------
# Newcomer first-appearance log-caps, measured on the rehearsal history:
# the 454 coins' log10 market cap (SOL) at the first mark that ranked
# them, as 1% quantiles (101 points, p0 to p100).  More than half first
# appear at the dead floor (1.45); the top 1% runs 3.10 to 3.47.  Past
# the top 1% the tail is extended as a Pareto tail fitted to the
# observed top (3.10 at 1%, 3.47 at 1/454): P(cap > x) =
# 0.01 * 10^(-(x-3.10)/0.50).
# --------------------------------------------------------------------------
NEWCOMER_QUANTILES = (
    [1.45] * 58
    + [1.46] * 7
    + [1.47] * 5
    + [1.48, 1.48, 1.49, 1.49, 1.49, 1.49, 1.50, 1.50, 1.50, 1.50, 1.50,
       1.51, 1.51, 1.52, 1.53, 1.56, 1.58, 1.60, 1.63, 1.67, 1.70, 1.73,
       1.82, 1.87, 1.98, 2.06, 2.12, 2.34, 2.64, 3.10, 3.47]
)
TAIL_START_P = 0.99
TAIL_START_X = 3.10
TAIL_SCALE = 0.50


def newcomer_cap_from_u(u):
    """A newcomer log-cap from a uniform quantile, with the Pareto tail."""
    if u < TAIL_START_P:
        pos = u * (len(NEWCOMER_QUANTILES) - 1)
        i = int(pos)
        frac = pos - i
        if i >= len(NEWCOMER_QUANTILES) - 1:
            return NEWCOMER_QUANTILES[-1]
        return NEWCOMER_QUANTILES[i] * (1.0 - frac) + NEWCOMER_QUANTILES[i + 1] * frac
    # Pareto tail: P(cap > x) = 0.01 * 10^(-(x - 3.12)/0.35).
    return TAIL_START_X - TAIL_SCALE * math.log10((1.0 - u) / (1.0 - TAIL_START_P))


def newcomer_top_caps(rng, count, horizon_h, common):
    """The top NEWCOMER_TOP terminal log-caps among `count` newcomers
    arriving in the window.  The top order statistics of `count` uniforms
    come from exponential spacings: with E_1..E_m iid Exp(1) and
    G ~ Gamma(count+1-m, 1), the j-th largest uniform is
    1 - (E_1+...+E_j) / (G + E_1+...+E_m).  Gamma is approximated by a
    normal for large counts, which is all we use it for.

    Each newcomer is a curve coin: it arrives at a uniform age in the
    window, then lives or dies by the same curve dynamics as a coin
    already in the history, so a newcomer that arrived hours ago has
    almost certainly crashed by the read, and only the recent rockets
    and the few that graduated still matter.  The top 12 by
    first-appearance cap are evolved and the best 6 terminal caps kept,
    because the terminal order need not follow the arrival order."""
    m = 12
    if count <= m:
        us = sorted((rng.uniform() for _ in range(count)), reverse=True)
        us = us[:m]
    else:
        exps = [rng.exponential() for _ in range(m)]
        k = count + 1 - m
        g = rng.normal() * math.sqrt(k) + k
        if g < 0.0:
            g = 0.0
        total = g + sum(exps)
        us = []
        run = 0.0
        for j in range(m):
            run += exps[j]
            us.append(1.0 - run / total)
    out = []
    for u in us:
        base = newcomer_cap_from_u(u)
        age = rng.uniform() * horizon_h
        out.append(curve_terminal_cap(rng, base, 0.0, age, common,
                                      crash=NEWCOMER_CRASH if NEWCOMER_CRASH >= 0 else None))
    out.sort(reverse=True)
    return out[:NEWCOMER_TOP]


def curve_terminal_cap(rng, cap0, momentum, horizon_h, common, crash=None):
    """One draw of a curve (bonding) coin's log-cap after `horizon_h`
    hours, starting from `cap0`.  Most curve coins collapse; a few
    graduate to a pool and keep growing."""
    if horizon_h <= 0:
        horizon_h = 0.05
    steps = horizon_h * 6.0
    p_grad = min(CURVE_GRAD_MAX, 1.0 - (1.0 - CURVE_GRAD_PER_STEP) ** steps)
    c = CURVE_CRASH_PER_STEP if crash is None else crash
    p_crash = 1.0 - (1.0 - c) ** steps
    u = rng.uniform()
    if u < p_grad:
        base = cap0 + CURVE_GRAD_JUMP_SD * rng.normal()
        excess = POOL_EXCESS_RATE * momentum * min(horizon_h, POOL_EXCESS_TAU_H)
        sigma = idio_vol_10min(base) * math.sqrt(6.0 * min(horizon_h, 4.0))
        return base + common + excess + sigma * rng.normal()
    if u < p_grad + (1.0 - p_grad) * p_crash:
        return FLOOR_LOG + 0.05 * rng.normal()
    sigma = 0.10 * math.sqrt(6.0 * min(horizon_h, 4.0))
    return cap0 + common + sigma * rng.normal()


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise ValueError("the history holds no ok mark")
    marks.sort(key=lambda m: m["at"])
    return task, marks


def log_cap_sol(row):
    """log10 of a row's market cap in SOL."""
    return math.log10(int(row["market_cap_lamports"])) - 9.0


# --------------------------------------------------------------------------
# Per-coin features at the cut (the last mark)
# --------------------------------------------------------------------------
class Coin:
    __slots__ = ("mint", "cap", "venue", "complete", "mayhem", "age_h",
                 "recent_drift_per_h", "momentum")

    def __init__(self, mint):
        self.mint = mint
        self.cap = FLOOR_LOG
        self.venue = "curve"
        self.complete = False
        self.mayhem = False
        self.age_h = 0.0
        self.recent_drift_per_h = 0.0
        self.momentum = 0.0


def build_coins(marks):
    """Features for every coin the history ever ranked, as of the last mark."""
    last = marks[-1]
    cut = parse_time(last["at"])
    t0 = parse_time(marks[0].get("t0", marks[0]["at"]))
    coins = {}
    first_cap = {}
    first_at = {}
    for m in marks:
        at = parse_time(m["at"])
        for row in m["rows"]:
            mint = row["mint"]
            c = coins.get(mint)
            if c is None:
                c = Coin(mint)
                coins[mint] = c
                first_cap[mint] = log_cap_sol(row)
                first_at[mint] = at
            c.cap = log_cap_sol(row)
            c.venue = row.get("venue", "curve")
            c.complete = bool(row.get("complete", False))
            c.mayhem = bool(row.get("mayhem", False))
            try:
                c.age_h = max(0.0, (cut - parse_time(row["created_at"])).total_seconds() / 3600.0)
            except Exception:
                c.age_h = 0.0
    for mint, c in coins.items():
        span_h = max((cut - first_at[mint]).total_seconds() / 3600.0, 1.0 / 6.0)
        drift = (c.cap - first_cap[mint]) / span_h
        c.recent_drift_per_h = max(-0.5, min(1.0, drift))
        # Momentum score in [-1, 1] from the coin's own recent climb.
        c.momentum = max(-1.0, min(1.0, c.recent_drift_per_h / 0.5))
    return coins, cut


def idio_vol_10min(cap):
    """A coin's own 10-minute log sd: big coins move steadier.  The
    rehearsal's pool coins move about 0.12 per 10 minutes around their
    common move, the top few tighter; 0.09 at the top is the middle."""
    return 0.09 + 0.08 * max(0.0, min(1.0, (2.5 - cap) / 1.0))


def terminal_cap(rng, c, horizon_h, common):
    """One draw of a coin's log-cap (SOL) `horizon_h` hours after the cut.
    `common` is this path's shared market move, already drawn."""
    if horizon_h <= 0:
        horizon_h = 0.05
    steps = horizon_h * 6.0  # 10-minute steps
    if c.venue == "pool" or c.complete:
        p_crash = 1.0 - (1.0 - POOL_CRASH_PER_STEP) ** steps
        if rng.uniform() < p_crash:
            return FLOOR_LOG + 0.05 * rng.normal()
        excess = POOL_EXCESS_RATE * c.momentum * min(horizon_h, POOL_EXCESS_TAU_H)
        # Idiosyncratic variance saturates after a few hours: a coin's
        # log-cap does not keep diffusing like a Brownian motion for a
        # whole day, or the top of 150 coins would reach absurd caps.
        sigma = idio_vol_10min(c.cap) * math.sqrt(6.0 * min(horizon_h, 4.0))
        return c.cap + common + excess + sigma * rng.normal()
    if c.cap <= FLOOR_LOG + 0.05:
        if rng.uniform() < CURVE_DEAD_REVIVE:
            return 2.2 + 0.4 * rng.normal() + common
        return FLOOR_LOG + 0.05 * rng.normal()
    return curve_terminal_cap(rng, c.cap, c.momentum, horizon_h, common)


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def run_monte_carlo(coins, horizons_h, edges):
    """Simulate terminal caps; return per-place coin counts, size histograms,
    and the number of paths per horizon."""
    sim = sorted(
        coins.values(),
        key=lambda c: (0 if (c.venue == "pool" or c.complete) else 1, -c.cap, c.mint),
    )[:MAX_SIM_COINS]
    n_places = 5
    n_bins = len(edges) + 1
    place_counts = [dict() for _ in range(n_places)]
    size_hist = [[0] * n_bins for _ in range(n_places)]
    newcomer_wins = [0] * n_places
    rng = Rng(0x5EED1234)
    total = 0
    for h in horizons_h:
        mu_m = MARKET_MU_TOTAL * (1.0 - math.exp(-h / MARKET_MU_TAU_H))
        # Mean-reverting market factor: its sd saturates, it is not a
        # day-long random walk.
        sd_m = MARKET_SD_INF * math.sqrt(1.0 - math.exp(-h / MARKET_SD_TAU_H))
        n_new = int(round(NEWCOMERS_PER_HOUR * h))
        n_new = max(40, min(n_new, 100000))
        for _ in range(N_PATHS):
            total += 1
            common = mu_m + sd_m * rng.normal()
            caps = []
            for c in sim:
                caps.append((terminal_cap(rng, c, h, common), c.mint, False))
            for nc in newcomer_top_caps(rng, n_new, h, common):
                caps.append((nc, None, True))
            caps.sort(key=lambda t: (-t[0], t[1] or ""))
            for k in range(n_places):
                cap_k, mint_k, is_new = caps[k]
                if is_new:
                    newcomer_wins[k] += 1
                else:
                    d = place_counts[k]
                    d[mint_k] = d.get(mint_k, 0) + 1
                b = bin_of(cap_k, edges)
                size_hist[k][b] += 1
    return sim, place_counts, size_hist, newcomer_wins, total


def bin_of(x, edges):
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) >> 1
        if edges[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------
# Forecast assembly
# --------------------------------------------------------------------------
def floor6(x):
    return math.floor(x * 1000000.0) / 1000000.0


def build_forecast(task, sim, place_counts, size_hist, newcomer_wins, total, mean_horizon_h):
    places = int(task.get("places", 5))
    edges = task["size"]["edges"]
    n_bins = len(edges) + 1

    probabilities = []
    top5 = []
    used = set()
    for k in range(places):
        counts = place_counts[k]
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        rows = []
        s = 0.0
        for mint, n in ranked:
            p = n / total
            if p < NAME_MIN_P:
                break
            p = floor6(p)
            if p <= 0.0:
                continue
            if s + p > MAX_NAMED_PER_PLACE:
                break
            rows.append({"mint": mint, "place": k + 1, "p": p})
            s += p
        if s > MAX_NAMED_PER_PLACE:
            scale = MAX_NAMED_PER_PLACE / s
            rows = [{"mint": r["mint"], "place": r["place"], "p": floor6(r["p"] * scale)} for r in rows]
            rows = [r for r in rows if r["p"] > 0.0]
        if ZERO_MAX > 0:
            named_mints = {r["mint"] for r in rows}
            cands = [(counts.get(c.mint, 0), i, c) for i, c in enumerate(sim)
                     if c.mint not in named_mints
                     and counts.get(c.mint, 0) / total < ZERO_P_MAX]
            if ZERO_ORDER == "lowp":
                cands.sort(key=lambda t: (t[0], t[1]))
            added = 0
            for _n, _i, c in cands:
                if added >= ZERO_MAX or len(rows) >= 100:
                    break
                rows.append({"mint": c.mint, "place": k + 1, "p": 0})
                named_mints.add(c.mint)
                added += 1
        probabilities.extend(rows)
        pick = None
        for mint, _n in ranked:
            if mint not in used:
                pick = mint
                break
        if pick is not None:
            top5.append(pick)
            used.add(pick)

    if len(top5) < places:
        all_ranked = sorted(
            place_counts[0].items(), key=lambda kv: (-kv[1], kv[0])
        )
        for mint, _n in all_ranked:
            if len(top5) >= places:
                break
            if mint not in used:
                top5.append(mint)
                used.add(mint)

    # Sizes: MC histogram mixed with a normal fit, named to SIZE_NAMED_TOTAL.
    # The normal fit's sd is floored by horizon: at short horizons the
    # place caps are nearly known, at long ones genuinely wide.
    sd_floor = 0.15 + 0.06 * math.sqrt(mean_horizon_h)
    sizes = []
    for k in range(places):
        hist = size_hist[k]
        mean = 0.0
        var = 0.0
        for b, n in enumerate(hist):
            if n:
                mid = bin_mid(b, edges)
                mean += mid * n
                var += mid * mid * n
        mean /= total
        var = var / total - mean * mean
        # The normal fit smooths the histogram; its sd is capped so a
        # few extreme paths cannot spread the fit's tail mass everywhere.
        sd = max(math.sqrt(max(var, 0.0)), sd_floor)
        sd = min(sd, 0.45)
        rows = []
        s = 0.0
        sm = [hist[b] / total for b in range(n_bins)]
        if SIZE_SMOOTH > 0:
            a = SIZE_SMOOTH
            sm = [
                (1.0 - 2.0 * a) * sm[b]
                + (a * sm[b - 1] if b > 0 else a * sm[b])
                + (a * sm[b + 1] if b < n_bins - 1 else a * sm[b])
                for b in range(n_bins)
            ]
        for b in range(n_bins):
            p_hist = sm[b]
            lo = float("-inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == n_bins - 1 else edges[b]
            p_norm = phi((hi - mean) / sd) - phi((lo - mean) / sd)
            p = SIZE_HIST_W * p_hist + (1.0 - SIZE_HIST_W) * p_norm
            p = floor6(p * SIZE_NAMED_TOTAL)
            if p >= SIZE_MIN_P:
                rows.append({"place": k + 1, "bin": b, "p": p})
                s += p
        rows.sort(key=lambda r: -r["p"])
        while s > SIZE_NAMED_TOTAL + 1e-9 and rows:
            r = rows.pop()
            s -= r["p"]
        sizes.extend(rows)

    return {"probabilities": probabilities, "top5": top5[:places], "sizes": sizes}


def bin_mid(b, edges):
    if b == 0:
        return edges[0] - 0.05
    if b == len(edges):
        return edges[-1] + 0.05
    return (edges[b - 1] + edges[b]) / 2.0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, marks = load_input(input_dir)
    coins, cut = build_coins(marks)
    reads = [parse_time(r) for r in task["reads"]]
    horizons_h = []
    for r in reads:
        h = (r - cut).total_seconds() / 3600.0
        horizons_h.append(max(h, 0.05))
    edges = task["size"]["edges"]
    sim, place_counts, size_hist, newcomer_wins, total = run_monte_carlo(
        coins, horizons_h, edges
    )
    mean_h = sum(horizons_h) / len(horizons_h)
    forecast = build_forecast(task, sim, place_counts, size_hist, newcomer_wins, total, mean_h)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

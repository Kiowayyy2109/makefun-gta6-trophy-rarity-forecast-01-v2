"""Trophy earn-rate forecaster for GTA VI (Exophase, PS5).

Approach in one paragraph: a printed earn rate is a ratio, earners over
players tracked, and the two sides of that ratio move for different
reasons.  Players tracked grows fast after launch and then slows;
earners for a given trophy grow as the tracked crowd plays deeper, and
also slow.  This program therefore forecasts the two sides separately
and divides at the end.  Working with earners instead of rates is what
keeps the forecast calm when Exophase refreshes its counts: a refresh
lifts the player count in one step, every printed rate drops at once,
yet the earner counts underneath barely move, so a model built on
earners does not mistake a bookkeeping jump for players un-earning
trophies.

Everything is deterministic, standard library only, and offline.
"""

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

Z_FOR_LEVEL = [-1.645, -0.674, 0.0, 0.674, 1.645]
PCT_LO = 0.01
PCT_HI = 99.99
TIME_SCALE_H = 24.0


# ---------------------------------------------------------------- basics

def pct_clamp(x):
    if x is None:
        return PCT_LO
    if x < PCT_LO:
        return PCT_LO
    if x > PCT_HI:
        return PCT_HI
    return x


def to_logit(pct):
    p = pct_clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def parse_stamp(text):
    """ISO-8601 stamp -> epoch seconds (float).  None when unparsable."""
    if not text or not isinstance(text, str):
        return None
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        from datetime import datetime

        moment = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if moment.tzinfo is None:
        from datetime import timezone

        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def solve_square(matrix, rhs):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        head = rows[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = rows[r][col] / head
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def wls(basis_rows, targets, weights, ridge=1e-6):
    """Weighted least squares for a small dense design; None on failure."""
    if not basis_rows:
        return None
    width = len(basis_rows[0])
    gram = [[0.0] * width for _ in range(width)]
    vec = [0.0] * width
    for row, y, w in zip(basis_rows, targets, weights):
        if w <= 0:
            continue
        for i in range(width):
            vec[i] += w * row[i] * y
            for j in range(width):
                gram[i][j] += w * row[i] * row[j]
    for i in range(width):
        gram[i][i] += ridge
    return solve_square(gram, vec)


def apply(coefs, row):
    total = 0.0
    for c, x in zip(coefs, row):
        total += c * x
    return total


def recency_weights(ages_min, span_min):
    """Newer reads weigh more; the oldest still keeps a floor weight."""
    scale = max(span_min * 0.7, 60.0)
    out = []
    for age in ages_min:
        w = math.exp(-age / scale)
        out.append(max(w, 0.12))
    return out


# ------------------------------------------------------------- players

def forecast_players(players, u_values, weights, u_target, last_u):
    """Decelerating growth fit for the tracked-player count.

    Two views are combined: a line for log(players) against log-time
    (multiplicative growth that slows) and a line for players itself
    against log-time (additive growth that slows).  Their geometric
    blend is used, guarded so the forecast never drops below the last
    count and never grows by an implausible factor over the horizon.
    """
    last_p = players[-1]
    if len(players) < 2 or max(players) - min(players) <= 0:
        return float(last_p)
    log_rows = [[1.0, u] for u in u_values]
    log_targets = [math.log(max(p, 1.0)) for p in players]
    log_coef = wls(log_rows, log_targets, weights, ridge=1e-5)
    lin_coef = wls(log_rows, [float(p) for p in players], weights, ridge=1e-5)
    guesses = []
    if log_coef is not None:
        guesses.append(math.exp(apply(log_coef, [1.0, u_target])))
    if lin_coef is not None:
        guesses.append(apply(lin_coef, [1.0, u_target]))
    guesses = [g for g in guesses if g > 0 and math.isfinite(g)]
    if not guesses:
        return float(last_p)
    if len(guesses) == 2:
        blended = math.sqrt(guesses[0] * guesses[1])
    else:
        blended = guesses[0]
    horizon_days = max((u_target - last_u), 0.0) * 2.0 + 1.0
    ceiling = last_p * (2.0 + 2.0 * horizon_days)
    if blended < last_p:
        blended = float(last_p)
    if blended > ceiling:
        blended = ceiling
    return blended


# ------------------------------------------------------------ per trophy

class TrophySeries:
    """One trophy's cleaned observations plus its first-pass fits."""

    def __init__(self, meta):
        self.tid = meta.get("id")
        self.tier = meta.get("tier") or "bronze"
        self.times = []      # minutes since the first read
        self.rates = []      # percent, as printed (may repeat)
        self.earners = []    # rate/100 * players at that read
        self.players = []
        self.logit_slope = 0.0
        self.logit_anchor = 0.0
        self.resid_sd = None
        self.confidence = 0.0
        self.last_rate = 0.0
        self.last_earners = 0.0
        self.earner_guess = None

    def finish_first_pass(self, u_values, weights, u_target, p_target, last_p):
        n = len(self.rates)
        if n == 0:
            self.logit_anchor = to_logit(PCT_LO)
            return
        self.last_rate = self.rates[-1]
        # Smoothed tail earners: average of the last few reads, newest
        # first, so one odd final read cannot drag the anchor around.
        tail_e = list(reversed(self.earners[-3:]))
        tail_w = [1.0, 0.6, 0.35][: len(tail_e)]
        self.last_earners = sum(e * w for e, w in zip(tail_e, tail_w)) / sum(tail_w)

        # --- logit trend of the rate against log-time -------------------
        y = [to_logit(r) for r in self.rates]
        self.logit_anchor = y[-1]
        if n >= 2 and len(set(u_values)) > 1:
            rows = [[1.0, u] for u in u_values]
            coef = wls(rows, y, weights, ridge=1e-4)
            if coef is not None:
                self.logit_anchor = apply(coef, [1.0, u_values[-1]])
                self.logit_slope = coef[1]
                resid = [yy - apply(coef, row) for yy, row in zip(y, rows)]
                if n >= 3:
                    var = sum(w * r * r for w, r in zip(weights, resid)) / max(sum(weights), 1e-9)
                    self.resid_sd = math.sqrt(max(var, 0.0))
                distinct = len(set(round(v, 4) for v in y))
                spread = self.resid_sd if self.resid_sd is not None else 0.4
                self.confidence = min(float(n), 24.0) * (0.4 + 0.6 * distinct / max(n, 1)) / (1.0 + 2.5 * spread)

        # --- earners: saturating time view ------------------------------
        e_time = None
        if n >= 2 and len(set(u_values)) > 1:
            rows = [[1.0, u] for u in u_values]
            coef = wls(rows, self.earners, weights, ridge=1e-3)
            if coef is not None:
                e_time = apply(coef, [1.0, u_target])
        # --- earners: against the player count --------------------------
        e_players = None
        p_span = max(self.players) - min(self.players) if self.players else 0.0
        scale_p = last_p if last_p > 0 else 1.0
        if n >= 2 and p_span > max(2.0, 0.02 * scale_p):
            rows = [[1.0, p / scale_p] for p in self.players]
            coef = wls(rows, self.earners, weights, ridge=1e-3)
            if coef is not None:
                e_players = apply(coef, [1.0, p_target / scale_p])
        if e_players is None:
            e_players = self.last_earners * (p_target / scale_p if scale_p else 1.0)
        if e_time is None:
            e_time = e_players
        guess = 0.5 * e_time + 0.5 * e_players
        # Earners cannot fall far in truth; refresh noise can fake a fall,
        # so keep at most a small give-back against the smoothed tail.
        floor_e = 0.82 * self.last_earners
        if guess < floor_e:
            guess = floor_e
        if guess < 0.0:
            guess = 0.0
        if guess > p_target:
            guess = p_target
        self.earner_guess = guess


# ----------------------------------------------------------------- case

def forecast_case(case):
    trophies_meta = case.get("trophies") or []
    history = case.get("history") or []

    # Clean the read list: parse stamps, sort, drop duplicates (last wins).
    reads = []
    for entry in history:
        stamp = parse_stamp(entry.get("at") or entry.get("slot"))
        if stamp is None:
            continue
        reads.append((stamp, entry))
    reads.sort(key=lambda pair: pair[0])
    deduped = []
    for stamp, entry in reads:
        if deduped and deduped[-1][0] == stamp:
            deduped[-1] = (stamp, entry)
        else:
            deduped.append((stamp, entry))
    reads = deduped

    horizon_min = float(case.get("horizon_minutes") or 0.0)
    checkpoint = parse_stamp(case.get("at"))
    target_stamp = parse_stamp(case.get("target"))

    if not reads:
        rows = []
        for meta in trophies_meta:
            rows.append({"id": meta.get("id"), "q": [1.0, 1.0, 1.0, 1.0, 1.0]})
        return {"trophies": rows}

    t_first = reads[0][0]
    t_last = reads[-1][0]
    if checkpoint is None:
        checkpoint = t_last
    if target_stamp is None:
        target_stamp = checkpoint + horizon_min * 60.0
    if horizon_min <= 0:
        horizon_min = max((target_stamp - checkpoint) / 60.0, 1.0)
    span_min = max((t_last - t_first) / 60.0, 1.0)
    horizon_days = horizon_min / 1440.0

    minutes = [(stamp - t_first) / 60.0 for stamp, _ in reads]
    ages = [(t_last - stamp) / 60.0 for stamp, _ in reads]
    weights = recency_weights(ages, span_min)

    def u_of(minute_value):
        return math.log1p((minute_value / 60.0) / TIME_SCALE_H)

    u_values = [u_of(m) for m in minutes]
    u_last = u_values[-1]
    u_target = u_of((target_stamp - t_first) / 60.0)

    players = []
    for _, entry in reads:
        p = entry.get("players_tracked")
        players.append(float(p) if p else (players[-1] if players else 1.0))
    p_target = forecast_players(players, u_values, weights, u_target, u_last)
    last_p = players[-1]
    # The earners fit and the players fit are two separate extrapolations;
    # dividing one by the other at full strength doubles the extrapolation
    # error, and a one-step refresh jump in the player count would dilute
    # every rate at once.  So the player count used to turn earners back
    # into a rate is damped geometrically halfway toward the last count,
    # and the pooled logit trend (below) carries the rest of the move.
    p_for_rate = math.sqrt(p_target * last_p) if last_p > 0 else p_target

    # Build one series per trophy, aligned by position in the rates list.
    series = []
    for idx, meta in enumerate(trophies_meta):
        ts = TrophySeries(meta)
        for k, (_, entry) in enumerate(reads):
            rates = entry.get("rates") or []
            if idx >= len(rates):
                continue
            rate = rates[idx]
            if rate is None:
                continue
            ts.times.append(minutes[k])
            ts.rates.append(float(rate))
            ts.players.append(players[k])
            ts.earners.append(float(rate) / 100.0 * players[k])
        series.append(ts)

    # Recompute per-trophy u values and weights from kept times.
    for ts in series:
        if not ts.times:
            ts.finish_first_pass([], [], u_target, p_target, last_p)
            continue
        u_ts = [u_of(m) for m in ts.times]
        age_ts = [minutes[-1] - m for m in ts.times]
        w_ts = recency_weights(age_ts, span_min)
        ts.finish_first_pass(u_ts, w_ts, u_target, p_target, last_p)

    # ------------------------------------------------ pooling of slopes
    def weighted_mean(pairs):
        tot_w = sum(w for _, w in pairs)
        if tot_w <= 0:
            return 0.0
        return sum(v * w for v, w in pairs) / tot_w

    global_slope = weighted_mean([(ts.logit_slope, ts.confidence) for ts in series if ts.confidence > 0])
    tier_slope = {}
    tier_conf = {}
    for name in ("platinum", "gold", "silver", "bronze"):
        members = [ts for ts in series if ts.tier == name and ts.confidence > 0]
        if members:
            tier_slope[name] = weighted_mean([(m.logit_slope, m.confidence) for m in members])
            tier_conf[name] = sum(m.confidence for m in members)
    for name in list(tier_slope):
        c = tier_conf[name]
        tier_slope[name] = (c * tier_slope[name] + 2.0 * global_slope) / (c + 2.0)

    # ------------------------------------------------ pooled residual sd
    def pooled_sd(members):
        vals = [(m.resid_sd, m.confidence) for m in members if m.resid_sd is not None]
        if not vals:
            return None
        tot = sum(w for _, w in vals)
        if tot <= 0:
            return None
        return math.sqrt(sum(w * v * v for v, w in vals) / tot)

    global_sd = pooled_sd(series)
    if global_sd is None:
        global_sd = 0.4
    tier_sd = {}
    for name in ("platinum", "gold", "silver", "bronze"):
        sd = pooled_sd([ts for ts in series if ts.tier == name])
        tier_sd[name] = sd if sd is not None else global_sd

    horizon_scale = math.sqrt(1.0 + horizon_min / max(span_min, 30.0))
    walk_part = 0.5 * math.sqrt(max(horizon_days, 0.0))

    rows = []
    for ts in series:
        # Shrunk slope: own estimate, pulled to tier, then to global.
        own = ts.logit_slope
        tier_part = tier_slope.get(ts.tier, global_slope)
        c = ts.confidence
        slope = (c * own + 3.0 * tier_part + 1.5 * global_slope) / (c + 4.5)

        mu_trend = ts.logit_anchor + slope * (u_target - u_last)

        if ts.earner_guess is not None and p_for_rate > 0:
            rate_e = 100.0 * ts.earner_guess / p_for_rate
            mu_earn = to_logit(rate_e)
            mu = 0.5 * mu_earn + 0.5 * mu_trend
        else:
            mu = mu_trend

        # Spread: own residuals blended with tier and global, horizon-scaled,
        # plus a random-walk term so a perfectly flat history still admits
        # that a refresh can move every rate before the target.
        if ts.resid_sd is not None:
            base_var = 0.5 * ts.resid_sd ** 2 + 0.3 * tier_sd.get(ts.tier, global_sd) ** 2 + 0.2 * global_sd ** 2
        else:
            base_var = 0.6 * tier_sd.get(ts.tier, global_sd) ** 2 + 0.4 * global_sd ** 2
        sigma = math.sqrt(base_var * horizon_scale * horizon_scale + walk_part * walk_part + 0.12 * 0.12)
        if sigma < 0.26:
            sigma = 0.26
        if sigma > 2.8:
            sigma = 2.8

        qs = []
        for z in Z_FOR_LEVEL:
            value = from_logit(mu + z * sigma)
            if value < 0.0:
                value = 0.0
            if value > 100.0:
                value = 100.0
            qs.append(round(value, 4))
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": ts.tid, "q": qs, "_tier": ts.tier})

    # ------------------------------------------------ platinum ceiling
    plat_rows = [r for r in rows if r["_tier"] == "platinum"]
    other_rows = [r for r in rows if r["_tier"] != "platinum"]
    if plat_rows and other_rows:
        for level in range(5):
            ceiling = min(r["q"][level] for r in other_rows)
            for r in plat_rows:
                if r["q"][level] > ceiling:
                    r["q"][level] = ceiling
        for r in plat_rows:
            for k in range(1, 5):
                if r["q"][k] < r["q"][k - 1]:
                    r["q"][k] = r["q"][k - 1]

    clean_rows = [{"id": r["id"], "q": r["q"]} for r in rows]
    return {"trophies": clean_rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not IN_DIR.is_dir():
        return
    for path in sorted(IN_DIR.glob("*.in")):
        with path.open("r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        target = OUT_DIR / (path.stem + ".out")
        target.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

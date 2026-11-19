# makefun-gta6-trophy-rarity-forecast-01-v2

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kiowayyy2109/makefun-gta6-trophy-rarity-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kiowayyy2109/makefun-gta6-trophy-rarity-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

## Methodology

A printed earn rate is a ratio: earners divided by players tracked. The two sides move for different reasons, so this forecaster models them separately and only divides at the end.

- **Players tracked.** The tracked-player count is forecast with a decelerating, log-time growth fit. Two views — a line for log(players) against log-time and a line for players itself against log-time — are blended geometrically, guarded so the forecast never falls below the last count and never grows by an implausible factor over the horizon.
- **Earners per trophy.** Each trophy's earner count (rate × players ÷ 100) is forecast two ways: a saturating log-time fit of earners against time, and a fit of earners against the player count itself, evaluated at the forecast player count. The two are averaged, floored so earners cannot fall far (players do not un-earn trophies; apparent falls are refresh noise), and capped at the forecast player count. Because the earners fit and the players fit are two separate extrapolations, dividing one by the other at full strength would double the extrapolation error — and a one-step refresh jump in the player count would dilute every rate at once — so the player count used for this division is damped geometrically halfway toward the last observed count, and the pooled logit trend below carries the rest of the move. The earners-implied rate is 100 × forecast earners ÷ damped forecast players.
- **Why earners, not rates.** When Exophase refreshes its counts, the player total jumps in one step and every printed rate drops at once, even though nobody lost a trophy. Earner counts barely move through such a refresh, so modelling earners keeps a bookkeeping jump from poisoning the trend. Recent reads are also weighted more heavily than old ones, so a single level shift has limited leverage.
- **Logit trend and hierarchical pooling.** Alongside the earners view, each trophy gets a weighted least-squares trend of its logit rate against log-time. That slope is shrunk toward the mean slope of its tier group (platinum / gold / silver / bronze), and the tier mean is itself shrunk toward the global mean. Shrinkage strength depends on how much usable history the trophy has and how dispersed its residuals are: quiet, well-observed trophies keep their own trend; noisy ones borrow their tier's. The final location blends the earners-implied logit with the pooled-trend logit.
- **Spread calibration.** Each trophy's predictive spread on the logit scale comes from its residual dispersion around the fit, blended with its tier's and the global pooled dispersion, then scaled up with the forecast horizon relative to the history span, plus a random-walk term (0.5 × √(horizon in days)) so that even a perfectly flat history admits that a refresh can move every rate before the target. The spread is floored and capped, and the five quantiles are the inverse-logit of location + z·σ for z = −1.645, −0.674, 0, 0.674, 1.645.
- **Structural cap.** Nobody holds the platinum trophy without holding every other trophy, so after all quantiles are computed the platinum row is capped, level by level, at the minimum quantile across all other trophies at that level. Quantile order is re-checked afterwards.
- **Edge cases.** Rates of 0.00 and 100.00 are handled through clamped logits; a single-read history falls back to that read with a pooled spread; missing rates and duplicate timestamps are cleaned before fitting. The program is deterministic, uses only the Python standard library, and needs no network.

## Run it

The program reads every `*.in` case file in the input directory and writes one `<id>.out` per case to the output directory:

```sh
python3 main.py
```

By default it reads `/input` and writes `/output`. Locally, point it at other directories with the environment variables:

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each output file is UTF-8 JSON of the form `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}` — one row per trophy in the case's last read, quantiles non-decreasing, each between 0 and 100.

<sub>Entered on [make.fun](https://make.fun/?ref=8b9bbsfs&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

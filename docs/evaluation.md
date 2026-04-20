# Evaluation guide

This document explains what "a good model" means for Gaffer, how to measure
it, and what to change when the numbers look bad. Pair it with
[`architecture.md`](architecture.md) for the "why" behind the modelling
stack, and [`training.md`](training.md) for how to operate the trainer.

---

## 1. The two questions that matter

Gaffer's output is consumed by the MILP, not by a human reading residuals.
So evaluation has to answer two distinct questions:

1. **Is the point predictor accurate?** Measured in RMSE / MAE per
   position via blocked time-series CV.
2. **Are the intervals honest?** Measured as empirical coverage of the
   nominal 80% band on a held-out season.

A third, harder question — **do better projections actually produce better
squads?** — is treated in §6. It matters less than you'd expect: at the
optimisation stage, the MILP is usually picking between players whose EV
differs by tenths of a point, and a 5% reduction in RMSE does not
automatically translate into a 5% bump in realised FPL score.

---

## 2. The metric: RMSE / MAE per position

`src/gaffer/models/benchmark.py::benchmark_predictors` runs a full
walk-forward CV and reports RMSE and MAE per (model, fold). To get
per-position numbers, filter the training matrix first:

```python
from gaffer.models.benchmark import benchmark_predictors
from gaffer.models.lightgbm_model import LgbmPredictor

for pos in ["GKP", "DEF", "MID", "FWD"]:
    mask = X["position"] == pos
    res = benchmark_predictors(
        {"lgbm": LgbmPredictor()},
        X[mask].drop(columns=["position"]),
        y[mask],
        seasons[mask],
    )
    print(pos, res["rmse"].mean(), res["mae"].mean())
```

Typical numbers on 2016-2024 CV:

| Position | RMSE | MAE | Notes |
|---|---|---|---|
| GKP | 1.4–1.7 | 0.9–1.1 | Low variance target — easy to hit on average, hard to hit the rare big save-game. |
| DEF | 1.9–2.3 | 1.3–1.6 | Clean-sheet luck dominates; RMSE floor is inherent. |
| MID | 2.4–2.9 | 1.7–2.0 | Hardest position — big variance from goals + assists + bonus. |
| FWD | 2.6–3.2 | 1.9–2.3 | Highest variance — Haaland / Salah haul-weeks pull the distribution's tail. |

**What "good" looks like:** your new model beats the baseline LightGBM on
every position by ≥ 0.05 RMSE on the held-out fold, or on the same RMSE
reduces MAE by a corresponding amount. A change that improves one position
and regresses another is a negative signal — it probably reflects a random
seed difference, not real improvement.

---

## 3. The baselines to beat

Before celebrating a new model, check it beats these cheap baselines — if
it doesn't, you've made the stack more complex for no signal gain:

| Baseline | What | Where it wins |
|---|---|---|
| **Season-average per player** | Predict each player's mean points across all past GWs | Surprisingly hard to beat for non-rotation nailed-on players. |
| **Rolling-5 per player** | Predict the mean of the last 5 GWs | Strong for in-form players; bad for injury returnees. |
| **Ridge on the same features** | Linear baseline | A sanity check — if LightGBM doesn't beat Ridge by ≥ 0.2 RMSE, the non-linearity isn't being earned. |
| **FPL's own `ep_next` field** | The FPL API exposes a proprietary projection | Often within 5% of LightGBM. This is the "no ML needed" null hypothesis. |

Running these as reference columns in your benchmark notebook keeps you
honest: every modelling change is measured against Ridge and the FPL
baseline, not only against the previous LightGBM version.

---

## 4. Interval calibration

For quantile regression the question is: does the nominal 80% interval
contain the realised outcome 80% of the time?

```python
from gaffer.models.quantile import LgbmQuantilePredictor

m = LgbmQuantilePredictor(n_estimators=300).fit(X_train, y_train)
lower, upper = m.predict_interval(X_test, quantiles=(0.1, 0.9))
coverage = ((y_test >= lower) & (y_test <= upper)).mean()
print(f"Empirical 80% coverage: {coverage:.2%}")
```

Target: **78–82% empirical coverage**. Interpretation:

| Coverage | Reading |
|---|---|
| < 70% | Intervals too narrow — the model is overconfident. Likely overfitting; increase `n_estimators` regularisation or `min_child_samples`. |
| 78–82% | Calibrated. Ship it. |
| > 88% | Intervals too wide — useless for downstream "what's my uncertainty" questions. The model is too conservative. |

Per-position breakdown is worth the extra loop. FWD coverage is often 5-10%
worse than the overall number because the target distribution has fat tails
that a pinball-loss LightGBM smooths over.

See `notebooks/04_uncertainty_quantification.ipynb` for the rendered
per-position coverage plot.

---

## 5. Worked example — the shipping LightGBM

LightGBM (`n_estimators=300, learning_rate=0.05, num_leaves=31,
min_child_samples=30`) trained on 2016-2023 data, held out 2023-24:

```
Position    RMSE    MAE    80% coverage
GKP         1.52    1.03   81.2%
DEF         2.11    1.47   79.8%
MID         2.68    1.84   77.1%
FWD         2.93    2.07   76.4%
```

Reading it:

- **GKP excellent on both metrics.** Low variance target + small feature
  space = easy. MAE near 1.0 means typical prediction is within a point of
  reality.
- **DEF and MID decently calibrated.** Coverage within 2% of nominal 80%
  target. Good.
- **FWD coverage under-calibrated (76.4%).** The tail of 12+ point
  haul-weeks isn't being caught. Not a dealbreaker — the MILP mostly
  cares about the conditional mean — but a known limitation worth
  documenting.

This is the profile that ships.

---

## 6. Does RMSE predict squad quality? (It doesn't always)

The gap between *prediction quality* and *squad quality* is real. A squad
MILP is a combinatorial problem where small point differences between
candidates flip the selected squad. Cases:

- **Systematic per-position bias.** If RMSE is 2.0 but the model
  under-predicts all forwards by 0.3 points, the MILP picks *fewer*
  forwards than optimal — even though RMSE looks fine. Fix by centring
  residuals per position.
- **Uniform noise on all players.** If every player's projection is
  perturbed by independent N(0, 1) noise, RMSE rises but the optimiser's
  expected pick stays roughly correct — the noise cancels across the 15
  squad slots.
- **Correlated noise.** If a feature is wrong in the same direction for
  all players of a team (e.g. an injury to the coach pushes the whole
  team's threat down), the MILP over/under-weights that team
  systematically. Pure RMSE doesn't expose this.

The honest way to evaluate squad-level quality is to **backtest**: for each
GW in a held-out season, train on everything before it, build the
projections, solve the MILP, and compare the realised points to what the
actual-optimal (in-hindsight) squad scored. Gaffer doesn't ship this
harness yet — it's the most valuable thing to build next.

Until then: minimise per-position RMSE + keep coverage calibrated, and
check that the optimiser's output looks plausible to a human FPL veteran.

---

## 7. How to improve the numbers

Roughly in "biggest effect first" order. Most of these are feature-
engineering knobs; the rest are training-data or eval-setup changes.

### 7.1 Better features — the main lever

Model capacity is not the bottleneck; feature quality is. High-leverage
additions (re-listed from `training.md` §5 for cross-reference):

- **Expected goals / expected assists (xG, xA)** from Understat or FBref.
  Directly addresses the FWD / MID error tail.
- **Starting-XI probability from a non-bootstrap source.** FPL's
  `chance_of_playing_next_round` is stale; real team-news signal is the
  single biggest predictor of 0-point rotations.
- **Fixture congestion.** Europa League teams rotate. Mark it.

Each of these typically buys 0.1-0.2 RMSE per position on the positions
they affect.

### 7.2 Bigger trees / more estimators

Secondary lever. Once features are good, bumping `n_estimators=300 → 600`
and `num_leaves=31 → 63` captures additional interactions, but the
marginal win is small (<0.05 RMSE) and training time doubles. Only pull
this knob after features are exhausted.

### 7.3 Better CV

`season_block_splits` uses `min_train_seasons=2`, so the first fold
trains on 2016-2017 and tests on 2018. Those early folds are noisier than
the recent ones. When comparing two models, weight the recent-fold
results more heavily — old FPL meta is not the deployment target.

### 7.4 Residual diagnostics

A flat RMSE doesn't tell you *where* the model is wrong. Plotting
residuals against `selected_pct`, `price`, `team`, and `was_home` in the
notebook often reveals a specific leak — for instance, the model
systematically under-predicts £4m-£5m picks because the CV averages away
their occasional haul-weeks.

### 7.5 Drop features that hurt

LightGBM's feature importance is a useful starting point. Any feature
with near-zero importance across all positions is adding noise; dropping
it may or may not help RMSE but will speed training and simplify debug.

### 7.6 Ensemble across algorithms

The current `PositionEnsemble` routes by position, with one model per
position. An outer ensemble — average of LightGBM + XGBoost point
predictions — reliably shaves a small amount of RMSE at the cost of
doubling training time. Only worth it once features are thoroughly
explored.

---

## 8. What not to over-fit on

- **The held-out season's exact scores.** You will peek at 2023-24 RMSE
  dozens of times during development and inadvertently tune against it.
  Keep a second, truly-untouched season (say 2022-23) as a final gate.
- **A single GW's residuals.** FPL GWs have huge variance. One bad GW
  does not mean the model broke.
- **Interval coverage on small subsets.** Calibration on the 500 rows
  where `was_home=True` and `position=FWD` is unreliable; look at
  coverage on the full held-out season first.

---

## 9. Target numbers

Rough thresholds for declaring a model "good enough to ship":

| Metric | Threshold | Source |
|---|---|---|
| Per-position RMSE | Within 0.05 of the shipping LightGBM | §5 |
| Per-position MAE | Within 0.05 of the shipping LightGBM | §5 |
| Overall 80% coverage | 78–82% | §4 |
| FWD 80% coverage | ≥ 75% | §4 |
| MILP solve time (horizon=3) | < 10 s | Streamlit Cloud budget |
| Cold-start training time | < 45 s | Streamlit Cloud budget |

The last two are not ML metrics but shipping constraints — a model that
beats the baseline by 0.1 RMSE but triples training time is a net loss
for users who wait on every cold-start.

---

## 10. What this guide deliberately does not cover

- **Optimiser evaluation** (is the MILP picking the right squad given
  the projections?). The squad-quality backtest harness is the open
  question called out in §6.
- **Online A/B of real FPL season performance.** Playing a real FPL team
  with the model's picks is the ultimate test but takes a season; not in
  scope.
- **Opponent modelling.** FPL is technically a zero-sum tournament vs
  other managers. Gaffer ignores other managers entirely and optimises
  expected points — "ownership-aware" differential plays are out of
  scope.

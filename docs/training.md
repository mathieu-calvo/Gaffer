# Training guide

This document explains how Gaffer's per-position ensemble is trained, what
knobs exist, and what to change when the numbers look bad. Pair it with
[`architecture.md`](architecture.md) §3 for the "why" behind the modelling
choices, and [`evaluation.md`](evaluation.md) for the measurement loop that
tells you whether a training change was worth it.

---

## 1. The command

There is no standalone training CLI — training is wired directly into the
app's cold-start via `train_or_load_ensembles`. To retrain explicitly:

```python
from gaffer.providers.fpl_api import LiveFplApiProvider
from gaffer.providers.historical_csv import HistoricalCsvProvider
from gaffer.services.model_cache import train_or_load_ensembles

fpl = LiveFplApiProvider()
hist = HistoricalCsvProvider()
point, quantile = train_or_load_ensembles(fpl, hist, force_retrain=True)
```

Or from the Streamlit app: **sidebar → "Force retrain models"** on the
Optimal Squad page. The joblib cache at `cache_dir/models/` is cleared and
both ensembles are retrained from scratch.

Source: `src/gaffer/services/model_cache.py` (train-or-load helper) +
`src/gaffer/services/prediction_service.py::build_training_set`
(feature matrix assembly) + `src/gaffer/models/ensemble.py`
(`PositionEnsemble.fit`).

---

## 2. What one training run does

1. **Assemble the training matrix.** `build_training_set` concatenates
   8 seasons of historical CSV with the current season's FPL-API history,
   applies `feature_engineer` (row-level reformatting) + `compute_rolling`
   (EWMA with T+1 shift), joins rolling features back onto per-GW fixture
   context, and produces `(X, y, seasons)` — `X` the feature matrix, `y`
   the realised FPL points, `seasons` the CV grouping key.
2. **Drop leakage rows.** First-gameweek rows for every player have
   all-NaN rolling features (nothing to average over yet). These are
   dropped: `X = td.X.dropna()`.
3. **Fit the point ensemble.** `PositionEnsemble(factory=LgbmPredictor)`
   splits `X` by `position` and trains four separate LightGBM regressors
   with squared-error loss.
4. **Fit the quantile ensemble.** Same matrix, same per-position routing,
   but `LgbmQuantilePredictor` (pinball loss at α=0.1, 0.5, 0.9).
5. **Persist both.** `joblib.dump` writes to
   `cache_dir/models/ensemble_{point,quantile}.joblib`.

Total cost on CPU with 8 seasons: **~30 seconds**. Both ensembles train in
series rather than parallel because LightGBM already saturates the
available cores via its own OpenMP runtime.

---

## 3. Knobs and what they do

| Knob | Where | Default | What it controls |
|---|---|---|---|
| `GAFFER_EWMA_ALPHA` | env / `.env` | 0.95 | Rolling-average decay. Higher = more weight on recent matches. |
| `LgbmPredictor(n_estimators=...)` | `model_cache.py::_default_factory` | 300 | Boosting rounds per leaf. |
| `LgbmPredictor(learning_rate=...)` | same | 0.05 | Step size. |
| `LgbmPredictor(num_leaves=...)` | LgbmPredictor default | 31 | Tree complexity. |
| `LgbmPredictor(min_child_samples=...)` | same | 30 | Regularisation against leaf-overfit. |
| `LgbmQuantilePredictor(quantiles=...)` | `quantile.py` | (0.1, 0.5, 0.9) | Which quantiles to fit. |
| `force_retrain` | `train_or_load_ensembles` arg | False | Skip the joblib cache. |

**Why you rarely need to change these.** LightGBM's defaults are
well-tuned out of the box and the dominant error in FPL prediction is
"genuine noise in the target" (bonus points, clean-sheet luck, rotation),
not model capacity. Bigger trees don't help much; getting better features
does.

---

## 4. The trade-off: `n_estimators` vs `min_child_samples`

| | Too few estimators | Too loose `min_child_samples` |
|---|---|---|
| What breaks | Underfits — per-position models don't learn position-specific patterns beyond broad ones | Overfits on small-sample positions (GKP has ~2k rows/season vs MID's ~8k) |
| Symptom | High train + test RMSE, ensemble barely beats a "predict season-average" baseline | Train RMSE plummets, test RMSE rises, coverage of 80% intervals collapses below 70% |
| Fix | Bump `n_estimators` (300 → 600) | Bump `min_child_samples` (30 → 50) |

In practice `(n_estimators=300, min_child_samples=30)` has been the sweet
spot for 8 seasons. If you add 2026-27 data, revisit — the GKP slice will
grow and tolerate less regularisation.

---

## 5. Feature engineering knobs

Features live upstream of the trainer — changing them is the highest-leverage
thing you can do to improve the model.

### `GAFFER_EWMA_ALPHA` (currently 0.95)

`α` is the EWMA decay. `1.0` = no smoothing (only the most recent GW),
`0.0` = uniform across all history. The FPL meta drifts meaningfully over
a season (teams change managers, players get injured, set-piece takers
rotate), so a high α (heavily recency-weighted) matters. Drop to 0.85 if
you want smoother behaviour through small hot streaks.

### Adding a new feature

Any new column in the engineered frame propagates through automatically —
`_ensure_feature_columns` in `prediction_service.py` picks up all numeric /
bool columns that aren't explicit metadata. Recipe:

1. Add the column name to the right list in
   `src/gaffer/features/constants.py` (`PLAYER_STATS_FEATURES`,
   `TEAM_STATS_FEATURES`, `FPL_FEATURES`, etc.).
2. Compute it in `engineering.py` or `rolling.py` as appropriate.
3. Write a test in `tests/unit/test_features.py` — at minimum, row-count
   preserved + T+1 shift intact.
4. Delete `cache_dir/models/*.joblib` and retrain. The ensemble's
   `feature_names_in_` is locked at fit-time, so old cached models will
   ignore new columns until retrained.

### High-leverage features to add (backlog)

- **Expected goals / expected assists (xG, xA).** FPL's own `creativity` /
  `threat` are coarse proxies. Adding Understat or FBref-style xG
  directly closes a big gap on forwards and attacking mids.
- **Starting-XI probability.** FPL's "chance_of_playing_next_round" is
  bootstrap-only and stale. Scraping team-news tweets into a probability
  would materially sharpen the projections for rotation-risk players.
- **Fixture congestion.** Teams in the Europa League mid-week are
  systematically rotated. A binary `has_midweek_fixture` column costs
  nothing and captures a real effect.

---

## 6. Cross-validation — blocked by season

`src/gaffer/models/training.py::season_block_splits` produces walk-forward
splits. To benchmark a model change:

```python
from gaffer.models.benchmark import benchmark_predictors
from gaffer.models.ridge import RidgePredictor
from gaffer.models.xgboost_model import XgbPredictor
from gaffer.models.lightgbm_model import LgbmPredictor

td = build_training_set(fpl, historical)
X = td.X.dropna()
y = td.y.loc[X.index]
seasons = td.seasons.loc[X.index]

candidates = {
    "ridge": RidgePredictor(),
    "xgb":   XgbPredictor(n_estimators=200, learning_rate=0.05),
    "lgbm":  LgbmPredictor(n_estimators=300, learning_rate=0.05),
}
results = benchmark_predictors(candidates, X, y, seasons)
print(results)
```

The output is a tidy frame with `(model, fold, rmse, mae)`. See
`notebooks/03_model_benchmark.ipynb` for the rendered version.

**Why CV by season and not random k-fold.** Player identities recur across
seasons; random k-fold leaks Erling Haaland's 2023-24 rows into the
training set of a fold whose test set is Haaland's 2022-23 rows. The
bench-numbers-great / prod-numbers-mediocre gap is often pure CV
contamination.

---

## 7. Quantile-interval calibration

`LgbmQuantilePredictor` gives intervals at α=0.1 / 0.9 — nominal 80%
coverage. Empirical coverage on a held-out season should be **~78–82%** if
calibrated. Check:

```python
from gaffer.providers.fpl_api import LiveFplApiProvider
from gaffer.providers.historical_csv import HistoricalCsvProvider
from gaffer.services.prediction_service import build_training_set
from gaffer.models.quantile import LgbmQuantilePredictor

td = build_training_set(LiveFplApiProvider(), HistoricalCsvProvider())
X = td.X.dropna()
y = td.y.loc[X.index]
seasons = td.seasons.loc[X.index]

holdout = seasons.max()
train_mask = seasons != holdout
m = LgbmQuantilePredictor(n_estimators=300).fit(X[train_mask], y[train_mask])
lower, upper = m.predict_interval(X[~train_mask], quantiles=(0.1, 0.9))
obs = y[~train_mask]
print(f"Empirical 80% coverage: {((obs >= lower) & (obs <= upper)).mean():.2%}")
```

Rendered in `notebooks/04_uncertainty_quantification.ipynb`.

If coverage is systematically too low (e.g. 65%), the quantile model is
too tight — bump `n_estimators` to slow overfitting, or widen the fitted
quantiles to 0.05 / 0.95 and present those in the UI with a "90%" label.
If coverage is too high (e.g. 92%), the intervals are too wide to be
useful — likely underfit; train longer.

---

## 8. Checkpoint discipline

Models are cached to disk in `cache_dir/models/ensemble_{point,quantile}.joblib`.
The cache has **no version tag**: any change to feature engineering, model
config, or provider code silently reuses the stale cached model.

Rules:

- **Delete the cache after any change to `src/gaffer/features/` or
  `src/gaffer/models/`.** The Streamlit sidebar's "Force retrain models"
  button does this.
- **Commit only source, never the cache.** `cache_dir/` is gitignored.
- **On Streamlit Cloud**, the cache lives on the ephemeral container
  filesystem. Every cold-start retrains; warm requests are fast.

Resuming training (i.e. loading a partially-trained model and training
further) is **not supported** — LightGBM's sklearn wrapper retrains from
scratch each `fit` call. This is fine because a full retrain takes 30 s.

---

## 9. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ValueError: cannot handle a non-unique multi-index` during training | Duplicated `(name, kickoff_date)` from double gameweeks or same-named players across seasons | `build_training_set` deduplicates with `drop_duplicates(["name","kickoff_date"], keep="last")`; if you hit this in a new code path, apply the same dedup before joining |
| `pandas dtypes must be int, float or bool. Fields with bad pandas dtypes: team: str` | String columns reaching LightGBM | Ensure the caller goes through `PositionEnsemble`, which drops non-numeric columns via `_numeric_features` |
| `The number of features in data (N) is not the same as it was in training data (N−1)` | Inference frame has extra columns the cached model never saw | Delete the cache and retrain — `feature_names_in_` is locked at fit-time |
| Empty `adv_buffers` / zero training rows | First-GW rows dropped (correct) but the season has too few GWs to leave anything | Ensure the joined current season has ≥2 GWs before training; otherwise fall back to historical-only |
| All-NaN column in `X` after `dropna` removes every row | A new feature was added but `compute_rolling` doesn't produce it | Verify the column is listed in the right `constants.py` group so rolling picks it up |
| Huge per-position RMSE gap (GKP 1.5, FWD 3.5) | Expected — forwards have inherently higher variance | Not a bug |

---

## 10. Quick sanity check

Before accepting any model change, run the fast tests and the Kuhn-style
CV smoke:

```bash
pytest -m "not slow" -q
.venv/Scripts/python -c "
from gaffer.providers.fpl_api import LiveFplApiProvider
from gaffer.providers.historical_csv import HistoricalCsvProvider
from gaffer.services.model_cache import train_or_load_ensembles
point, quant = train_or_load_ensembles(
    LiveFplApiProvider(), HistoricalCsvProvider(), force_retrain=True
)
print('trained ok')
"
```

If the train succeeds and tests pass, the change is safe to merge. The
final check is empirical: run the full pipeline (train → project →
optimise) end-to-end and confirm the optimiser produces a sensible XI.

---

## 11. What's not covered here

- **Optimising the MILP** — see [`architecture.md`](architecture.md) §5.
  MILP knobs (`bench_weight`, `solver_time_limit`, `horizon`) live in
  config and the UI, not in the training loop.
- **Deploying a retrained model** — there's no model registry. Retrain
  happens on the cloud on next cold-start after a push; for an immediate
  rotation, trigger Streamlit Cloud's "Reboot app".
- **Evaluating a model's impact on optimiser output** — belongs in
  `evaluation.md`. A lower RMSE doesn't always produce better squads
  (see §6 of that doc).

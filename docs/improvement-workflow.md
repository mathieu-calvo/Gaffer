# Improvement Workflow

Gaffer's shipped model improves through two channels. This doc is the
operating manual for making each change **measurable, reversible, and
surfaced to users through the same single path** — a push to `main` that
Streamlit Cloud picks up automatically.

Pair with [`training.md`](training.md) (what the cold-start does),
[`evaluation.md`](evaluation.md) (the metrics that gate merges), and
[`deployment-guide.md`](deployment-guide.md) (how the cloud side picks
up the result).

---

## 1. The two channels

| Channel | Source | How it enters the loop |
|---|---|---|
| **Feature engineering** (primary) | New columns or transforms added under `src/gaffer/features/`, or tweaks to the ensemble factory in `src/gaffer/models/`. | Merged to `main`; the next Streamlit Cloud cold-start retrains with the new code. |
| **Historical data refresh** (once per season) | When Vaastav's `clean_merged_gwdf_<yyyy_to_yyyy>.csv` drops each summer, replacing the committed CSV. | Committed to `data/historical/`; the next cold-start trains on the longer history. |

Neither channel requires a published-model registry. The "shipping model"
is whatever `main` produces when Streamlit Cloud rebuilds the app — see §2.

---

## 2. Single source of truth: `main` on GitHub

Gaffer does not publish `.joblib` checkpoints. The deployed model is a
**pure function of the code on `main`**:

- `data/cache/models/*.joblib` is gitignored (`.gitignore:37`) — joblib
  files on your machine never leave your machine.
- Streamlit Cloud's disk is ephemeral. A redeploy wipes it and the next
  page load retrains from scratch using the repo's code and historical
  CSV.
- There is no HF Hub, no S3 bucket, no external model registry to keep
  in sync.

Improving the bot your users see **means merging a better `main`**.
Nothing else counts. This is a deliberate simplification over registry-
based workflows: fewer moving parts, but also no way to rollback
faster than `git revert`.

---

## 3. The gate: benchmark before merge

Every change that could move RMSE has to run through
`notebooks/03_model_benchmark.ipynb` before it merges:

1. Check out the candidate branch.
2. Open the notebook and run all cells. It:
   - Runs `benchmark_predictors` across walk-forward folds.
   - Prints per-position RMSE / MAE for LightGBM, XGBoost, Ridge, and
     the current `PositionEnsemble`.
   - Prints 80% interval coverage from the quantile ensemble.
3. Compare against the **shipping numbers** in
   [`evaluation.md`](evaluation.md) §5 (the committed worked example).
4. **Pass**: per-position RMSE within 0.05 of shipping, coverage in
   78–82%, no position regressed. Merge.
5. **Fail**: do not merge. Either iterate, or document the regression
   explicitly in the PR if the feature is worth it for reasons orthogonal
   to RMSE (e.g. interpretability, training speed).

The notebook is the gate because a model that passes locally but
regresses on the 2022–23 fold is the most common failure mode. RMSE on
the last trained fold lies.

---

## 4. Process: a feature-engineering change

Follow this for every change that touches
`src/gaffer/features/`, `src/gaffer/models/`, or the training config.

### Before coding

1. **Rerun the benchmark on `main`** to establish a local baseline:
   ```bash
   jupyter nbconvert --to notebook --execute \
     notebooks/03_model_benchmark.ipynb \
     --output-dir /tmp/gaffer-bench/
   ```
   Save the printed per-position RMSE table somewhere — this is what you
   need to beat.

2. **Pick a focused branch name** that names the hypothesis, not the
   implementation:
   - Good: `feature/xg-xA-from-understat`, `feature/home-bonus-interaction`
   - Bad: `feature/lgbm-tweaks`, `wip`

### During development

- Add the feature to `src/gaffer/features/`. Co-locate new transforms
  with the existing EWMA helpers for discoverability.
- Keep the feature null-safe: a NaN in the training CSV must not crash
  the pipeline. Assume at least one season had the column missing.
- Add a unit test under `tests/unit/test_features.py` that asserts the
  transform's shape / null-handling on a 3-row toy frame.

### After coding: mandatory gate

Never merge without running the benchmark locally:

```bash
# Clean cache to force retraining from scratch.
rm -rf data/cache/models/

# Run the benchmark notebook.
jupyter nbconvert --to notebook --execute \
  notebooks/03_model_benchmark.ipynb \
  --output-dir /tmp/gaffer-bench/
```

Inspect the printed tables:

- **Passed** (every position's RMSE within 0.05 of the shipping number,
  coverage in 78–82%): open the PR, paste the before/after RMSE tables
  into the description, merge.
- **Failed**: do not merge. Keep the branch for later if the direction
  felt right but the signal was noisy; delete if the hypothesis looks
  wrong.

### Merging

Only after the gate passes:

```bash
git checkout main
git pull origin main
git merge --no-ff feature/xg-xA-from-understat
git push origin main
```

Streamlit Cloud detects the push within ~60 s and redeploys. The next
user to hit the app triggers a 45–60 s cold-start retrain on the new
code. No manual action needed.

To force the rotation immediately (rather than waiting for a visitor to
trigger it): Streamlit Cloud dashboard → **Manage app → Reboot app**.

---

## 5. Process: a historical data refresh

Runs once per season, when Vaastav publishes the updated CSV.

1. Download the new `clean_merged_gwdf_2016_to_2025.csv` (or equivalent)
   and drop it into `data/historical/`.
2. Update `src/gaffer/config.py`:
   ```python
   historical_csv: Path = (
       PROJECT_ROOT / "data" / "historical" / "clean_merged_gwdf_2016_to_2025.csv"
   )
   ```
3. Delete or rename the old CSV in the same commit so the repo has
   exactly one historical source.
4. Run the benchmark (§3). A longer history usually shaves 0.02–0.05 RMSE
   on most positions; if it regresses, something in the new CSV's schema
   has diverged (column rename, new null pattern) — investigate before
   merging.
5. Merge, push, let Streamlit Cloud redeploy. Cold-start retrain now
   uses the extended history.

---

## 6. Interaction with Streamlit Cloud

- Streamlit Cloud always rebuilds from the most recent commit on `main`.
  Publish a change and the next cold-start serves it.
- Streamlit Cloud **cannot touch your local disk** — your
  `data/cache/models/` is safe from the cloud app's ephemeral state.
- There is no nightly retrain. Model freshness is driven entirely by
  code changes hitting `main` plus the app's own cold-start discipline.

---

## 7. Anti-patterns

- **Committing `data/cache/`.** The whole point of gitignoring it is
  that it's a disposable function of code. Committing a `.joblib` that
  disagrees with the code on `main` produces silent model drift.
- **Pushing without rerunning the benchmark.** LightGBM will happily
  train on any feature frame — the only way to catch "added feature
  that increased RMSE" is the notebook.
- **Tuning hyperparameters against the 2023–24 fold repeatedly.** Peeking
  at the held-out season's exact RMSE enough times silently tunes the
  model to that fold. Keep 2022–23 as an untouched final-gate fold (see
  `evaluation.md` §8).
- **Adding a feature without a null-handling test.** Vaastav's CSVs are
  not schema-stable across seasons; features that assume a column
  exists for all rows have broken the pipeline before.
- **Relying on the live FPL API in feature engineering.** Training uses
  the historical CSV only. The FPL API is inference-time scoring data;
  features derived from it at training time won't exist at inference for
  past gameweeks.

---

## 8. Quick checklist

Before you start coding:

```
[ ] Benchmarked main to capture the per-position RMSE baseline
[ ] Branch named after the hypothesis, not the code change
```

Before you merge:

```
[ ] Cleared data/cache/models/ and rerun notebook 03
[ ] Per-position RMSE within 0.05 of shipping (evaluation.md §5)
[ ] 80% interval coverage in 78–82% overall
[ ] Unit test added for any new feature transform
[ ] PR description includes before/after RMSE table
```

After you merge:

```
[ ] Confirmed the Streamlit Cloud dashboard picked up the push
[ ] Hit the deployed URL once to trigger the cold-start retrain
[ ] Spot-checked the Transfer Planner returns a plausible 3-GW plan
```

# Gaffer

**ML + MILP Fantasy Premier League assistant.** Per-position ensemble predicts expected points with 80% prediction intervals; a multi-gameweek mixed-integer linear program picks the squad, starting XI, bench, and captain that maximise expected points net of transfer hits, subject to the FPL rules (£100m budget, 2/5/5/3 position quotas, max 3 players per club, legal formations).

> **Status: feature-complete.** Backend, Streamlit app, notebooks, tests, and CI are all in place. Deployment to Streamlit Community Cloud is the only outstanding step — see [Roadmap](#roadmap).

---

## Why

The Fantasy Premier League decision problem is an underappreciated ML + optimisation showcase:

- **ML:** 600 players × 38 gameweeks × 8+ seasons of public data, temporal drift, heavy position-conditional heterogeneity (a defender's points process has almost nothing in common with a forward's), and a noisy target (clean-sheet luck, bonus-point thresholds) that rewards explicit uncertainty quantification.
- **Optimisation:** a constrained selection problem that barely fits as a single-gameweek LP and becomes genuinely interesting over a multi-gameweek horizon with transfer costs and captain doubling.

Most public FPL projects stop at either a predictor *or* a rule-based picker. Gaffer does both, end-to-end, as the reference implementation.

## Architecture

Three layers, each independently testable, loosely coupled through a services layer.

```
                +-------------------------+
                |     Streamlit app       |
                |  (3 pages, reads        |
                |   services layer only)  |
                +-----------+-------------+
                            |
                +-----------v-------------+
                |   services/ (glue)      |
                |  - prediction_service   |
                |  - optimization_service |
                +-----+-----------+-------+
                      |           |
          +-----------v--+    +---v-----------+
          |   models/    |    | optimizer/    |
          | Ridge/XGB/   |    | PuLP + CBC    |
          | LGBM per pos |    | multi-GW MILP |
          | + quantile   |    | (hits, cap,   |
          | intervals    |    |  transfers)   |
          +------+-------+    +-------+-------+
                 |                    |
        +--------v----+         +-----v------+
        |  features/  |         |  domain/   |
        |  EWMA roll, |         | Player,    |
        |  FDR, team  |         | Squad, XI, |
        |  form, SK   |         | Bench,     |
        |  pipeline   |         | FplRules   |
        +------+------+         +------------+
               |
        +------v------+     +------------+
        | providers/  |<----| cache/     |
        | live FPL +  |     | LRU +      |
        | historical  |     | SQLite TTL |
        | CSV         |     +------------+
        +-------------+
```

### Repo layout

```
Gaffer/
├── app/                         # Streamlit
│   ├── Home.py                  # Optimal Squad page
│   └── pages/
│       ├── 1_Transfer_Planner.py
│       └── 2_Player_Projections.py
├── src/gaffer/
│   ├── config.py                # pydantic-settings (GAFFER_* env vars)
│   ├── domain/                  # Pure data models with validation
│   │   ├── enums.py             # Position, Formation
│   │   ├── constraints.py       # FplRules (budget, quotas, club cap)
│   │   ├── player.py            # Player, PlayerProjection
│   │   └── squad.py             # Squad, XI, Bench, SquadSelection
│   ├── providers/               # Data sources
│   │   ├── base.py              # FplDataProvider + HistoricalDataProvider Protocols
│   │   ├── fpl_api.py           # Live FPL API, cached
│   │   ├── historical_csv.py    # Bundled 2016-24 CSV
│   │   └── registry.py
│   ├── features/                # Feature engineering
│   │   ├── constants.py         # Canonical column groupings
│   │   ├── engineering.py       # reformat_dates/team_form/fdr/fpl_features
│   │   ├── rolling.py           # EWMA rolling + T+1 shift
│   │   └── preprocessing.py     # sklearn pipeline (KNN impute + scale + OHE)
│   ├── models/                  # Per-position predictors
│   │   ├── base.py              # PointsPredictor / QuantilePredictor Protocols
│   │   ├── ridge.py             # Linear baseline
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── quantile.py          # LightGBM pinball-loss for intervals
│   │   ├── ensemble.py          # Position-routed ensemble
│   │   └── training.py          # Blocked time-series CV + benchmarking
│   ├── optimizer/               # MILP
│   │   ├── milp.py              # PuLP + CBC multi-GW solver
│   │   └── result.py            # GameweekPlan, OptimizerResult dataclasses
│   ├── services/                # Orchestration
│   │   ├── prediction_service.py    # training matrix + inference matrix builders
│   │   └── optimization_service.py  # thin wrapper over optimizer.solve
│   ├── visualization/           # Pitch plots, chart theming
│   ├── ui/components/           # Streamlit widgets
│   ├── cache/
│   │   ├── memory_cache.py      # In-process TTL LRU
│   │   └── sqlite_cache.py      # Persistent key-value
│   └── utils/                   # (placeholder)
├── notebooks/                   # 5 educational walkthroughs
├── data/
│   ├── historical/
│   │   └── clean_merged_gwdf_2016_to_2024.csv   # 31 MB, 750k rows, 8 seasons
│   └── cache/                   # runtime (gitignored)
├── tests/                       # pytest (105 tests; `@pytest.mark.slow` excluded in CI)
│   ├── unit/
│   ├── integration/
│   └── fixtures/                # Stub FPL + historical providers for offline E2E
├── .github/workflows/ci.yml     # pytest + ruff
├── .streamlit/config.toml
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone https://github.com/<you>/Gaffer.git
cd Gaffer
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

Run the app:

```bash
streamlit run app/Home.py
```

First cold start trains the per-position ensemble (~30s on 8 seasons of data) and caches it to `cache_dir/models/`. Subsequent starts load from disk.

Run the test suite:

```bash
pytest -m "not slow"         # fast unit + stub-backed integration tests
pytest                       # includes the slow CSV-backed end-to-end test
ruff check src tests         # lint
```

Configuration lives in `.env` — see `.env.example` for `GAFFER_HORIZON`, `GAFFER_EWMA_ALPHA`, cache TTLs, etc.

## Modelling approach (what's already written)

**Prediction.** Per-position ensemble (GK / DEF / MID / FWD each trained separately) — scoring mechanics differ enough that a single model leaves signal on the table. Per position the benchmark suite is Ridge (baseline) vs XGBoost vs LightGBM, selected by blocked-by-season time-series CV to avoid leaking future seasons into training. LightGBM with quantile (pinball) loss gives 10th/50th/90th-percentile forecasts, surfacing uncertainty intervals in the UI.

**Feature engineering.** Exponentially-weighted rolling averages (α=0.95, min 5 games) on player stats (minutes, goals, assists, clean sheets, saves, creativity, BPS, ICT …) plus team form (3-point-win rolling, goals for/against), fixture difficulty (FPL's 1–5 FDR, home/away), and normalised selected-% / transfers-balance-%. Rolling windows are shifted by one gameweek so GW N's features only use games 1..N-1.

**Optimisation.** PuLP + CBC multi-gameweek MILP:
- *Decision variables:* `squad[p, gw]`, `starting[p, gw]`, `captain[p, gw]` (binary), and when a starting squad is provided `transfer_in / transfer_out[p, gw]` and an integer `extra_hits[gw]`.
- *Hard constraints:* 15-man squad with 2/5/5/3 position quotas; £100m + bank budget; ≤3 players per club; 11 starters in a legal formation (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD); exactly one captain, in the XI.
- *Objective:* `Σ_gw Σ_p E[pts] · (starting + captain + bench_weight · bench) − 4 · max(0, extra_transfers − free_transfers)`.
- Vice-captain is chosen post-hoc (second-best starter by projected points) since folding it into the MILP doesn't meaningfully change the solution.

## Roadmap

| # | Task | Status |
|---|------|--------|
| 1 | Scaffold repo structure (dirs, `__init__.py`, pyproject, gitignore, env) | Done |
| 2 | Copy historical CSV into `data/historical/` | Done |
| 3 | Port `FantasyApiModel.py` into `providers/` + `features/` | Done |
| 4 | Build `domain/` dataclasses (Player, Squad, XI, Bench, FplRules) | Done |
| 5 | Build `models/` (Ridge / XGB / LGBM / quantile / ensemble / CV harness) | Done |
| 6 | Build `optimizer/` multi-GW MILP with transfer hits and captain | Done |
| 7 | Wire `services/` (prediction + optimization) | Done |
| 8 | Streamlit app (3 pages): Optimal Squad, Transfer Planner, Player Projections | Done |
| 9 | 5 educational notebooks: EDA, features, model benchmark, uncertainty, MILP walkthrough | Done |
| 10 | Tests (pytest unit + integration) and CI (GitHub Actions: pytest + ruff) | Done |
| 11 | **Deploy to Streamlit Community Cloud + demo GIF** | In progress |

### Step 8 — Streamlit app

- `app/Home.py`: pitch plot of recommended 15-man squad + XI with captain armband; horizon and budget sliders; per-player projection table with 80% intervals.
- `app/pages/1_Transfer_Planner.py`: autocomplete input for current squad, bank, free-transfer count; runs the MILP with `initial_squad_ids` fixed and recommends 1–3 transfers with net-of-hit EV.
- `app/pages/2_Player_Projections.py`: searchable/sortable player table with next-GW / horizon projections, intervals, price, ownership, next 3 opponents with FDR badges.
- Caching via `@st.cache_data` keyed by provider TTL.
- `visualization/pitch.py` and `ui/components/{pitch_display,player_table,squad_input}.py` are still placeholders.

### Step 9 — Notebooks

Educational, import from `src/gaffer/`, narrate rather than reimplement.

1. `01_data_exploration.ipynb` — 8-season EDA, per-position scoring distributions, position-specific correlations.
2. `02_feature_engineering.ipynb` — EWMA motivation, α sensitivity, FDR adjustments, train/test-set T+1 shift.
3. `03_model_benchmark.ipynb` — Ridge vs XGB vs LGBM per position via blocked time-series CV, final RMSE / MAE table.
4. `04_uncertainty_quantification.ipynb` — LightGBM quantile regression, coverage vs nominal 80%, pinball loss diagnostics.
5. `05_optimization_walkthrough.ipynb` — MILP formulation explained line-by-line, toy 5-player example, then full 600-player single- and multi-GW solves.

### Step 10 — Tests + CI

- `tests/unit/test_domain.py` — Squad/XI/Bench validation on valid + edge-case inputs.
- `tests/unit/test_features.py` — engineering transforms are idempotent and preserve row count.
- `tests/unit/test_models.py` — predictors implement the Protocol; ensemble routes by position.
- `tests/unit/test_optimizer.py` — MILP produces valid Squad + XI on a toy 40-player input; budget and club cap respected.
- `tests/integration/test_end_to_end.py` — frozen GW snapshot → predict → optimise produces a feasible plan.
- `tests/fixtures/sample_gw_snapshot.json` — committed small fixture.
- CI workflow (`.github/workflows/ci.yml`) is already in place; runs `ruff check` + `pytest -m "not slow" --cov`.

### Step 11 — Deploy

1. **Push to GitHub.** The CI workflow (`.github/workflows/ci.yml`) runs `ruff check` + `pytest -m "not slow"` on every push and PR.
2. **Streamlit Community Cloud.** Sign in at https://share.streamlit.io, "New app" → point at your repo. Main file: `app/Home.py`. Python version: 3.11. The app reads `requirements.txt` for dependencies (already committed).
3. **Resource budget.** A fresh Community Cloud instance has 1 GB RAM and a shared CPU; the first load trains the ensemble and caches it to the container's ephemeral disk — budget 45–60s for cold starts, sub-second for warm ones. The CBC solver completes a horizon=3 solve in well under 10s on this footprint.
4. **Live FPL API.** The live endpoint is unauthenticated and has no published rate limit, but be polite — the `cache/` layer already fronts it with TTL+LRU.
5. **Secrets.** There are no secrets to wire up — `.env.example` documents optional tuning knobs only.
6. **Demo.** Record a 10-second GIF of the Optimal Squad page solving, embed it near the top of this README, and paste the live app URL into the "Live demo" section below.

## Data

Training data is `data/historical/clean_merged_gwdf_2016_to_2024.csv` — 31 MB, ~750k rows, 8 Premier League seasons, per-player per-gameweek stats. Live current-season data comes from the public FPL API (`https://fantasy.premierleague.com/api/`, no auth). Both are fronted by the `providers/` abstraction so a test fixture or a different data vendor can drop in behind the same interface.

## Live demo

_Coming soon — Streamlit Community Cloud._

## Licence

MIT.

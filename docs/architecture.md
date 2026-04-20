# Gaffer — Architecture & Technology Decisions

This document records the architectural decisions made while building Gaffer,
an ML + MILP Fantasy Premier League assistant. Each section describes **what**
was chosen and — more importantly — **why** for *this* project. The goal is to
give a future maintainer (or future me) enough context to extend the system
confidently or re-evaluate a decision when requirements change.

---

## 1. Project goals and constraints

Gaffer has three jobs, each exposed as a page in the Streamlit app:

- **Optimal Squad** — given the current gameweek and the next N fixtures,
  recommend the 15-man squad, starting XI, and captain that maximise expected
  points net of transfer hits.
- **Transfer Planner** — same optimiser, but fixed on a user-supplied starting
  squad. Recommends 1–3 transfers subject to FPL's 4-points-per-extra-transfer
  penalty.
- **Player Projections** — every available player with point estimates and
  80% prediction intervals, searchable and sortable.

Concrete goals:

- **Theoretically grounded.** Per-position ensemble (scoring mechanics differ
  between GKs and forwards enough that a single model leaves signal on the
  table) with blocked time-series CV (no leakage from future seasons) and
  quantile regression for real intervals (not Gaussian-assumption bands).
- **Honest uncertainty.** The optimiser currently uses point estimates, but
  the app always shows intervals next to them — a 6.0 ± 1.0 projection and a
  6.0 ± 4.0 projection are not the same pick.
- **Free to host, free to run.** Streamlit Community Cloud, CPU-only,
  everything fits in the free tier. No paid APIs, no database, no external
  model registry.
- **Local-first.** `pip install -e ".[dev]" && pytest -m "not slow"` runs
  green on a fresh clone with no secrets. The only external call is to the
  public FPL API, which the `cache/` layer fronts.
- **Readable over clever.** One-person codebase, ~4k lines. No framework
  tourism, no premature abstractions.

These constraints shape every decision below.

---

## 2. Language and build system

### Python 3.11+

**Why Python.** pandas / scikit-learn / lightgbm / xgboost / pulp / streamlit
are all Python-first. The ML + OR ecosystem here is Python-first. Switching
languages would buy nothing.

**Why 3.11+.** Modern typing (`X | Y`, `list[...]`), pydantic v2, pattern
matching if ever useful, meaningful interpreter speedups. Pinned in
`pyproject.toml`.

### `src/` layout + Hatchling

`src/gaffer/` instead of a flat `gaffer/` at repo root. Forces an install
before import, which catches the "works because of CWD" class of bug that a
flat layout hides. `pythonpath = ["src"]` in `[tool.pytest.ini_options]` lets
pytest find the package without a full install during fast iteration.

Hatchling is the default modern PEP 621 build backend. No Poetry lock-file
machinery — Streamlit Cloud and GitHub Actions both `pip install -e .`, so a
second source of truth would just drift.

### Optional extras: `[dev]`

Only one extras group: `[dev]` covers pytest, ruff, pytest-cov. The app and
training dependencies are in `[project.dependencies]` because every runtime
target (local, Streamlit Cloud, CI) needs them.

---

## 3. Modelling — per-position ensemble + quantile head

This is the load-bearing ML decision. The FPL points process is strongly
position-conditional: a GK's points come from saves, clean sheets, and
occasional bonus — features like `creativity` or `threat` are near-constant
zero. Forwards' points are dominated by `goals_scored`, with massive variance.
Training one global regressor forces a bias-variance compromise against both.

### Per-position ensemble

`src/gaffer/models/ensemble.py` holds one trained predictor per
`Position ∈ {GKP, DEF, MID, FWD}`. Each leaf predictor is a full model
instance (LightGBM by default; Ridge and XGBoost also available). The
ensemble:

- At `fit`, splits rows by position and trains each leaf on its own slice.
- Locks feature names at fit-time (`feature_names_in_`) so inference frames
  get re-indexed to the same set. This is defensive against upstream
  feature-engineering changes that silently add or drop columns.
- Drops non-numeric columns (team strings, dates) inside `_numeric_features`
  rather than expecting callers to prepare the frame. LightGBM and XGBoost
  need numeric inputs; pushing that concern down into the ensemble keeps the
  training service clean.

**Why not one model with `position` as a feature.** Tried it implicitly by
letting LightGBM branch on a position dummy. It works but converges slower
and loses interpretability — you can't look at per-position feature
importance, RMSE, or residual plots independently. Per-position cost is one
extra `fit` call per position; negligible.

### Point + quantile heads

`train_or_load_ensembles` returns **two** `PositionEnsemble` instances:

- **Point ensemble** — `LgbmPredictor` with squared-error loss. Used by the
  optimiser (MILP needs scalar expected points).
- **Quantile ensemble** — `LgbmQuantilePredictor` with pinball loss at
  α=0.1 and α=0.9 (plus 0.5 for the median). Used by the UI to render 80%
  intervals next to every projection.

Same training matrix, same per-position routing — the only difference is the
loss. Quantile regression gives real conditional intervals: a rotation-risk
midfielder gets a wider band than a nailed-on starter, automatically, with
no separate uncertainty model.

**Why LightGBM as the default leaf.** Benchmarked against Ridge (too weak)
and XGBoost (comparable RMSE, slower to train on this CPU, less forgiving
of missing values). LightGBM handles the 30-ish numeric features with
missingness, trains in <30 s on 8 seasons, and ships a first-class quantile
objective.

### Blocked time-series CV

`src/gaffer/models/training.py::season_block_splits` produces walk-forward
folds: train on seasons `[2016..2020]` → test on 2021, train on
`[2016..2021]` → test on 2022, and so on. Random k-fold on player-GWs would
leak *future* seasons into training through shared player identities and
correlated meta — exactly the kind of silent optimism that produces a model
that benchmarks brilliantly and ships poorly.

---

## 4. Feature engineering

`src/gaffer/features/`:

- **`engineering.py`** — row-level reformatting. Converts FPL's home/away
  split (`team_h_score`, `team_h_difficulty`) into per-side columns
  (`team_points`, `team_fdr`, `opponent_team_fdr`). Normalises
  `transfers_balance` and `selected` to fractions of total managers so
  season-over-season manager counts don't distort the signal. Fixes the
  `GK` → `GKP` position-code inconsistency that haunts the FPL history.
- **`rolling.py`** — EWMA rolling aggregation on player/team/FPL features
  with a **T+1 shift**. GW N's feature row sees only stats from GWs 1..N-1.
  Without the shift, the target (`total_points` for GW N) leaks into the
  features (which are functions of that same GW). This is the single most
  common bug in FPL modelling and the reason
  `tests/unit/test_features.py` has an explicit T+1 assertion.
- **`constants.py`** — canonical column groupings. Every other module
  concatenates these lists rather than hard-coding column lists, so adding
  a feature only touches one file.

**Why EWMA rather than fixed windows.** A hot player is more predictive of
next week than their season-long average; an injured player's stale games
shouldn't dominate. EWMA smoothly down-weights older observations without
the hard boundary of a window. `α=0.95` (default, configurable via
`GAFFER_EWMA_ALPHA`) was tuned on the CV.

---

## 5. Optimiser — PuLP + CBC MILP

`src/gaffer/optimizer/milp.py`.

### Variables

| Variable | Binary? | Meaning |
|---|---|---|
| `squad[p, gw]` | ✓ | Player `p` is in the squad for GW `gw`. |
| `starting[p, gw]` | ✓ | `p` is in the XI for `gw`. |
| `captain[p, gw]` | ✓ | `p` is captain for `gw` (2× points). |
| `transfer_in[p, gw]` | ✓ | `p` entered the squad at `gw`. |
| `transfer_out[p, gw]` | ✓ | `p` left the squad at `gw`. |
| `extra_hits[gw]` | int ≥ 0 | Transfers at `gw` above the free allowance. |

### Hard constraints

- 15-man squad with 2/5/5/3 position quotas.
- Budget ≤ £100m + user-supplied bank.
- ≤ 3 players per club.
- 11 starters in a legal FPL formation (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD).
- Exactly one captain, and the captain must be in the XI.
- `transfer_in - transfer_out = squad[gw] - squad[gw-1]` (flow balance).

### Objective

```
maximise  Σ_gw Σ_p E[pts(p,gw)] · (starting + captain + bench_weight · bench)
          − 4 · extra_hits[gw]
```

**Why PuLP + CBC.** CBC is open-source, bundled with PuLP, runs on Streamlit
Community Cloud with zero setup. Commercial solvers (Gurobi, CPLEX) would
solve faster but cost money, need license seats, and add a dependency the
app can't install in the cloud.

**Why a multi-gameweek formulation at all.** Single-GW optimisation picks
the best XI for this week, but ignores transfer-cost amortisation: a
transfer paid this week is "free" for the next N weeks. Optimising over a
3–5 GW horizon captures that trade-off. 5 GWs is about the longest horizon
where CBC still finishes in under 60 s on Cloud CPU.

**Why vice-captain is chosen post-hoc.** The vice-captain's EV contribution
is tiny (it only pays off if the captain doesn't play, which happens a few
% of the time) and adding `vice[p, gw]` variables with "vice ≠ captain"
constraints doubles the captain sub-model for near-zero benefit. Post-hoc:
pick the second-highest projected starter.

**Why `bench_weight` is configurable.** Bench players score nothing in
starters' week but matter when a starter is rotated or injured. Defaulting
to 0.1 gives the solver a reason to keep *some* bench depth; raising it to
0.3 gets a more robust squad at the cost of some starter quality. Exposed
as a slider in the app.

---

## 6. Providers — Protocol abstraction over FPL API + historical CSV

`src/gaffer/providers/`:

- **`base.py`** — `FplDataProvider` + `HistoricalDataProvider` Protocols.
  Everything downstream consumes these, not the concrete implementations.
- **`fpl_api.py`** — live endpoints (`bootstrap-static`, `element-summary/{id}`,
  `fixtures`). Returns pandas frames.
- **`historical_csv.py`** — bundled `clean_merged_gwdf_2016_to_2024.csv`
  (~31 MB, 750k rows, 8 seasons).
- **`registry.py`** — resolves which provider to use from
  `settings.data_source`.

**Why Protocols rather than ABCs.** Duck-typed, zero runtime cost, no
inheritance contract to maintain. Tests in `tests/fixtures/stub_providers.py`
satisfy the same Protocol with in-memory synthetic data — no network, no
31 MB CSV, integration tests run in seconds.

**Why the CSV ships in the repo.** It's the training set. Downloading it at
app boot would make cold-starts slow and flaky. 31 MB is well within
GitHub's soft file-size limit and the Streamlit Cloud build tarball fits
comfortably.

---

## 7. Domain models — pydantic v2 frozen dataclasses

`src/gaffer/domain/` holds the shapes: `Player`, `PlayerProjection`, `Squad`,
`XI`, `Bench`, `SquadSelection`, `Formation`, `FplRules`.

Every model is **frozen** and validates on construction via
`model_validator(mode="after")`. `Squad` rejects wrong size, broken
position quotas, budget overrun, club-cap violations, and duplicate players —
before any other code gets a chance to assume the invariants hold. The
optimiser's output is fed through `SquadSelection` as a final integrity
check.

**Why pydantic and not dataclasses + hand-rolled validation.** Pydantic v2
gives declarative validation, free JSON schema, and frozen-by-default at
zero runtime cost. The alternative is 200 lines of `__post_init__` spaghetti
for the same effect.

**Why validate at all, given the MILP already enforces the constraints.**
The MILP's solution is only as valid as the LP formulation. A bug in
constraint wiring (wrong coefficient, missing sum) might produce a
"solution" that the domain model catches before it reaches the UI. Two
checks with different implementations catch bugs one check would miss.

---

## 8. UI framework — Streamlit

**Chosen:** Streamlit (`>=1.30`), multi-page app rooted at `app/Home.py`
with `app/pages/1_Transfer_Planner.py` and `app/pages/2_Player_Projections.py`.

**Alternatives considered:**

| Option | Why not |
|---|---|
| **Dash / Plotly** | More flexible, forces callbacks + hand-managed state. Too much plumbing. |
| **Gradio** | Great for single-model demos, weak for multi-page data-dense apps. |
| **FastAPI + React** | The "real" answer for production SaaS. Massive overbuild for a personal project. |
| **Jupyter** | No shareable UI for non-technical users; notebooks are for explanation, not interaction. |

**Why Streamlit wins:** pure Python, multi-page model, free hosting
(§10), and `@st.cache_resource` / `@st.cache_data` fit the expensive-model /
cheap-output split exactly.

**Trade-offs accepted:**

- Streamlit reruns the full script on every interaction. Fine because the
  model load is cached and the MILP solve is itself cached keyed on
  (horizon, bench_weight).
- Community Cloud sleeps after ~15 min idle and cold-starts in ~45 s. Fine
  for a personal tool.

---

## 9. Caching — three layers

There are three distinct things that benefit from caching, and each gets a
different strategy:

### 9.1 FPL API responses — in-process TTL LRU + SQLite

`src/gaffer/cache/memory_cache.py` + `sqlite_cache.py`. The bootstrap
endpoint changes rarely (player prices update nightly); fixtures change
when gameweek deadlines pass. Default TTLs are 6h (bootstrap) and 24h
(fixtures), configurable via `GAFFER_BOOTSTRAP_TTL_HOURS`.

### 9.2 Trained ensembles — joblib on disk + `@st.cache_resource`

`src/gaffer/services/model_cache.py` trains once per (code change, data
change) and persists to `cache_dir/models/ensemble_{point,quantile}.joblib`.
The Streamlit app wraps that in `@st.cache_resource` so the in-memory model
is also shared across sessions on the same container.

Training takes ~30 s on 8 seasons. Without disk caching, every cold-start
on Streamlit Cloud would retrain — an unacceptable 30 s tax that has
nothing to do with serving a new user.

### 9.3 MILP solutions — `@st.cache_data` keyed by inputs

`solve_optimal_squad` in `app/Home.py` caches by `(horizon, bench_weight)`.
Moving the bench-weight slider from 0.1 to 0.2 and back doesn't re-solve.
Transfer Planner uses a similar pattern keyed by the selected squad.

**Why `cache_resource` for models but `cache_data` for MILP output.**
Loaded models are stateful objects (PyTorch-style) holding internal
buffers. `cache_resource` is the correct decorator — one instance per
worker. The MILP solution is a pure dataclass, serialisable, and different
inputs produce different outputs — `cache_data` (snapshot semantics) fits.

---

## 10. Hosting — Streamlit Community Cloud

**Chosen:** Streamlit Community Cloud (free tier).

**Why.** Free, GitHub-integrated, HTTPS by default, TOML-based secrets (we
don't use any). Zero Docker to maintain. Deploys on every push to the
configured branch.

**Deployment shim.** Streamlit Cloud runs
`streamlit run app/Home.py`. `pip install -e ".[dev]"` happens
automatically because `pyproject.toml` is at the repo root. The
`src/gaffer/` package is importable because `pip install -e .` adds `src/`
to `sys.path`.

**Resource budget:**

- 1 GB RAM, shared CPU.
- 45–60 s cold start (model training + cache warm).
- <10 s per MILP solve at horizon=3.
- Apps sleep after ~15 min of inactivity — expected, harmless for a
  personal demo.

See `docs/deployment-guide.md` for the step-by-step.

---

## 11. Configuration — pydantic-settings

`src/gaffer/config.py` exposes one `Settings` singleton. Every knob is
read from `GAFFER_*` env vars (or `.env` locally), defaults validated via
pydantic:

| Setting | Default | What |
|---|---|---|
| `horizon` | 3 | MILP lookahead in gameweeks (1..5). |
| `solver_time_limit` | 60 s | CBC time limit. |
| `ewma_alpha` | 0.95 | Rolling-average decay. |
| `bootstrap_ttl_hours` | 6 | FPL bootstrap cache TTL. |
| `fixtures_ttl_hours` | 24 | Fixtures cache TTL. |
| `cache_dir` | `data/cache` | Where joblib + SQLite live. |
| `data_source` | `"live"` | `"live"` for FPL API, `"csv"` for historical-only. |

**Why pydantic-settings and not plain `os.environ`.** Validation on load.
A typo in `GAFFER_HORIZON=foo` fails fast at import time, not deep inside
the MILP with a confusing type error.

---

## 12. Testing — pytest

`tests/` is split by concern:

- `tests/unit/test_domain.py` — Player / Squad / XI / FplRules validation.
- `tests/unit/test_features.py` — engineering transforms + T+1 leakage gate.
- `tests/unit/test_models.py` — Protocol satisfaction, ensemble routing,
  season-block CV splitter.
- `tests/unit/test_optimizer.py` — MILP produces valid Squad / XI on a toy
  40-player pool; budget, club cap, formation all respected.
- `tests/unit/test_cache.py` — TTL / LRU / SQLite round-trip.
- `tests/unit/test_config.py` — Settings defaults + validation errors.
- `tests/integration/test_end_to_end.py` — stub providers → train → project
  → optimise. Plus a `@pytest.mark.slow` test using the real 31 MB CSV.

**Why a T+1 leakage assertion.** The single most common FPL-modelling bug
is letting GW N's target leak into GW N's features through an unshifted
rolling stat. An explicit test here is worth ten vague "model looks
plausible" checks.

**Why stub providers in `tests/fixtures/stub_providers.py`.** Real providers
need network (FPL API) or a 31 MB file. Integration tests that use
either are slow and flaky. Stubs implement the same Protocol with
deterministic synthetic data, keyed by the same `season`/`kickoff_time`
shape. Run in < 1 s; `@pytest.mark.slow` covers the real-data path.

CI runs `ruff check src tests` and `pytest -m "not slow" --cov=src/gaffer`.

---

## 13. What was deliberately left for later

- **Double-gameweek and blank-gameweek handling.** The MILP assumes one
  fixture per team per GW. DGWs and BGWs are handled by summing
  projections per (player, gw) upstream — correct EV but the optimiser
  doesn't know about the doubled rotation risk or blank weeks where a
  transfer is unusually valuable.
- **Chips (Wildcard, Free Hit, Triple Captain, Bench Boost).** Not
  modelled. Adding a Wildcard is a one-GW constraint relaxation
  (free_transfers = ∞), which the current formulation supports already;
  Free Hit and the others need extra variables.
- **Uncertainty-aware optimisation.** The MILP uses point estimates. Using
  quantile predictions (e.g. maximise 25th percentile) is a one-line swap
  and produces a more conservative squad; not enabled by default.
- **Price-change modelling.** FPL player prices drift weekly. The
  optimiser treats prices as fixed — over a 5-GW horizon this is a
  small-but-real error.
- **Transfer-hit risk-aversion.** Hits are modelled as a deterministic 4
  points. A risk-averse manager might want a penalty that grows with
  uncertainty; a risk-loving one might lower the hit cost. Exposed as
  future work.
- **Wildcard / Free Hit timing recommendation.** A separate question from
  squad selection — "which GW is best to wildcard given the fixture
  swings ahead?" — not in scope.

---

## 14. Summary — the one-paragraph version

Gaffer is a **Fantasy Premier League assistant** written in **Python 3.11+**
built around a **per-position ensemble** (LightGBM with a quantile head for
80% prediction intervals) trained under **blocked-by-season time-series CV**
on 8 seasons of historical FPL data plus the live FPL API. A **PuLP + CBC
MILP** picks the squad / XI / captain over a 1–5 GW horizon, accounting for
£100m budget, 2/5/5/3 quotas, the club cap, and 4-point transfer hits. The
frontend is a **Streamlit multi-page app** (Optimal Squad, Transfer Planner,
Player Projections) deployed free on **Streamlit Community Cloud**. All
model-training and solver work happens inside the user's session — there is
no database, no external model registry, no paid API. `pip install -e ".[dev]"
&& pytest -m "not slow"` runs green on a fresh clone.

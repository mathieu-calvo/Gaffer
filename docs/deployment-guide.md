# Deployment Guide — Streamlit Community Cloud

This guide walks through deploying Gaffer as a 24/7 hosted Streamlit app
where visitors plan their FPL transfers against live FPL API data, with
the model re-training on first cold-start from the bundled historical
CSV.

**Stack:**
- **Hosting:** Streamlit Community Cloud (free)
- **Data source at runtime:** the public FPL API
  (`https://fantasy.premierleague.com/api/bootstrap-static/` — no key
  required)
- **Training data:** `data/historical/clean_merged_gwdf_2016_to_2024.csv`
  shipped in-repo (~20 MB)
- **Model storage:** `data/cache/models/*.joblib`, written on first boot,
  lives on the app's ephemeral disk
- **Auth:** none — the app is stateless; no user data is collected
- **Retraining:** none scheduled. Triggered manually by rebooting the app
  (see §5) or by clicking the "Force retrain" button in the sidebar

Pair with [`architecture.md`](architecture.md) for the "why" of each
component and [`training.md`](training.md) for what happens during the
cold-start training run.

---

## Prerequisites

- A GitHub account with this repo pushed to it
- Python 3.11+ locally (for a pre-flight end-to-end test)

Install the package locally with all extras to smoke-test before pushing:

```bash
pip install -e ".[dev,app]"
```

---

## Step 1: Smoke-test locally first

Don't deploy broken code. Before pushing, run the full local boot the way
Streamlit Cloud will run it:

```bash
# Clean cold-start — simulates the first boot on Streamlit Cloud.
rm -rf data/cache/

streamlit run app/Home.py
```

Expectations on a cold start:

- First page load takes 45–60 s (feature engineering + per-position
  ensemble training on 2016–2024).
- `data/cache/models/position_ensemble.joblib` and
  `quantile_ensemble.joblib` appear after training completes.
- Subsequent page loads are <2 s because both model files and the FPL
  bootstrap JSON are `@st.cache_resource` / `@st.cache_data` backed.

Walk through the two pages — Home, Transfer Planner, Player Projections —
and solve at least one MILP horizon on the Transfer Planner to confirm
the end-to-end pipeline works.

---

## Step 2: Commit and push the code to GitHub

Make sure all changes are committed and pushed to `main`:

```bash
git add -A
git commit -m "prep for streamlit cloud deploy"
git push origin main
```

Streamlit Cloud deploys from a named branch, not from a specific commit —
every push to `main` triggers a re-deploy automatically.

---

## Step 3: Create the Streamlit Cloud app

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Fill in:
   - **Repository**: `your-username/Gaffer`
   - **Branch**: `main`
   - **Main file path**: `app/Home.py`
4. Click **Advanced settings**.
5. Set **Python version** to `3.11`.
6. **Secrets**: leave empty. Gaffer has no API keys, no database, no
   tokens. The FPL public API requires no auth.

   (If you ever add a secret — e.g. a paid weather API for fixture
   modelling — the format is TOML; read via `st.secrets["NAME"]` or by
   setting `GAFFER_...` env vars that `pydantic-settings` picks up.)

7. Click **Deploy!**

---

## Step 4: Verify the app works end-to-end

1. Wait 2–3 minutes for the build, then another 45–60 s for the first
   page load (cold-start training). Watch the logs in the Cloud
   dashboard — you should see lines from
   `src/gaffer/services/prediction_service.py` reporting "training
   position ensemble…".
2. Once deployed, you'll get a URL like
   `https://gaffer-<hash>.streamlit.app`.
3. Open the URL. The home page should render the current FPL gameweek
   summary pulled live from the FPL API.
4. Click **Transfer Planner** in the sidebar.
5. Enter a team ID (any valid FPL manager id — yours, or a public one
   like `1` for the overall-rank-1 team) and solve a 3-GW horizon.
6. The planner should return a net-points number and a list of
   suggested transfers in <10 s.
7. Click **Player Projections** — a filterable table of all 820 players
   with point estimates and 80% intervals should render.

If any of these fail, the Streamlit Cloud **Manage app → Logs** pane is
the first place to look.

---

## Step 5: Rotating the model in production

Gaffer has no model registry — the "published model" is whatever joblib
file sits in the app's `data/cache/models/` directory on Streamlit
Cloud's ephemeral disk. The rotation flow is therefore:

1. Make your change locally (feature engineering, hyperparameter tweak,
   new base learner — see [`improvement-workflow.md`](improvement-workflow.md)
   for the gate).
2. Push to `main`.
3. Streamlit Cloud auto-redeploys within ~1 minute.
4. The redeploy wipes the ephemeral disk, which deletes the old joblib
   files. The next page load triggers a cold-start retrain on the new
   code.

Alternatively, to force a retrain without pushing code (e.g., you've
swapped in an updated `clean_merged_gwdf_2016_to_2024.csv`):

- **Option A** — Streamlit Cloud dashboard → your app → **Manage app** →
  **Reboot app**. Wipes the cache, triggers cold-start.
- **Option B** — a user-facing "Force retrain" button exists in the
  sidebar; clicking it calls `st.cache_resource.clear()` and re-fits
  both ensembles in-process (faster than a full reboot because the FPL
  API cache is preserved).

There is no roll-back path: the app always trains from the current
`main`. Roll back by reverting the commit and pushing.

---

## Step 6: Resource budget — what fits on the free tier

Streamlit Community Cloud gives each app **1 GB RAM** and a shared CPU.
Gaffer is designed to fit:

| Stage | Budget | Actual | Headroom |
|---|---|---|---|
| Cold-start training (first page load) | 60 s | 45–55 s | ~5–15 s |
| Warm page load | 5 s | <2 s | fine |
| MILP solve (horizon=3) | 10 s | 3–8 s | fine |
| MILP solve (horizon=5) | 30 s | 15–25 s | fine |
| Peak memory (training + FPL cache) | 1 GB | ~700 MB | ~300 MB |

If you add features that push training over 60 s, the app will appear to
hang to first visitors. Two mitigations:

- Train locally and commit the `.joblib` files (removes them from
  `.gitignore`). Trades repo size for cold-start latency.
- Cache training output to a small external store (S3, R2). More plumbing
  than most changes warrant.

---

## Step 7: Local development (unchanged)

None of the above affects local development. With no environment
variables or `.streamlit/secrets.toml` set, the app behaves identically:

- Reads `data/historical/clean_merged_gwdf_2016_to_2024.csv` for
  training.
- Hits the live FPL API at runtime.
- Writes joblib files to `data/cache/models/`.

```bash
# Run the app.
streamlit run app/Home.py
```

To force the local app to use a pinned historical CSV as its *live*
source (useful for offline demos), set:

```bash
export GAFFER_DATA_SOURCE=csv
```

`src/gaffer/config.py` picks it up via pydantic-settings; the provider
factory swaps in the CSV-backed `HistoricalProvider` instead of the
live one.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| App build fails with `ModuleNotFoundError` | The package isn't in `requirements.txt`. Add it and push; Streamlit Cloud re-builds automatically. |
| Cold-start page hangs past 90 s | Training is exceeding the soft budget. Check Cloud logs for a specific slow step; see §6 for mitigations. |
| `FileNotFoundError: clean_merged_gwdf_2016_to_2024.csv` | The historical CSV isn't tracked in git (check `data/historical/.gitignore`). Pull it back and force-add with `git add -f data/historical/*.csv`. |
| FPL API returns 503 / rate-limits | The public FPL API sometimes throttles. The provider cache (`bootstrap_ttl_hours=6`) limits hit rate; a reboot usually clears it. |
| MILP solver returns "Infeasible" | User's inputs are inconsistent — e.g., bench budget too low for required GKP price. Shown to user as an error banner; not a deploy issue. |
| Joblib files not regenerating after a push | Streamlit Cloud occasionally caches the disk across redeploys. Force it via **Manage app → Reboot app**. |
| App cold-start is slow after ~15 min idle | Streamlit Cloud free tier sleeps after inactivity. First-hit-after-sleep pays the cold-start cost again. Expected and harmless. |

---

## Architecture notes

- **No secrets.** The FPL API is public; there is no user database, no
  auth, no third-party paid services. The `.streamlit/secrets.toml`
  entry in `.gitignore` is precautionary for future additions.
- **Ephemeral state.** Everything under `data/cache/` is disposable.
  Rebooting the app throws it away and the next cold-start rebuilds it.
  This is why the `.joblib` files are gitignored — there's no point
  versioning a cache.
- **Training data is versioned.** The 2016–2024 historical CSV lives in
  the repo (force-tracked despite being in a "data" subdirectory) so
  that a fresh clone can train reproducibly. When a new season's CSV
  drops, replace the file, update `config.historical_csv`, push — the
  cold-start retrain absorbs it.
- **Cost:** the free tier is the only tier Gaffer needs. No paid
  storage, no paid compute, no paid APIs. Moving to a paid Streamlit
  plan would only matter if you wanted a custom domain or private app
  access.

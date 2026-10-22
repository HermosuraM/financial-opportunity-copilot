# Financial Opportunity Copilot

A small end-to-end system that flags financial opportunities for bank customers
(idle cash earning near-zero interest, an emergency-fund gap, employer 401k
match left on the table, high-interest debt, tax-loss-harvest candidates,
creeping subscription spend) with ML, then uses an **agentic LLM layer** to
decide which ones actually matter for a given customer and explain them in
plain English — with every dollar figure backed by a deterministic tool call,
never by the model's own arithmetic.

An independent portfolio project covering ML for financial insight, an agentic
system for opportunity detection, model diagnostics, and blending ML with
financial-planning logic. It uses no real bank's customer data: the customer
base is the public UCI Bank Marketing dataset and the transactions are
synthetic.

## Architecture

```
UCI Bank Marketing (real, 41,188 customers)
        │  age, job, marital, education, existing loan flags
        ▼
build_customers.py  ──►  customers.csv  (+ derived income & tax bracket)
        │
        ▼
generate_synthetic.py  ──►  accounts.csv, transactions.csv, holdings.csv
        │  (5,000-customer sample: balances, 401k, credit cards, brokerage lots,
        │   12 months of categorized spend)
        ▼
load_db.py  ──►  SQLite (data/processed/finance.db)
        │
        ▼
queries.sql + build_features.py  ──►  feature_table.csv
        │  (SQL aggregates + window-function subscription trend)
        ▼
heuristic_labels.py  ──►  6 weak-supervision opportunity labels
        │
        ▼
train_opportunity_models.py  ──►  one classifier per opportunity
        │  (trained on a noised T-1-style snapshot; see below)
        ▼
pipeline.py score_customer()  ──►  ranked opportunity list per customer
        │
        ▼
copilot_agent.py  ──►  Claude + 6 finance tool functions  ──►  prioritized brief
```

## Modeling approach

There's no public dataset of "did this customer act on opportunity X," so
labels are bootstrapped from six deterministic financial rules (weak
supervision) — see `src/features/heuristic_labels.py`. Rather than have each
model trivially memorize its own rule, training data is a **noised snapshot**:
balances, spend, and holdings get multiplicative drift and ~12% missingness
(`simulate_snapshot_noise`) to mimic a real batch-scoring pipeline reading a
slightly stale, occasionally-unsynced data feed, while labels are computed
from the clean ground truth.

Results (full table + ROC/PR/calibration/permutation-importance plots for
every label in [`reports/`](reports/)):

| opportunity | best model | PR-AUC | ROC-AUC | base rate |
|---|---|---|---|---|
| opp_subscription_creep | HistGradientBoosting | 0.904 | 0.969 | 18.3% |
| opp_tax_loss_harvest | HistGradientBoosting | 0.979 | 0.998 | 7.9% |
| opp_high_yield_savings | HistGradientBoosting | 0.939 | 0.951 | 40.3% |
| opp_emergency_fund_gap | HistGradientBoosting | 0.955 | 0.946 | 55.6% |
| opp_retirement_gap | HistGradientBoosting | 1.000 | 1.000 | 40.1% |
| opp_high_interest_debt | LogisticRegression | 1.000 | 1.000 | 65.0% |

**Reading this honestly, not just reporting numbers:** `opp_retirement_gap`
and `opp_high_interest_debt` hit ~1.00 because their rules key off account
*metadata* a bank always knows exactly (contribution rate vs. employer match
cap; card APR) — there's no staleness to model, so it's a rule lookup, not a
learned pattern. The other four key off noised balance/spend/holdings feeds
and land in a more believable 0.90–0.98 range, which is where the modeling
actually earns its keep. The demo notebook
([`notebooks/demo.ipynb`](notebooks/demo.ipynb)) walks through this
distinction with the diagnostic plots inline.

## The agent

`src/agent/copilot_agent.py` wraps the Claude API with six tool functions
(`src/agent/tools.py`): retirement-match-gap, high-yield-savings-gain,
emergency-fund-gap, tax-loss-harvest-estimate, debt-payoff-priority (avalanche
method), and 401k contribution headroom. Given a customer profile and the
opportunities the models flagged, the agent:

1. picks the 1-3 opportunities actually worth surfacing this month (not every
   flagged item — a wall of nudges gets ignored),
2. calls a tool for every number it states instead of computing it,
3. writes a short, plain-English brief,
4. never recommends a specific security or gives individualized investment/tax
   advice, and closes with an explicit informational-only disclaimer.

Every tool call is logged for auditability (`reports/briefs/*.json`) — useful
both for debugging and for the kind of explainability a financial-services
compliance review would ask for.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then put your ANTHROPIC_API_KEY in .env
```

## Running it

```bash
python -m src.pipeline build-data       # download UCI data, synthesize overlay, load SQLite
python -m src.pipeline build-features   # run the SQL feature queries
python -m src.pipeline train            # train all 6 opportunity models + diagnostics
python -m src.pipeline brief --customer-id 57   # score a customer + generate a live agent brief
```

Or open [`notebooks/demo.ipynb`](notebooks/demo.ipynb) for the narrated
walkthrough (already executed, with plots and tables inline — no API key
needed to read it; the last section shows how to run the live agent).

## Tests

```bash
pytest tests/ -v
```

19 tests cover the deterministic finance-math tools (exact expected dollar
values), the heuristic labeling rules and noise simulation, the SQL feature
builder (against an in-memory fixture DB), and the agent's tool-call loop
(mocked Claude client, no network/API key required).

## Data & assumptions

- **Real**: demographics and existing-loan flags for 41,188 customers, from
  the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
  (Moro, Cortez & Rita, 2014).
- **Synthetic**: income (derived from job/education/age via a documented
  lookup table, not a real salary field), account balances, 401k details,
  credit card terms, brokerage holdings, and 12 months of categorized
  transactions — generated for a 5,000-customer sample with a fixed random
  seed (`src/config.py`).
- **Heuristic**: the six opportunity labels are rule-based weak supervision,
  not observed outcomes.

None of this is real customer financial data, and the agent is explicitly
restricted from giving individualized investment or tax advice.

## Repo layout

```
src/
  config.py                       paths, seed, tax brackets, constants
  data/                           UCI ingestion + synthetic overlay + SQLite load
  features/                       SQL queries, feature builder, heuristic labels
  models/                         training + diagnostics
  agent/                          tool functions, tool schemas, Claude agent
  pipeline.py                     CLI entrypoint
notebooks/demo.ipynb              executed narrative walkthrough
tests/                            pytest suite (19 tests, no API key required)
reports/                          metrics.json, diagnostic figures, agent briefs
```

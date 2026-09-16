"""End-to-end CLI: build data -> build features -> train models -> generate a
per-customer agent brief.

    python -m src.pipeline build-data
    python -m src.pipeline build-features
    python -m src.pipeline train
    python -m src.pipeline brief --customer-id 42
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.agent.copilot_agent import FinancialOpportunityAgent
from src.config import BRIEFS_DIR, DATA_PROCESSED, MODELS_DIR, RNG_SEED
from src.data import build_customers, generate_synthetic, load_db
from src.features import build_features
from src.features.heuristic_labels import OPPORTUNITY_LABELS, add_heuristic_labels, simulate_snapshot_noise
from src.models.train_opportunity_models import CATEGORICAL_COLS, load_dataset

SCORE_THRESHOLD = 0.5


def cmd_build_data(_args) -> None:
    build_customers.main()
    generate_synthetic.main()
    load_db.main()


def cmd_build_features(_args) -> None:
    build_features.main()


def cmd_train(_args) -> None:
    from src.models import train_opportunity_models

    train_opportunity_models.main()


def _load_customer_row(customer_id: int) -> pd.Series:
    features = pd.read_csv(DATA_PROCESSED / "feature_table.csv")
    row = features.loc[features["customer_id"] == customer_id]
    if row.empty:
        raise SystemExit(f"customer_id {customer_id} not found in feature_table.csv")
    return row.iloc[0]


def score_customer(customer_id: int) -> tuple[pd.Series, list[dict]]:
    """Score one customer against every trained opportunity model, applying
    the same snapshot-noise simulation used at training time so scoring
    conditions match what the models were evaluated under.
    """
    df = load_dataset()  # noised, label columns attached, same as training
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)
    row = df_encoded.loc[df_encoded["customer_id"] == customer_id]
    if row.empty:
        raise SystemExit(f"customer_id {customer_id} not found")

    opportunities = []
    for label in OPPORTUNITY_LABELS:
        bundle = joblib.load(MODELS_DIR / f"{label}.joblib")
        model, cols = bundle["model"], bundle["feature_columns"]
        X = row.reindex(columns=cols, fill_value=np.nan)
        proba = float(model.predict_proba(X)[:, 1][0])
        opportunities.append({"label": label, "probability": round(proba, 4), "flagged": proba >= SCORE_THRESHOLD})

    clean_row = _load_customer_row(customer_id)
    return clean_row, sorted(opportunities, key=lambda o: o["probability"], reverse=True)


def _customer_profile_for_agent(row: pd.Series) -> dict:
    return {
        "customer_id": int(row["customer_id"]),
        "age": int(row["age"]),
        "job": row["job"],
        "annual_income": float(row["est_annual_income"]),
        "marginal_tax_rate": float(row["marginal_tax_rate"]),
        "checking_balance": float(row["checking_balance"]),
        "savings_balance": float(row["savings_balance"]),
        "savings_apy": float(row["savings_apy"]),
        "avg_monthly_spend": float(row["avg_monthly_spend"]),
        "has_401k": bool(row["has_401k"]),
        "retirement_contribution_rate": float(row["retirement_contribution_rate"]),
        "employer_match_rate": float(row["employer_match_rate"]),
        "employer_match_cap": float(row["employer_match_cap"]),
        "has_credit_card": bool(row["has_credit_card"]),
        "credit_card_balance": float(row["credit_card_balance"]),
        "credit_card_apr": float(row["credit_card_apr"]),
        "has_brokerage": bool(row["has_brokerage"]),
        "harvestable_loss": float(row["harvestable_loss"]),
    }


def cmd_brief(args) -> None:
    customer_id = args.customer_id
    row, opportunities = score_customer(customer_id)
    flagged = [o for o in opportunities if o["flagged"]]

    agent = FinancialOpportunityAgent()
    result = agent.generate_brief(_customer_profile_for_agent(row), flagged or opportunities[:3])

    print("=" * 70)
    print(f"Customer {customer_id} -- flagged opportunities:")
    for o in opportunities:
        marker = "*" if o["flagged"] else " "
        print(f"  [{marker}] {o['label']:<24} p={o['probability']:.3f}")
    print("-" * 70)
    print(result.brief)
    print("=" * 70)
    print(f"Tool calls made: {len(result.tool_calls)} | API round-trips: {result.n_api_calls}")

    out = {
        "customer_id": customer_id,
        "opportunities": opportunities,
        "brief": result.brief,
        "tool_calls": [{"name": t.name, "input": t.input, "output": t.output} for t in result.tool_calls],
        "n_api_calls": result.n_api_calls,
    }
    out_path = BRIEFS_DIR / f"customer_{customer_id}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Financial Opportunity Copilot pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-data").set_defaults(func=cmd_build_data)
    sub.add_parser("build-features").set_defaults(func=cmd_build_features)
    sub.add_parser("train").set_defaults(func=cmd_train)

    brief_parser = sub.add_parser("brief")
    brief_parser.add_argument("--customer-id", type=int, required=True)
    brief_parser.set_defaults(func=cmd_brief)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

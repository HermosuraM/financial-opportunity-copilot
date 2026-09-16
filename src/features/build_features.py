"""Run the SQL feature queries in queries.sql against the SQLite DB, join them
with the customers/accounts tables, and write one feature-table row per
customer to data/processed/feature_table.csv.
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED, DB_PATH

QUERIES_PATH = Path(__file__).with_name("queries.sql")


def load_named_queries() -> dict[str, str]:
    text = QUERIES_PATH.read_text()
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    # parts = ['', name1, sql1, name2, sql2, ...]
    return {name: sql.strip() for name, sql in zip(parts[1::2], parts[2::2])}


def build_feature_table() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        queries = load_named_queries()

        spend_by_cat = pd.read_sql(queries["monthly_spend_by_category"], conn)
        spend_wide = spend_by_cat.pivot(index="customer_id", columns="category", values="avg_monthly_amount")
        spend_wide.columns = [f"avg_spend_{c}" for c in spend_wide.columns]

        sub_trend = pd.read_sql(queries["subscription_trend"], conn).set_index("customer_id")
        holdings_summary = pd.read_sql(queries["holdings_summary"], conn).set_index("customer_id")
        total_spend = pd.read_sql(queries["total_monthly_spend"], conn).set_index("customer_id")

        customers = pd.read_sql("SELECT * FROM customers", conn).set_index("customer_id")
        accounts = pd.read_sql("SELECT * FROM accounts", conn).set_index("customer_id")
    finally:
        conn.close()

    features = (
        customers.join(accounts, how="left")
        .join(spend_wide, how="left")
        .join(sub_trend, how="left")
        .join(holdings_summary, how="left")
        .join(total_spend, how="left")
    )

    spend_cols = [c for c in features.columns if c.startswith("avg_spend_")]
    features[spend_cols] = features[spend_cols].fillna(0)
    features["avg_subscription_mom_growth"] = features["avg_subscription_mom_growth"].fillna(0)
    for col in ["total_market_value", "total_cost_basis", "total_unrealized_gain_loss", "harvestable_loss", "num_lots"]:
        features[col] = features[col].fillna(0)

    features["emergency_fund_months"] = features["savings_balance"] / features["avg_monthly_spend"].replace(0, pd.NA)
    features["emergency_fund_months"] = features["emergency_fund_months"].fillna(0)

    return features.reset_index()


def main() -> None:
    features = build_feature_table()
    out_path = DATA_PROCESSED / "feature_table.csv"
    features.to_csv(out_path, index=False)
    print(f"Wrote feature table: {features.shape[0]:,} rows x {features.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    main()

import sqlite3

import pandas as pd

from src.features.build_features import build_feature_table


def test_build_feature_table_joins_and_aggregates(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)

    customers = pd.DataFrame({"customer_id": [1, 2], "job": ["admin.", "technician"]})
    accounts = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "savings_balance": [6000.0, 100.0],
            "checking_balance": [1000.0, 200.0],
        }
    )
    # customer 1: flat subscription spend; customer 2: growing subscription spend.
    transactions = pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
            "month": ["2025-01", "2025-02", "2025-03"] * 4,
            "category": ["subscriptions"] * 6 + ["groceries"] * 6,
            "amount": [10, 10, 10, 10, 15, 22.5, 300, 300, 300, 100, 100, 100],
        }
    )
    holdings = pd.DataFrame(
        {"customer_id": [1], "ticker": ["VOO"], "shares": [10.0], "cost_basis": [1000.0], "market_value": [800.0]}
    )

    customers.to_sql("customers", conn, index=False)
    accounts.to_sql("accounts", conn, index=False)
    transactions.to_sql("transactions", conn, index=False)
    holdings.to_sql("holdings", conn, index=False)
    conn.close()

    monkeypatch.setattr("src.features.build_features.DB_PATH", db_path)

    features = build_feature_table().set_index("customer_id")

    assert features.loc[1, "harvestable_loss"] == 200.0
    assert features.loc[2, "harvestable_loss"] == 0.0

    # customer 2's subscription spend grows ~50% MoM; customer 1's is flat.
    assert features.loc[2, "avg_subscription_mom_growth"] > features.loc[1, "avg_subscription_mom_growth"]
    assert features.loc[1, "avg_subscription_mom_growth"] == 0.0

    # emergency_fund_months = savings_balance / avg_monthly_spend (groceries + subscriptions).
    assert features.loc[1, "avg_monthly_spend"] == 310.0
    assert features.loc[1, "emergency_fund_months"] == 6000.0 / 310.0

"""Synthesize the account/transaction/holdings overlay that no public dataset
carries: balances, retirement contributions, brokerage lots, and 12 months of
categorized transactions. Sampled down to N_SYNTHETIC_CUSTOMERS real profiles
(see src/config.py) and keyed by customer_id so it joins cleanly onto the real
UCI-derived customer table.
"""
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, N_SYNTHETIC_CUSTOMERS, RNG_SEED

SPEND_CATEGORIES = ["groceries", "dining", "utilities", "subscriptions", "transfers", "shopping"]
CATEGORY_SHARE = {  # base share of monthly discretionary spend
    "groceries": 0.28,
    "dining": 0.16,
    "utilities": 0.18,
    "subscriptions": 0.08,
    "transfers": 0.15,
    "shopping": 0.15,
}
TICKERS = ["VOO", "AAPL", "MSFT", "AMZN", "BND", "VXUS", "TSLA", "NVDA"]


def _rng():
    return np.random.default_rng(RNG_SEED)


def sample_customers(customers: pd.DataFrame) -> pd.DataFrame:
    rng = _rng()
    n = min(N_SYNTHETIC_CUSTOMERS, len(customers))
    idx = rng.choice(customers.index, size=n, replace=False)
    return customers.loc[idx].sort_values("customer_id").reset_index(drop=True)


def generate_accounts(customers: pd.DataFrame) -> pd.DataFrame:
    rng = _rng()
    n = len(customers)
    income = customers["est_annual_income"].to_numpy()
    monthly_income = income / 12

    checking_balance = np.clip(rng.normal(0.35, 0.15, n), 0.05, 1.2) * monthly_income
    savings_balance = np.clip(rng.lognormal(mean=np.log(np.maximum(monthly_income, 1) * 1.2), sigma=0.9, size=n), 0, None)
    savings_apy = np.where(rng.random(n) < 0.75, rng.uniform(0.001, 0.005, n), rng.uniform(0.04, 0.048, n))

    has_401k = rng.random(n) < np.where(customers["job"].isin(["unemployed", "student", "retired"]), 0.05, 0.72)
    contribution_rate = np.where(has_401k, np.clip(rng.normal(0.05, 0.035, n), 0.0, 0.20), 0.0)
    employer_match_rate = np.full(n, 0.50)
    employer_match_cap = np.full(n, 0.06)
    retirement_balance = np.where(
        has_401k,
        np.clip(rng.lognormal(mean=np.log(np.maximum(income, 1) * 0.6), sigma=1.0, size=n), 0, None),
        0.0,
    )

    has_credit_card = rng.random(n) < 0.65
    credit_card_balance = np.where(
        has_credit_card, np.clip(rng.lognormal(mean=np.log(1500), sigma=1.1, size=n), 0, 30_000), 0.0
    )
    credit_card_apr = np.where(has_credit_card, rng.uniform(0.1799, 0.2699, n), 0.0)

    has_brokerage = rng.random(n) < 0.30

    accounts = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy(),
            "checking_balance": checking_balance.round(2),
            "savings_balance": savings_balance.round(2),
            "savings_apy": savings_apy.round(4),
            "has_401k": has_401k,
            "retirement_contribution_rate": contribution_rate.round(3),
            "employer_match_rate": employer_match_rate,
            "employer_match_cap": employer_match_cap,
            "retirement_balance": retirement_balance.round(2),
            "has_credit_card": has_credit_card,
            "credit_card_balance": credit_card_balance.round(2),
            "credit_card_apr": credit_card_apr.round(4),
            "has_brokerage": has_brokerage,
            "monthly_income": monthly_income.round(2),
        }
    )
    return accounts


def generate_transactions(accounts: pd.DataFrame) -> pd.DataFrame:
    rng = _rng()
    months = pd.period_range("2025-01", periods=12, freq="M").astype(str)
    rows = []
    n = len(accounts)

    # ~18% of customers show creeping subscription spend (a real fintech
    # pattern: forgotten trials/add-ons stacking month over month).
    creeping = rng.random(n) < 0.18
    monthly_income = accounts["monthly_income"].to_numpy()
    discretionary = np.clip(monthly_income * rng.uniform(0.35, 0.55, n), 200, None)

    for cat in SPEND_CATEGORIES:
        base_amt = discretionary * CATEGORY_SHARE[cat]
        for m_idx, month in enumerate(months):
            noise = rng.normal(1.0, 0.08, n)
            drift = np.where(creeping & (cat == "subscriptions"), 1.0 + 0.045 * m_idx, 1.0)
            amt = np.clip(base_amt * noise * drift, 0, None).round(2)
            rows.append(
                pd.DataFrame(
                    {
                        "customer_id": accounts["customer_id"].to_numpy(),
                        "month": month,
                        "category": cat,
                        "amount": amt,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def generate_holdings(accounts: pd.DataFrame) -> pd.DataFrame:
    rng = _rng()
    brokerage_customers = accounts.loc[accounts["has_brokerage"], "customer_id"].to_numpy()
    rows = []
    for cid in brokerage_customers:
        n_lots = rng.integers(1, 4)
        tickers = rng.choice(TICKERS, size=n_lots, replace=False)
        for ticker in tickers:
            shares = round(float(rng.uniform(1, 40)), 2)
            cost_basis = round(float(shares * rng.uniform(50, 400)), 2)
            # 35% of lots are underwater (tax-loss-harvest candidates)
            move = rng.uniform(-0.35, 0.10) if rng.random() < 0.35 else rng.uniform(-0.05, 0.45)
            market_value = round(cost_basis * (1 + move), 2)
            rows.append(
                {
                    "customer_id": cid,
                    "ticker": ticker,
                    "shares": shares,
                    "cost_basis": cost_basis,
                    "market_value": market_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    customers = pd.read_csv(DATA_PROCESSED / "customers.csv")
    sampled = sample_customers(customers)

    accounts = generate_accounts(sampled)
    transactions = generate_transactions(accounts)
    holdings = generate_holdings(accounts)

    sampled.to_csv(DATA_PROCESSED / "customers_sample.csv", index=False)
    accounts.to_csv(DATA_PROCESSED / "accounts.csv", index=False)
    transactions.to_csv(DATA_PROCESSED / "transactions.csv", index=False)
    holdings.to_csv(DATA_PROCESSED / "holdings.csv", index=False)

    print(f"customers_sample: {len(sampled):,} rows")
    print(f"accounts:         {len(accounts):,} rows")
    print(f"transactions:     {len(transactions):,} rows")
    print(f"holdings:         {len(holdings):,} rows")


if __name__ == "__main__":
    main()

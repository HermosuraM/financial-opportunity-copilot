"""Weak-supervision opportunity labels.

There is no ground-truth "this customer took action X" outcome available (no
public dataset has one), so we bootstrap labels from deterministic financial
rules -- a standard weak-supervision pattern. Labels are computed from the
*clean* feature table; the models in src/models/train_opportunity_models.py
are then trained against a *noised* snapshot of the same features (see
`simulate_snapshot_noise`) to mimic a real batch-scoring pipeline that sees
slightly stale balances and occasional missing account links rather than the
live truth the rule uses. See README "Modeling approach" for the rationale.
"""
import numpy as np
import pandas as pd

from src.config import EMERGENCY_FUND_TARGET_MONTHS, TARGET_HIGH_YIELD_APY

SUBSCRIPTION_CREEP_THRESHOLD = 0.03
TAX_LOSS_HARVEST_THRESHOLD = 500.0
HIGH_INTEREST_APR_THRESHOLD = 0.15


def add_heuristic_labels(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()

    df["opp_high_yield_savings"] = (
        (df["savings_balance"] > 3 * df["avg_monthly_spend"].replace(0, 1))
        & (df["savings_apy"] < TARGET_HIGH_YIELD_APY)
    ).astype(int)

    df["opp_emergency_fund_gap"] = (df["emergency_fund_months"] < EMERGENCY_FUND_TARGET_MONTHS).astype(int)

    df["opp_retirement_gap"] = (
        df["has_401k"].astype(bool)
        & (df["retirement_contribution_rate"] < df["employer_match_cap"])
    ).astype(int)

    df["opp_high_interest_debt"] = (
        (df["credit_card_balance"] > 0) & (df["credit_card_apr"] > HIGH_INTEREST_APR_THRESHOLD)
    ).astype(int)

    df["opp_tax_loss_harvest"] = (
        df["has_brokerage"].astype(bool) & (df["harvestable_loss"] > TAX_LOSS_HARVEST_THRESHOLD)
    ).astype(int)

    df["opp_subscription_creep"] = (df["avg_subscription_mom_growth"] > SUBSCRIPTION_CREEP_THRESHOLD).astype(int)

    return df


OPPORTUNITY_LABELS = [
    "opp_high_yield_savings",
    "opp_emergency_fund_gap",
    "opp_retirement_gap",
    "opp_high_interest_debt",
    "opp_tax_loss_harvest",
    "opp_subscription_creep",
]

# Magnitude columns that come from a data feed (balances, spend, holdings) and
# so are subject to snapshot staleness / sync gaps in a real pipeline. Rates
# and boolean flags (APY, has_401k, ...) are account metadata the bank sets
# directly and are treated as always current.
NOISY_FEED_COLUMNS = [
    "checking_balance",
    "savings_balance",
    "retirement_balance",
    "credit_card_balance",
    "total_market_value",
    "total_cost_basis",
    "total_unrealized_gain_loss",
    "harvestable_loss",
    "avg_monthly_spend",
    "avg_subscription_mom_growth",
    "avg_spend_groceries",
    "avg_spend_dining",
    "avg_spend_utilities",
    "avg_spend_subscriptions",
    "avg_spend_transfers",
    "avg_spend_shopping",
]


def simulate_snapshot_noise(
    df: pd.DataFrame, rng: np.random.Generator, noise_sigma: float = 0.25, missing_rate: float = 0.12
) -> pd.DataFrame:
    """Corrupt feed-derived columns to mimic a T-1 batch snapshot: multiplicative
    drift on every value plus a fraction of missing links (not yet re-synced).
    Labels are computed upstream from the clean data, so models trained on this
    noised view have to tolerate realistic staleness rather than reading the
    rule off exact inputs. `emergency_fund_months` is a ratio *derived* from
    two noised columns, so it is recomputed here instead of reusing the clean
    value -- otherwise the exact clean ratio would leak the answer back in.
    """
    noisy = df.copy()
    for col in NOISY_FEED_COLUMNS:
        if col not in noisy.columns:
            continue
        values = noisy[col].to_numpy(dtype=float)
        values = values * rng.normal(1.0, noise_sigma, size=len(values))
        missing_mask = rng.random(len(values)) < missing_rate
        values[missing_mask] = np.nan
        noisy[col] = values

    if {"savings_balance", "avg_monthly_spend"}.issubset(noisy.columns):
        noisy["emergency_fund_months"] = noisy["savings_balance"] / noisy["avg_monthly_spend"].replace(0, np.nan)

    return noisy

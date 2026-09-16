import numpy as np
import pandas as pd

from src.features.heuristic_labels import NOISY_FEED_COLUMNS, add_heuristic_labels, simulate_snapshot_noise


def _base_row(**overrides):
    row = {
        "savings_balance": 1000.0,
        "avg_monthly_spend": 2000.0,
        "savings_apy": 0.001,
        "emergency_fund_months": 0.5,
        "has_401k": True,
        "retirement_contribution_rate": 0.03,
        "employer_match_cap": 0.06,
        "credit_card_balance": 500.0,
        "credit_card_apr": 0.22,
        "has_brokerage": True,
        "harvestable_loss": 1000.0,
        "avg_subscription_mom_growth": 0.10,
    }
    row.update(overrides)
    return row


def test_high_yield_savings_flags_idle_low_apy_cash():
    df = pd.DataFrame([_base_row(savings_balance=8000.0, avg_monthly_spend=2000.0, savings_apy=0.001)])
    out = add_heuristic_labels(df)
    assert out.loc[0, "opp_high_yield_savings"] == 1


def test_high_yield_savings_not_flagged_when_apy_already_good():
    df = pd.DataFrame([_base_row(savings_balance=8000.0, avg_monthly_spend=2000.0, savings_apy=0.05)])
    out = add_heuristic_labels(df)
    assert out.loc[0, "opp_high_yield_savings"] == 0


def test_emergency_fund_gap_boundary():
    below = pd.DataFrame([_base_row(emergency_fund_months=2.9)])
    above = pd.DataFrame([_base_row(emergency_fund_months=3.1)])
    assert add_heuristic_labels(below).loc[0, "opp_emergency_fund_gap"] == 1
    assert add_heuristic_labels(above).loc[0, "opp_emergency_fund_gap"] == 0


def test_retirement_gap_requires_401k():
    no_401k = pd.DataFrame([_base_row(has_401k=False, retirement_contribution_rate=0.0)])
    with_401k = pd.DataFrame([_base_row(has_401k=True, retirement_contribution_rate=0.0)])
    assert add_heuristic_labels(no_401k).loc[0, "opp_retirement_gap"] == 0
    assert add_heuristic_labels(with_401k).loc[0, "opp_retirement_gap"] == 1


def test_tax_loss_harvest_requires_brokerage_and_threshold():
    no_brokerage = pd.DataFrame([_base_row(has_brokerage=False, harvestable_loss=5000.0)])
    small_loss = pd.DataFrame([_base_row(has_brokerage=True, harvestable_loss=100.0)])
    big_loss = pd.DataFrame([_base_row(has_brokerage=True, harvestable_loss=5000.0)])
    assert add_heuristic_labels(no_brokerage).loc[0, "opp_tax_loss_harvest"] == 0
    assert add_heuristic_labels(small_loss).loc[0, "opp_tax_loss_harvest"] == 0
    assert add_heuristic_labels(big_loss).loc[0, "opp_tax_loss_harvest"] == 1


def test_simulate_snapshot_noise_preserves_columns_and_introduces_missingness():
    df = pd.DataFrame(
        {col: np.full(500, 100.0) for col in NOISY_FEED_COLUMNS if col not in ("avg_monthly_spend",)}
    )
    df["avg_monthly_spend"] = 100.0
    df["savings_balance"] = 100.0
    rng = np.random.default_rng(0)
    noisy = simulate_snapshot_noise(df, rng, noise_sigma=0.2, missing_rate=0.2)

    assert set(noisy.columns) == set(df.columns) | {"emergency_fund_months"}
    assert not np.allclose(noisy["savings_balance"].dropna(), 100.0)
    assert noisy["savings_balance"].isna().mean() > 0
    # emergency_fund_months must be recomputed from the noised inputs, not left at a stale value.
    valid = noisy.dropna(subset=["savings_balance", "avg_monthly_spend"])
    expected = valid["savings_balance"] / valid["avg_monthly_spend"]
    assert np.allclose(valid["emergency_fund_months"], expected)

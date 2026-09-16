import pytest

from src.agent.tools import (
    debt_payoff_priority,
    emergency_fund_gap,
    high_yield_savings_gain,
    retirement_contribution_headroom,
    retirement_match_gap,
    tax_loss_harvest_estimate,
)


def test_retirement_match_gap_full_unclaimed():
    result = retirement_match_gap(salary=100_000, contribution_rate=0.0, employer_match_rate=0.5, employer_match_cap=0.06)
    assert result["unclaimed_annual_match_dollars"] == pytest.approx(3000.0)
    assert result["full_annual_match_dollars"] == pytest.approx(3000.0)
    assert result["current_annual_match_dollars"] == pytest.approx(0.0)


def test_retirement_match_gap_fully_captured():
    result = retirement_match_gap(salary=100_000, contribution_rate=0.10, employer_match_rate=0.5, employer_match_cap=0.06)
    assert result["unclaimed_annual_match_dollars"] == pytest.approx(0.0)
    assert result["current_annual_match_dollars"] == pytest.approx(3000.0)


def test_high_yield_savings_gain():
    result = high_yield_savings_gain(balance=10_000, current_apy=0.002, target_apy=0.045)
    assert result["current_annual_interest"] == pytest.approx(20.0)
    assert result["target_annual_interest"] == pytest.approx(450.0)
    assert result["estimated_annual_gain"] == pytest.approx(430.0)


def test_emergency_fund_gap_shortfall():
    result = emergency_fund_gap(avg_monthly_spend=2000, savings_balance=1000, target_months=3)
    assert result["target_balance"] == pytest.approx(6000.0)
    assert result["shortfall"] == pytest.approx(5000.0)
    assert result["months_covered"] == pytest.approx(0.5)


def test_emergency_fund_gap_no_shortfall():
    result = emergency_fund_gap(avg_monthly_spend=2000, savings_balance=10_000, target_months=3)
    assert result["shortfall"] == pytest.approx(0.0)


def test_tax_loss_harvest_estimate_caps_at_3000():
    result = tax_loss_harvest_estimate(harvestable_loss=5000, annual_income=60_000)
    assert result["offset_this_year"] == pytest.approx(3000.0)
    assert result["carryforward_loss"] == pytest.approx(2000.0)
    assert result["marginal_tax_rate"] == pytest.approx(0.22)
    assert result["estimated_tax_savings"] == pytest.approx(660.0)


def test_tax_loss_harvest_estimate_under_cap():
    result = tax_loss_harvest_estimate(harvestable_loss=1200, annual_income=20_000)
    assert result["offset_this_year"] == pytest.approx(1200.0)
    assert result["carryforward_loss"] == pytest.approx(0.0)


def test_debt_payoff_priority_orders_by_apr():
    debts = [
        {"name": "student_loan", "balance": 10_000, "apr": 0.05},
        {"name": "credit_card", "balance": 3_000, "apr": 0.24},
        {"name": "car_loan", "balance": 8_000, "apr": 0.07},
    ]
    result = debt_payoff_priority(debts)
    assert result["payoff_order"] == ["credit_card", "car_loan", "student_loan"]
    assert result["highest_apr_debt"] == "credit_card"
    assert result["estimated_annual_interest_cost"] == pytest.approx(10_000 * 0.05 + 3_000 * 0.24 + 8_000 * 0.07)


def test_debt_payoff_priority_ignores_zero_balance():
    debts = [{"name": "paid_off", "balance": 0, "apr": 0.20}, {"name": "active", "balance": 500, "apr": 0.10}]
    result = debt_payoff_priority(debts)
    assert result["payoff_order"] == ["active"]


def test_retirement_contribution_headroom():
    result = retirement_contribution_headroom(salary=80_000, contribution_rate=0.10)
    assert result["current_annual_contribution"] == pytest.approx(8000.0)
    assert result["remaining_headroom"] == pytest.approx(23_500 - 8000.0)

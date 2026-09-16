"""Deterministic financial-math tools the agent calls instead of doing math
itself. Keeping every numeric claim behind a callable, unit-tested function
is the whole point: the LLM decides *what to say and prioritize*, never
*what the numbers are*.
"""
from dataclasses import dataclass

from src.config import IRS_401K_EMPLOYEE_LIMIT_2025, TARGET_HIGH_YIELD_APY
from src.data.build_customers import marginal_tax_bracket


def retirement_match_gap(salary: float, contribution_rate: float, employer_match_rate: float, employer_match_cap: float) -> dict:
    """Annual employer 401k match dollars left unclaimed if contribution_rate < employer_match_cap."""
    capped_rate = min(contribution_rate, employer_match_cap)
    unclaimed_rate = max(employer_match_cap - contribution_rate, 0.0)
    unclaimed_dollars = round(salary * unclaimed_rate * employer_match_rate, 2)
    full_match_dollars = round(salary * employer_match_cap * employer_match_rate, 2)
    current_match_dollars = round(salary * capped_rate * employer_match_rate, 2)
    return {
        "unclaimed_annual_match_dollars": unclaimed_dollars,
        "current_annual_match_dollars": current_match_dollars,
        "full_annual_match_dollars": full_match_dollars,
    }


def high_yield_savings_gain(balance: float, current_apy: float, target_apy: float = TARGET_HIGH_YIELD_APY) -> dict:
    """Estimated annual interest gain from moving idle cash to a high-yield account."""
    current_annual_interest = round(balance * current_apy, 2)
    target_annual_interest = round(balance * target_apy, 2)
    return {
        "current_annual_interest": current_annual_interest,
        "target_annual_interest": target_annual_interest,
        "estimated_annual_gain": round(target_annual_interest - current_annual_interest, 2),
    }


def emergency_fund_gap(avg_monthly_spend: float, savings_balance: float, target_months: int = 3) -> dict:
    """Dollar shortfall to reach `target_months` of expenses in savings."""
    target_balance = round(avg_monthly_spend * target_months, 2)
    shortfall = round(max(target_balance - savings_balance, 0.0), 2)
    months_covered = round(savings_balance / avg_monthly_spend, 1) if avg_monthly_spend else 0.0
    return {"target_balance": target_balance, "shortfall": shortfall, "months_covered": months_covered}


def tax_loss_harvest_estimate(harvestable_loss: float, annual_income: float) -> dict:
    """Estimated tax savings from realizing harvestable capital losses this year.
    Simplified: assumes losses offset ordinary income up to the $3,000/yr IRS
    cap, taxed at the filer's marginal bracket; excess losses carry forward.
    """
    rate = marginal_tax_bracket(annual_income)
    offset_this_year = min(harvestable_loss, 3_000.0)
    carryforward = round(max(harvestable_loss - 3_000.0, 0.0), 2)
    estimated_savings = round(offset_this_year * rate, 2)
    return {
        "marginal_tax_rate": rate,
        "offset_this_year": round(offset_this_year, 2),
        "carryforward_loss": carryforward,
        "estimated_tax_savings": estimated_savings,
    }


@dataclass
class Debt:
    name: str
    balance: float
    apr: float


def debt_payoff_priority(debts: list[dict]) -> dict:
    """Avalanche-method payoff order (highest APR first) and annual interest
    cost of doing nothing, for prioritizing high_interest_debt opportunities.
    """
    parsed = [Debt(**d) for d in debts if d.get("balance", 0) > 0]
    ordered = sorted(parsed, key=lambda d: d.apr, reverse=True)
    total_annual_interest = round(sum(d.balance * d.apr for d in parsed), 2)
    return {
        "payoff_order": [d.name for d in ordered],
        "estimated_annual_interest_cost": total_annual_interest,
        "highest_apr_debt": ordered[0].name if ordered else None,
        "highest_apr": ordered[0].apr if ordered else None,
    }


def retirement_contribution_headroom(salary: float, contribution_rate: float) -> dict:
    """Dollars of IRS annual 401k contribution room remaining for the year."""
    current_contribution = round(salary * contribution_rate, 2)
    headroom = round(max(IRS_401K_EMPLOYEE_LIMIT_2025 - current_contribution, 0.0), 2)
    return {
        "current_annual_contribution": current_contribution,
        "irs_limit": IRS_401K_EMPLOYEE_LIMIT_2025,
        "remaining_headroom": headroom,
    }


TOOL_FUNCTIONS = {
    "retirement_match_gap": retirement_match_gap,
    "high_yield_savings_gain": high_yield_savings_gain,
    "emergency_fund_gap": emergency_fund_gap,
    "tax_loss_harvest_estimate": tax_loss_harvest_estimate,
    "debt_payoff_priority": debt_payoff_priority,
    "retirement_contribution_headroom": retirement_contribution_headroom,
}

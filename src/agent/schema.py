"""Anthropic tool-use JSON schemas for the functions in tools.py."""

TOOL_SCHEMAS = [
    {
        "name": "retirement_match_gap",
        "description": "Compute the annual employer 401k match dollars a customer is leaving unclaimed because their contribution rate is below the employer match cap.",
        "input_schema": {
            "type": "object",
            "properties": {
                "salary": {"type": "number"},
                "contribution_rate": {"type": "number", "description": "Current employee contribution rate, e.g. 0.03 for 3%"},
                "employer_match_rate": {"type": "number", "description": "e.g. 0.5 for a 50% match"},
                "employer_match_cap": {"type": "number", "description": "Contribution rate up to which the employer matches, e.g. 0.06 for 6%"},
            },
            "required": ["salary", "contribution_rate", "employer_match_rate", "employer_match_cap"],
        },
    },
    {
        "name": "high_yield_savings_gain",
        "description": "Estimate the annual interest gained by moving a savings balance from its current APY to a target high-yield APY.",
        "input_schema": {
            "type": "object",
            "properties": {
                "balance": {"type": "number"},
                "current_apy": {"type": "number", "description": "e.g. 0.003 for 0.3%"},
                "target_apy": {"type": "number", "description": "Optional; defaults to the current market high-yield benchmark"},
            },
            "required": ["balance", "current_apy"],
        },
    },
    {
        "name": "emergency_fund_gap",
        "description": "Compute the dollar shortfall between a customer's current savings and a target number of months of expenses covered.",
        "input_schema": {
            "type": "object",
            "properties": {
                "avg_monthly_spend": {"type": "number"},
                "savings_balance": {"type": "number"},
                "target_months": {"type": "integer", "description": "Optional; defaults to 3"},
            },
            "required": ["avg_monthly_spend", "savings_balance"],
        },
    },
    {
        "name": "tax_loss_harvest_estimate",
        "description": "Estimate tax savings from realizing a customer's harvestable brokerage losses this year, using their marginal tax bracket.",
        "input_schema": {
            "type": "object",
            "properties": {
                "harvestable_loss": {"type": "number"},
                "annual_income": {"type": "number"},
            },
            "required": ["harvestable_loss", "annual_income"],
        },
    },
    {
        "name": "debt_payoff_priority",
        "description": "Given a list of debts (name, balance, apr), return the avalanche-method payoff order and the annual interest cost of not paying them down.",
        "input_schema": {
            "type": "object",
            "properties": {
                "debts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "balance": {"type": "number"},
                            "apr": {"type": "number"},
                        },
                        "required": ["name", "balance", "apr"],
                    },
                }
            },
            "required": ["debts"],
        },
    },
    {
        "name": "retirement_contribution_headroom",
        "description": "Compute remaining IRS annual 401k employee contribution headroom for the year given salary and current contribution rate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "salary": {"type": "number"},
                "contribution_rate": {"type": "number"},
            },
            "required": ["salary", "contribution_rate"],
        },
    },
]

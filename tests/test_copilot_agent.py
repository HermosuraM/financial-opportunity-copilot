"""Exercises the tool-call loop in FinancialOpportunityAgent against a fake
Anthropic client, so the orchestration logic is verified without a network
call or a real API key.
"""
from types import SimpleNamespace

from src.agent.copilot_agent import FinancialOpportunityAgent


class FakeBlock:
    def __init__(self, type, **kwargs):
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeClient:
    """Returns a tool_use turn once, then a final text turn."""

    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)
        self.calls = 0

    def _create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            tool_block = FakeBlock(
                "tool_use",
                id="tool_1",
                name="emergency_fund_gap",
                input={"avg_monthly_spend": 2000, "savings_balance": 1000, "target_months": 3},
            )
            return SimpleNamespace(stop_reason="tool_use", content=[tool_block])
        text_block = FakeBlock("text", text="You're $5,000.00 short of a 3-month emergency fund.")
        return SimpleNamespace(stop_reason="end_turn", content=[text_block])


def test_generate_brief_runs_tool_then_returns_text(monkeypatch):
    agent = FinancialOpportunityAgent.__new__(FinancialOpportunityAgent)
    agent.model = "claude-sonnet-5"
    agent.client = FakeClient()

    result = agent.generate_brief(
        customer={"customer_id": 1, "avg_monthly_spend": 2000, "savings_balance": 1000},
        opportunities=[{"label": "opp_emergency_fund_gap", "probability": 0.9}],
    )

    assert "5,000" in result.brief or "5000" in result.brief
    assert result.n_api_calls == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "emergency_fund_gap"
    assert result.tool_calls[0].output["shortfall"] == 5000.0


def test_generate_brief_stops_without_tool_use():
    agent = FinancialOpportunityAgent.__new__(FinancialOpportunityAgent)
    agent.model = "claude-sonnet-5"

    class DirectAnswerClient:
        def __init__(self):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            return SimpleNamespace(
                stop_reason="end_turn", content=[FakeBlock("text", text="No action needed this month.")]
            )

    agent.client = DirectAnswerClient()
    result = agent.generate_brief(customer={}, opportunities=[])
    assert result.brief == "No action needed this month."
    assert result.n_api_calls == 1
    assert result.tool_calls == []

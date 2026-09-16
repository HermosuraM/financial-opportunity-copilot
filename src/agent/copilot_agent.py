"""The agentic layer: Claude decides which ML-flagged opportunities to
surface and how to explain them, but every dollar figure it states comes from
a deterministic tool call (src/agent/tools.py) -- the model is never allowed
to do the arithmetic itself.
"""
import json
import os
from dataclasses import dataclass, field

import anthropic

from src.agent.schema import TOOL_SCHEMAS
from src.agent.tools import TOOL_FUNCTIONS

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOOL_ITERATIONS = 6

SYSTEM_PROMPT = """You are a financial-wellbeing copilot embedded in a bank/brokerage app.

You receive: (1) a customer's profile and account snapshot, (2) a list of \
financial opportunities an ML model flagged for that customer, each with a \
probability score.

Your job:
1. Decide which 1-3 opportunities are actually worth surfacing this month -- \
   prioritize by a mix of estimated dollar impact and model confidence. Do \
   not list every flagged opportunity; a customer ignores a wall of nudges.
2. For every dollar figure, percentage, or dated estimate you state, call the \
   matching tool. Never compute or estimate a number yourself -- if no tool \
   covers it, describe it qualitatively instead of guessing a number.
3. Write a short brief: one headline priority, 1-2 supporting notes, plain \
   English, warm but not salesy. No more than ~150 words of prose.
4. Never recommend a specific security, fund, or investment product, and \
   never give individualized investment or tax advice -- surface general \
   financial-planning actions (e.g. "consider moving idle cash to a \
   higher-yield account") and suggest the customer confirm specifics with an \
   advisor or tax professional.
5. End with one line: "This is informational only, not individualized \
   financial or tax advice."
"""


@dataclass
class ToolCallLog:
    name: str
    input: dict
    output: dict


@dataclass
class AgentResult:
    brief: str
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    n_api_calls: int = 0


class FinancialOpportunityAgent:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate_brief(self, customer: dict, opportunities: list[dict]) -> AgentResult:
        user_content = (
            "Customer profile and account snapshot:\n"
            f"{json.dumps(customer, indent=2)}\n\n"
            "ML-flagged opportunities (label, probability):\n"
            f"{json.dumps(opportunities, indent=2)}"
        )
        messages = [{"role": "user", "content": user_content}]
        tool_calls: list[ToolCallLog] = []
        n_api_calls = 0

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            n_api_calls += 1

            if response.stop_reason != "tool_use":
                brief = "".join(block.text for block in response.content if block.type == "text")
                return AgentResult(brief=brief.strip(), tool_calls=tool_calls, n_api_calls=n_api_calls)

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                func = TOOL_FUNCTIONS[block.name]
                output = func(**block.input)
                tool_calls.append(ToolCallLog(name=block.name, input=block.input, output=output))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)}
                )
            messages.append({"role": "user", "content": tool_results})

        return AgentResult(
            brief="[Agent did not converge to a final answer within the tool-call budget.]",
            tool_calls=tool_calls,
            n_api_calls=n_api_calls,
        )

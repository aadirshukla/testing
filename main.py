#!/usr/bin/env python3
"""A tiny, dependency-free agent for learning the basic architecture.

Try:
    python main.py "calculate 12 * (3 + 4)"
    python main.py "what time is it?"
    python main.py "remember that my favorite color is blue"
    python main.py --interactive

This example deliberately uses a rule-based model instead of an API. That
keeps the important agent mechanics visible and makes the program runnable
offline. Later, MockModel.decide() can be replaced with a real LLM call.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 1. Tools: actions the agent is allowed to take
# ---------------------------------------------------------------------------


ToolFunction = Callable[..., str]


@dataclass
class Tool:
    name: str
    description: str
    function: ToolFunction


def current_time() -> str:
    """Return the current local time."""
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def safe_calculate(expression: str) -> str:
    """Evaluate basic arithmetic without using unsafe eval()."""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            left, right = visit(node.left), visit(node.right)
            return allowed_operators[type(node.op)](left, right)
        raise ValueError("only basic arithmetic is supported")

    try:
        tree = ast.parse(expression, mode="eval")
        result = visit(tree)
        return str(result)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculation error: {exc}"


def remember(facts: dict[str, str], fact: str) -> str:
    """Store one simple fact in the agent's in-memory state."""
    fact = fact.strip().rstrip(".")
    if not fact:
        return "I need a fact to remember."
    key = f"fact_{len(facts) + 1}"
    facts[key] = fact
    return f"Remembered: {fact}"


def recall(facts: dict[str, str]) -> str:
    if not facts:
        return "I do not remember anything yet."
    return "I remember: " + "; ".join(facts.values())


# ---------------------------------------------------------------------------
# 2. Model: chooses either a tool or a final answer
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    kind: str  # "tool" or "final"
    message: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)


class MockModel:
    """A tiny stand-in for an LLM.

    A real model would receive the conversation and tool descriptions, then
    return a structured tool call or a final response. The Agent class below
    does not need to know how that decision was produced.
    """

    def decide(self, user_request: str, tool_results: list[str]) -> Decision:
        request = user_request.strip()
        lower = request.lower()

        # After a tool has run, this demo turns its result into a final answer.
        if tool_results:
            return Decision(kind="final", message=tool_results[-1])

        if "time" in lower:
            return Decision(kind="tool", tool_name="clock")

        if lower.startswith("calculate ") or lower.startswith("what is "):
            expression = re.sub(r"^(calculate|what is)\s+", "", request, flags=re.I)
            return Decision(
                kind="tool", tool_name="calculator", tool_args={"expression": expression}
            )

        if lower.startswith("remember "):
            fact = re.sub(r"^remember\s+(that\s+)?", "", request, flags=re.I)
            return Decision(kind="tool", tool_name="memory_write", tool_args={"fact": fact})

        if "remember" in lower or "recall" in lower:
            return Decision(kind="tool", tool_name="memory_read")

        return Decision(
            kind="final",
            message=(
                "I am a small teaching agent. Try: calculate 2 + 2, "
                "what time is it, or remember that I like tea."
            ),
        )


# ---------------------------------------------------------------------------
# 3. Agent loop: gives the model tools, executes decisions, and repeats
# ---------------------------------------------------------------------------


class Agent:
    def __init__(self, model: MockModel, tools: list[Tool], verbose: bool = False):
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.memory: dict[str, str] = {}
        self.verbose = verbose

    def run(self, user_request: str, max_steps: int = 5) -> str:
        """Run one task through the observe -> decide -> act loop."""
        tool_results: list[str] = []

        for step in range(1, max_steps + 1):
            decision = self.model.decide(user_request, tool_results)

            if self.verbose:
                print(f"[step {step}] model decision: {decision}")

            if decision.kind == "final":
                return decision.message

            if decision.kind != "tool" or decision.tool_name not in self.tools:
                return "The model requested an unavailable tool."

            tool = self.tools[decision.tool_name]
            try:
                result = tool.function(**decision.tool_args)
            except TypeError as exc:
                result = f"Tool error: {exc}"

            tool_results.append(result)
            if self.verbose:
                print(f"[step {step}] {tool.name} -> {result}")

        return "Stopped: maximum number of agent steps reached."


def build_agent(verbose: bool = False) -> Agent:
    # The registry is the boundary between what the model may request and
    # what the program is actually willing to execute.
    facts: dict[str, str] = {}
    tools = [
        Tool("clock", "Get the current local time", current_time),
        Tool("calculator", "Perform basic arithmetic", safe_calculate),
        Tool("memory_write", "Remember one fact", lambda fact: remember(facts, fact)),
        Tool("memory_read", "Recall remembered facts", lambda: recall(facts)),
    ]
    agent = Agent(MockModel(), tools, verbose=verbose)
    # Keep the demo memory shared across calls to agent.run().
    agent.memory = facts
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the educational mini-agent")
    parser.add_argument("request", nargs="?", help="one request to send to the agent")
    parser.add_argument("--interactive", action="store_true", help="start a chat loop")
    parser.add_argument("--verbose", action="store_true", help="show internal agent steps")
    args = parser.parse_args()

    agent = build_agent(verbose=args.verbose)
    if args.interactive or not args.request:
        print("Mini-agent. Type 'quit' to exit.")
        while True:
            try:
                request = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if request.lower() in {"quit", "exit"}:
                break
            if request:
                print(f"agent> {agent.run(request)}")
    else:
        print(agent.run(args.request))


if __name__ == "__main__":
    main()

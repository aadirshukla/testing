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

# Standard-library modules used for the command-line interface, parsing math,
# reading the current time, matching request text, and defining data types.
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


# A tool is any callable that accepts arguments and returns text for the user.
ToolFunction = Callable[..., str]


@dataclass
class Tool:
    # The model uses the name to request a tool, the description documents it,
    # and the function contains the actual operation that will be executed.
    name: str
    description: str
    function: ToolFunction


def current_time() -> str:
    """Return the current local time."""
    # astimezone() converts the system time into the machine's local timezone.
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def safe_calculate(expression: str) -> str:
    """Evaluate basic arithmetic without using unsafe eval()."""
    # Map only approved AST operator types to their arithmetic implementations.
    # Restricting the map prevents arbitrary Python code from being executed.
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def visit(node: ast.AST) -> float:
        # Recursively walk the parsed expression and accept only safe node types.
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
        # Parse the input as one expression, then evaluate it with the safe walker.
        tree = ast.parse(expression, mode="eval")
        result = visit(tree)
        return str(result)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        # Return a readable error instead of allowing invalid input to crash the agent.
        return f"Calculation error: {exc}"


def remember(facts: dict[str, str], fact: str) -> str:
    """Store one simple fact in the agent's in-memory state."""
    # Normalize whitespace and remove a trailing period before saving the fact.
    fact = fact.strip().rstrip(".")
    if not fact:
        return "I need a fact to remember."
    # Use sequential keys so multiple facts can be stored in the dictionary.
    key = f"fact_{len(facts) + 1}"
    facts[key] = fact
    return f"Remembered: {fact}"


def recall(facts: dict[str, str]) -> str:
    # Handle an empty memory separately so the response is meaningful.
    if not facts:
        return "I do not remember anything yet."
    # Join all saved values into one response for the user.
    return "I remember: " + "; ".join(facts.values())


# ---------------------------------------------------------------------------
# 2. Model: chooses either a tool or a final answer
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    # kind determines whether the agent should call a tool or finish with text.
    kind: str  # "tool" or "final"
    # message is used for final answers; tool_name and tool_args describe a call.
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
        # Normalize the request once so each routing rule can inspect it easily.
        request = user_request.strip()
        lower = request.lower()

        # After a tool has run, this demo turns its result into a final answer.
        if tool_results:
            return Decision(kind="final", message=tool_results[-1])

        # Route time-related requests to the clock tool.
        if "time" in lower:
            return Decision(kind="tool", tool_name="clock")

        # Extract the arithmetic expression and route it to the calculator tool.
        if lower.startswith("calculate ") or lower.startswith("what is "):
            expression = re.sub(r"^(calculate|what is)\s+", "", request, flags=re.I)
            return Decision(
                kind="tool", tool_name="calculator", tool_args={"expression": expression}
            )

        # Remove the command prefix before sending the fact to memory_write.
        if lower.startswith("remember "):
            fact = re.sub(r"^remember\s+(that\s+)?", "", request, flags=re.I)
            return Decision(kind="tool", tool_name="memory_write", tool_args={"fact": fact})

        # Requests containing these words read the facts already in memory.
        if "remember" in lower or "recall" in lower:
            return Decision(kind="tool", tool_name="memory_read")

        # Explain the supported commands when no routing rule matches.
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
        # Store the model and convert the tool list into a name-to-tool registry.
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        # This dictionary is available for future agent state; build_agent wires
        # it to the facts dictionary used by the memory tools.
        self.memory: dict[str, str] = {}
        self.verbose = verbose

    def run(self, user_request: str, max_steps: int = 5) -> str:
        """Run one task through the observe -> decide -> act loop."""
        # Keep the outputs from tools so the model can turn the latest result
        # into a final response on the next loop iteration.
        tool_results: list[str] = []

        # Limit the number of model/tool cycles so a faulty model cannot loop forever.
        for step in range(1, max_steps + 1):
            # Ask the model what to do with the original request and prior results.
            decision = self.model.decide(user_request, tool_results)

            if self.verbose:
                print(f"[step {step}] model decision: {decision}")

            if decision.kind == "final":
                return decision.message

            # Reject malformed decisions and requests for tools outside the registry.
            if decision.kind != "tool" or decision.tool_name not in self.tools:
                return "The model requested an unavailable tool."

            # Look up the requested tool and execute it with the model-supplied arguments.
            tool = self.tools[decision.tool_name]
            try:
                result = tool.function(**decision.tool_args)
            except TypeError as exc:
                # Convert argument mismatches into normal tool output.
                result = f"Tool error: {exc}"

            # Save the result for the next decision and optionally show it while debugging.
            tool_results.append(result)
            if self.verbose:
                print(f"[step {step}] {tool.name} -> {result}")

        return "Stopped: maximum number of agent steps reached."


def build_agent(verbose: bool = False) -> Agent:
    # The registry is the boundary between what the model may request and
    # what the program is actually willing to execute.
    facts: dict[str, str] = {}
    # Register the concrete functions that the model is allowed to call.
    tools = [
        Tool("clock", "Get the current local time", current_time),
        Tool("calculator", "Perform basic arithmetic", safe_calculate),
        Tool("memory_write", "Remember one fact", lambda fact: remember(facts, fact)),
        Tool("memory_read", "Recall remembered facts", lambda: recall(facts)),
    ]
    # Combine the rule-based model with the registered tools.
    agent = Agent(MockModel(), tools, verbose=verbose)
    # Keep the demo memory shared across calls to agent.run().
    agent.memory = facts
    return agent


def main() -> None:
    # Define the supported command-line arguments and parse the user's input.
    parser = argparse.ArgumentParser(description="Run the educational mini-agent")
    parser.add_argument("request", nargs="?", help="one request to send to the agent")
    parser.add_argument("--interactive", action="store_true", help="start a chat loop")
    parser.add_argument("--verbose", action="store_true", help="show internal agent steps")
    args = parser.parse_args()

    # Create one agent instance so interactive mode can retain its memory.
    agent = build_agent(verbose=args.verbose)
    if args.interactive or not args.request:
        # Start a repeated prompt when no one-shot request was supplied.
        print("Mini-agent. Type 'quit' to exit.")
        while True:
            try:
                request = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                # Exit cleanly when input ends or the user presses Ctrl+C.
                print()
                break
            if request.lower() in {"quit", "exit"}:
                # Stop the interactive session on either exit command.
                break
            if request:
                # Ignore blank lines and send non-empty requests to the agent.
                print(f"agent> {agent.run(request)}")
    else:
        # In one-shot mode, run the supplied request and print its answer.
        print(agent.run(args.request))


# Only invoke the CLI when this file is run directly, not when it is imported.
if __name__ == "__main__":
    main()

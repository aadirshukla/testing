# Mini Agent Architecture

This is a tiny, dependency-free agent designed for learning. It does not
call an AI API yet, so it runs offline and you can see every moving part.

## Run it

```bash
python3 main.py "calculate 12 * (3 + 4)"
python3 main.py --verbose "what time is it?"
python3 main.py --interactive
```

In interactive mode, try:

```text
remember that my favorite color is blue
what do you remember?
calculate 2 ** 8
```

## Architecture

```text
user request
     |
     v
  MockModel.decide() -----> final answer
     |
     v
  tool call
     |
     v
  Agent executes tool
     |
     v
  tool result goes back to the model
```

The important pieces are:

- `Tool`: an action the agent is allowed to use.
- `MockModel`: chooses a tool or returns a final answer. In a production agent,
  this is where an LLM call would go.
- `Agent.run()`: the agent loop that coordinates decisions and tool execution.
- `build_agent()`: the tool registry and dependency setup.
- `--verbose`: exposes the internal decisions and tool results.

## Suggested next experiment

Add a `weather` or `read_file` tool, then update `MockModel.decide()` so it can
choose that tool. Once this makes sense, replace `MockModel` with a real model
client while leaving the `Agent` and tools mostly unchanged.

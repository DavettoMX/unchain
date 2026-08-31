# Unchain

Lightweight, zero-framework agentic runtime for [llama.cpp](https://github.com/ggml-org/llama.cpp) servers.

Unchain drives llama-server's **native HTTP API** directly — no OpenAI-compatible layer, no LLM SDKs, no agent frameworks. Deterministic tool calling is enforced server-side through grammar-constrained JSON decoding, and all execution state is persisted to PostgreSQL for local containerized deployments.

## Why Unchain

Most agent frameworks add layers of abstraction between your code and the model. Unchain removes them:

- **No SDK bloat** — a hand-rolled async HTTP/1.1 client (`src/unchain/transport/`) talks directly to llama-server over plain sockets. The only third-party packages are `pydantic` and `asyncpg`.
- **Structurally impossible hallucinated tools** — every generation request carries a JSON Schema grammar. The server's sampler can only decode into a valid tool call or a final answer. No post-hoc parsing, no retry loops, no "oops the model said something weird".
- **Full execution history** — sessions, multi-turn messages, run traces, and per-step timing land in PostgreSQL. Resume any conversation, replay any run.
- **Local-first** — designed for `llama-server` on localhost. No API keys, no cloud calls, no telemetry.

## The output envelope

Every generation is grammar-constrained to exactly one of two shapes:

```json
{"tool": {"name": "<tool>", "arguments": {...}}}
```

```json
{"final": "<answer>"}
```

The tool name is enum-restricted to registered tools, and each `arguments` object is validated against the tool's Pydantic model before execution. If the model somehow produces invalid output, the agent feeds the error back and retries (up to a configurable limit).

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Quickstart

### 1. Start the backend

Postgres only:

```bash
docker compose up -d unchain-db
uv run unchain db migrate
```

Postgres + llama-server (put a GGUF under `models/`):

```bash
MODEL_FILE=qwen2.5-7b-instruct-q4_k_m.gguf docker compose --profile llm up -d
uv run unchain db migrate
```

### 2. Run

```bash
uv run unchain health                        # check llama-server and postgres
uv run unchain run "what time is it?"        # one-shot agent execution
uv run unchain chat --new-session            # multi-turn REPL (persisted)
uv run unchain chat --session <uuid>         # resume a session
uv run unchain tools list                    # list registered tools
uv run unchain tools sync                    # persist tool definitions to postgres
```

### Configuration

| Env var | Default | Purpose |
|---|---|---|
| `UNCHAIN_SERVER` | `http://127.0.0.1:8080` | llama-server base URL |
| `UNCHAIN_DSN` | `postgresql://unchain:unchain@127.0.0.1:5432/unchain` | Postgres DSN |

All env vars can be overridden with CLI flags (`--server`, `--dsn`).

## Library usage

```python
import asyncio

from pydantic import BaseModel

from unchain import Agent, AgentConfig, LlamaServer, ToolRegistry


class WordCountArgs(BaseModel):
    text: str


registry = ToolRegistry()


@registry.register("word_count", args_model=WordCountArgs)
def word_count(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())


async def main() -> None:
    async with LlamaServer("http://127.0.0.1:8080") as server:
        agent = Agent(server, registry, config=AgentConfig(max_iterations=8))
        result = await agent.run("how many words are in 'the quick brown fox'?")
        print(result.final)


asyncio.run(main())
```

### Tool handlers

- **Sync handlers** are offloaded to a worker thread automatically.
- **Async handlers** are awaited directly.
- Handlers never crash the loop — validation errors, timeouts, and exceptions are captured and returned as tool results the model can react to.

### Streaming events

`Agent.run()` accepts an `on_event` callback that streams structured events as the loop progresses:

| Event | When |
|---|---|
| `IterationStarted` | Each loop iteration begins |
| `TextDelta` | A token chunk arrives (streaming mode) |
| `ToolCallStarted` | A tool call is parsed and about to execute |
| `ToolCallFinished` | A tool call completed (with result or error) |
| `FinalAnswer` | The model produced a final answer |
| `AgentFailed` | The run failed or hit max iterations |

### Extending

- **Custom prompt template** — pass `AgentConfig(template=PromptTemplate(...))` to override ChatML markers for other model families.
- **Custom persona** — pass `AgentConfig(persona="You are ...")` to replace the default system prompt header.
- **Extra tools from the CLI** — `--tool-module my_pkg.tools:registry` loads a `ToolRegistry`, or `--tool-module my_pkg.tools:my_handler` loads a single callable.

### Built-in tools

Unchain ships with three demo tools (used by default in the CLI):

| Tool | Description |
|---|---|
| `calculator` | Safe AST-based arithmetic evaluator (`+ - * / // % **`) |
| `current_time` | Returns the current UTC time in ISO-8601 |
| `echo` | Echoes text back verbatim (smoke test) |

## Architecture

```
src/unchain/
├── transport/           # Zero-dep async HTTP/1.1 client
│   ├── http.py          #   Raw sockets, keep-alive, chunked transfer, streaming
│   ├── sse.py           #   Incremental Server-Sent Events parser
│   └── llama_server.py  #   /completion, /tokenize, /detokenize, /health, /props
├── tools/
│   ├── registry.py      #   Pydantic → JSON Schema envelope, grammar fingerprint
│   └── executor.py      #   Typed dispatch, timeouts, error capture
├── agent/
│   ├── envelope.py      #   Output contract parsing + validation
│   ├── history.py       #   Prompt templates, system prompt, token budgeting
│   └── loop.py          #   generate → parse → execute → feed back
├── state/
│   ├── pool.py          #   asyncpg pool with startup retry
│   ├── migrate.py       #   Forward-only SQL migrations
│   └── repo.py          #   Sessions, messages, runs, steps, tools
├── cli/
│   ├── __main__.py      #   CLI entry point (run, chat, tools, db, health)
│   └── repl.py          #   Multi-turn chat REPL
├── models.py            #   Shared domain types (Message, StepRecord, etc.)
└── builtins.py          #   Built-in demo tools (calculator, clock, echo)
```

### How the agent loop works

1. The user prompt is appended to the conversation history.
2. A system prompt is assembled with tool descriptions and the output contract.
3. The full transcript is formatted with ChatML markers (or a custom template) and sent to `/completion` with the envelope JSON Schema as a grammar constraint.
4. The response is parsed into an `Envelope` — either a tool call or a final answer.
5. If it's a tool call: arguments are validated with Pydantic, the handler runs (with a timeout), and the result is fed back as a tool message. Go to step 3.
6. If it's a final answer: the loop exits and returns the result.
7. If the model produces invalid output: an error message is fed back and the model retries (up to `max_invalid_retries`).

### Database schema

The state layer uses four main tables plus a tool registry:

- **sessions** — conversation containers with metadata
- **messages** — ordered multi-turn transcript (role, content, tool_name)
- **runs** — one agent execution within a session (status, timing, final answer)
- **run_steps** — per-step trace (generate, tool_call, tool_result, error) with timing and payloads
- **tools** — versioned tool definitions synced from the registry

Migrations are forward-only and run with `uv run unchain db migrate`.

## Testing

```bash
uv run pytest                                # unit tests (no services needed)
uv run ruff check . && uv run mypy           # lint + strict typecheck

# integration (opt-in, requires live services)
UNCHAIN_DSN=postgresql://unchain:unchain@127.0.0.1:5432/unchain \
UNCHAIN_SERVER=http://127.0.0.1:8080 \
uv run pytest -m integration
```

## License

MIT

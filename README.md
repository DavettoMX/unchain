# Unchain

Lightweight, zero-framework agentic runtime for [llama.cpp](https://github.com/ggml-org/llama.cpp) servers.

Unchain drives llama-server's **native HTTP API** directly — no OpenAI-compatible layer, no LLM SDKs, no agent frameworks. Deterministic tool calling is enforced server-side through grammar-constrained JSON decoding, and all execution state is persisted to PostgreSQL for local containerized deployments.

## Design

- **Local-first** — talks straight to `llama-server` over plain HTTP on localhost.
- **Minimal dependencies** — pure-stdlib `asyncio` HTTP/1.1 + SSE client (hand-rolled, `src/unchain/transport/`); the only third-party packages are `pydantic` and `asyncpg`.
- **Deterministic tool calling** — instead of trusting the model, every generation request carries a `json_schema` grammar built from the registered tools, so decoding can only produce a valid tool call or a final answer. Arguments are re-validated with Pydantic before execution.
- **Strict output boundaries, zero latency overhead** — the constraint lives in the server's sampler, not in post-hoc parsing or retry loops.
- **Persistent state** — sessions, multi-turn messages, run history, per-step traces, and dynamic tool definitions in PostgreSQL.

## The output envelope

Each generation is grammar-constrained to exactly one of:

```json
{"tool": {"name": "<tool>", "arguments": {...}}}
{"final": "<answer>"}
```

The tool name is enum-restricted to registered tools (hallucinated tools are impossible), and each `arguments` schema is derived from the tool's Pydantic model.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

### Start the backend

Postgres only:

```bash
docker compose up -d unchain-db
uv run unchain db migrate
```

Postgres + llama-server (put a GGUF under `models/`):

```bash
MODEL_FILE=qwen2.5-7b-instruct-q4_k_m.gguf docker compose --profile llm up -d
```

### CLI

```bash
uv run unchain health                        # check llama-server and postgres
uv run unchain run "what time is it?"        # one-shot agent execution
uv run unchain chat --new-session            # multi-turn REPL (persisted)
uv run unchain chat --session <uuid>         # resume a session
uv run unchain tools list                    # list registered tools
uv run unchain tools sync                    # persist tool definitions to postgres
```

Configuration via flags or environment:

| Env var | Default | Purpose |
|---|---|---|
| `UNCHAIN_SERVER` | `http://127.0.0.1:8080` | llama-server base URL |
| `UNCHAIN_DSN` | `postgresql://unchain:unchain@127.0.0.1:5432/unchain` | Postgres DSN |

### Library

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

Sync handlers are offloaded to a worker thread; async handlers are awaited directly. Handlers never crash the loop — validation errors, timeouts, and exceptions come back as tool results the model can react to.

### Extending

- **Custom prompt template**: pass `AgentConfig(template=PromptTemplate(...))` — defaults are ChatML markers; override for other model families.
- **Extra tools from the CLI**: `--tool-module my_pkg.tools:registry` (a `ToolRegistry`) or `--tool-module my_pkg.tools:my_handler`.
- **Events**: `Agent.run(..., on_event=callback)` streams `IterationStarted`, `TextDelta`, `ToolCallStarted/Finished`, `FinalAnswer`, `AgentFailed` events.

## Architecture

```
src/unchain/
├── transport/          # zero-dep async HTTP: raw sockets, chunked + SSE parsing
│   ├── http.py         #   HTTP/1.1 client, keep-alive, streaming
│   ├── sse.py          #   incremental SSE parser
│   └── llama_server.py #   /completion, /tokenize, /embedding, /health, /props
├── tools/
│   ├── registry.py     #   Pydantic → JSON Schema envelope, grammar fingerprint
│   └── executor.py     #   typed dispatch, timeouts, error capture
├── agent/
│   ├── envelope.py     #   output contract parsing + validation
│   ├── history.py      #   prompt templates, system prompt, token budgeting
│   └── loop.py         #   generate → parse → execute → feed back
├── state/
│   ├── pool.py         #   asyncpg pool with startup retry
│   ├── migrate.py      #   forward-only SQL migrations
│   └── repo.py         #   sessions, messages, runs, steps, tools
└── cli/                # run, chat REPL, tools, db, health
```

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

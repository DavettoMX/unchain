"""Command line entry point: run, chat, tools, db, health."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
from typing import Any
from uuid import UUID

from unchain.agent import (
    Agent,
    AgentConfig,
    AgentEvent,
    FinalAnswer,
    ToolCallFinished,
    ToolCallStarted,
)
from unchain.builtins import default_registry
from unchain.cli.repl import repl
from unchain.state.pool import DEFAULT_DSN, connect
from unchain.state.repo import PgRunRecorder, SessionRepo, ToolRepo
from unchain.tools.registry import ToolRegistry
from unchain.transport.llama_server import LlamaServer, SamplingParams

DEFAULT_SERVER = "http://127.0.0.1:8080"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unchain",
        description="Lightweight agentic runtime for llama.cpp servers.",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("UNCHAIN_SERVER", DEFAULT_SERVER),
        help=f"llama-server base URL (default: {DEFAULT_SERVER}, env: UNCHAIN_SERVER)",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("UNCHAIN_DSN", DEFAULT_DSN),
        help="PostgreSQL DSN (env: UNCHAIN_DSN)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="one-shot agent execution")
    run_cmd.add_argument("prompt")
    run_cmd.add_argument("--session", type=UUID, default=None)
    run_cmd.add_argument("--max-iters", type=int, default=16)
    run_cmd.add_argument("--temperature", type=float, default=None)
    run_cmd.add_argument("--max-tokens", type=int, default=512)
    run_cmd.add_argument("--no-stream", action="store_true")
    run_cmd.add_argument("--tool-module", action="append", default=[])

    chat_cmd = sub.add_parser("chat", help="interactive multi-turn REPL")
    chat_cmd.add_argument("--session", type=UUID, default=None)
    chat_cmd.add_argument("--new-session", action="store_true")
    chat_cmd.add_argument("--max-iters", type=int, default=16)
    chat_cmd.add_argument("--tool-module", action="append", default=[])

    tools_cmd = sub.add_parser("tools", help="inspect or sync the tool registry")
    tools_cmd.add_argument("action", choices=["list", "sync"], default="list", nargs="?")

    db_cmd = sub.add_parser("db", help="database operations")
    db_cmd.add_argument("action", choices=["migrate", "status"], default="migrate", nargs="?")

    sub.add_parser("health", help="check llama-server (and postgres) availability")
    return parser


def _load_extra_tools(registry: ToolRegistry, modules: list[str]) -> None:
    for spec in modules:
        module_name, _, attr = spec.partition(":")
        module = importlib.import_module(module_name)
        target = getattr(module, attr) if attr else module
        if isinstance(target, ToolRegistry):

            def _add(tool: Any) -> None:
                registry.add(
                    tool.name,
                    description=tool.description,
                    args_model=tool.args_model,
                    handler=tool.handler,
                )

            for tool in target.definitions():
                _add(tool)
        elif callable(target):
            registry.add(module_name.rsplit(".", 1)[-1], handler=target)
        else:
            raise ValueError(f"tool module target is not callable: {spec}")


def _registry_from_args(args: argparse.Namespace) -> ToolRegistry:
    registry = default_registry()
    _load_extra_tools(registry, getattr(args, "tool_module", []))
    return registry


def _agent_from_args(args: argparse.Namespace, registry: ToolRegistry) -> Agent:
    sampling = SamplingParams(
        temperature=getattr(args, "temperature", None), max_tokens=getattr(args, "max_tokens", 512)
    )
    config = AgentConfig(
        max_iterations=getattr(args, "max_iters", 16),
        sampling=sampling,
        stream=not getattr(args, "no_stream", False),
    )
    server = LlamaServer(args.server)
    return Agent(server, registry, config=config)


async def _maybe_db(args: argparse.Namespace, *, required: bool) -> Any:
    try:
        return await connect(args.dsn, attempts=1 if not required else 8)
    except ConnectionError:
        if required:
            raise
        print(
            f"[warn] postgres unreachable at {args.dsn}; running without persistence",
            file=sys.stderr,
        )
        return None


async def cmd_run(args: argparse.Namespace) -> int:
    agent = _agent_from_args(args, _registry_from_args(args))
    session_repo = None
    recorder = None
    session_id = args.session
    history = []
    if session_id is not None:
        db = await _maybe_db(args, required=True)
        session_repo = SessionRepo(db)
        history = await session_repo.list_messages(session_id)
        recorder = PgRunRecorder(db)

    async def on_event(event: AgentEvent) -> None:
        match event:
            case ToolCallStarted(name, arguments):
                print(f"  -> tool {name}({arguments})", file=sys.stderr, flush=True)
            case ToolCallFinished(name, result):
                outcome = result.value if result.ok else f"ERROR: {result.error}"
                print(f"  <- {name} = {outcome!s:.300}", file=sys.stderr, flush=True)
            case FinalAnswer():
                pass
            case _:
                pass

    result = await agent.run(
        args.prompt,
        history=history,
        session_id=session_id,
        recorder=recorder,
        on_event=on_event,
    )
    if session_repo is not None and session_id is not None:
        await session_repo.append_all(session_id, result.history[len(history) :])
    if result.final is not None:
        print(result.final)
        return 0
    print(f"[agent stopped: {result.status}]", file=sys.stderr)
    return 1


async def cmd_chat(args: argparse.Namespace) -> int:
    agent = _agent_from_args(args, _registry_from_args(args))
    session_repo = None
    session_id = args.session
    if session_id is None and args.new_session:
        db = await _maybe_db(args, required=True)
        session_repo = SessionRepo(db)
        session = await session_repo.create()
        session_id = session.id
        print(f"new session: {session_id}", file=sys.stderr)
    elif session_id is not None:
        db = await _maybe_db(args, required=True)
        session_repo = SessionRepo(db)
    try:
        return await repl(
            agent,
            session_repo=session_repo,
            session_id=session_id,
        )
    finally:
        await agent.aclose()


async def cmd_tools(args: argparse.Namespace) -> int:
    registry = _registry_from_args(args)
    if args.action == "sync":
        db = await _maybe_db(args, required=True)
        await ToolRepo(db).sync_registry(registry)
        print(f"synced {len(registry)} tools to {args.dsn}")
        return 0
    if not registry.names():
        print("(no tools registered)")
        return 0
    for tool in registry.definitions():
        print(f"{tool.name:<16} {tool.description}")
    return 0


async def cmd_db(args: argparse.Namespace) -> int:
    from unchain.state.migrate import applied_versions, migrate

    db = await _maybe_db(args, required=True)
    if args.action == "status":
        print(f"applied migrations: {await applied_versions(db.pool)}")
    else:
        versions = await migrate(db.pool)
        print(f"applied migrations: {versions or '(none — already up to date)'}")
    await db.close()
    return 0


async def cmd_health(args: argparse.Namespace) -> int:
    ok = True
    server = LlamaServer(args.server)
    try:
        healthy = await server.health()
        print(f"llama-server {args.server}: {'ok' if healthy else 'unhealthy'}")
        ok = ok and healthy
    except Exception as exc:
        print(f"llama-server {args.server}: unreachable ({exc})")
        ok = False
    finally:
        await server.aclose()
    try:
        db = await connect(args.dsn, attempts=1)
        await db.fetchrow("SELECT 1")
        await db.close()
        print(f"postgres {args.dsn}: ok")
    except Exception as exc:
        print(f"postgres {args.dsn}: unreachable ({exc})")
        ok = False
    return 0 if ok else 1


async def dispatch(args: argparse.Namespace) -> int:
    handlers: dict[str, Any] = {
        "run": cmd_run,
        "chat": cmd_chat,
        "tools": cmd_tools,
        "db": cmd_db,
        "health": cmd_health,
    }
    handler = handlers[args.command]
    result = handler(args)
    if asyncio.iscoroutine(result):
        return int(await result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(dispatch(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

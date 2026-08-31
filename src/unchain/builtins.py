"""Bundled demo tools: a safe AST calculator and a UTC clock."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from unchain.tools.registry import ToolRegistry

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. (2 + 3) * 7.5")


class ClockArgs(BaseModel):
    pass


class EchoArgs(BaseModel):
    text: str = Field(description="Text to echo back verbatim")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression safely (no names, no calls)."""
    return str(_eval(ast.parse(expression, mode="eval").body))


def current_time() -> str:
    """Return the current time in UTC, ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def echo(text: str) -> str:
    """Echo the given text back (smoke-test tool)."""
    return text


def _eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, tuple(_BINARY)):
        return _BINARY[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, tuple(_UNARY)):
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.add(
        "calculator",
        description="Evaluate arithmetic expressions with + - * / // % ** and parentheses.",
        args_model=CalculatorArgs,
        handler=calculator,
    )
    registry.add(
        "current_time",
        description="Get the current UTC time in ISO-8601 format.",
        args_model=ClockArgs,
        handler=current_time,
    )
    registry.add(
        "echo",
        description="Echo text back verbatim. Useful for smoke tests.",
        args_model=EchoArgs,
        handler=echo,
    )
    return registry

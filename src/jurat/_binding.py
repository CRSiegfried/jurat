"""What an oracle binding call passes and returns.

A binding is an async function the caller supplies, one per batch key and one under
``default``. It takes the batch, a tuple of ``SourceCall``s in source id order, and returns a
``CallResult`` or a ``Failed`` outcome, or raises. Interpret is the one engine producer of a
``SourceCall`` and the one reader of a ``CallResult``. The three types are frozen and check
nothing when built: interpret tests a call result at the write, so a bad value fails its ref
or the call by the fault scope rather than raising inside the binding.
"""

from collections.abc import Awaitable, Callable
from dataclasses import KW_ONLY, dataclass
from typing import Any

from jurat._answer import AnswerKind


@dataclass(frozen=True, slots=True)
class RenderedQuestion:
    """A ``q:`` ref's question as asked: its kind, the filled wording and its labels in order,
    ``yes`` and ``no`` for ``noul``."""

    kind: AnswerKind
    wording: str
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceCall:
    """One source of a batch: its id, its payload, its needs' values in the form a read
    returns, keyed by each need as declared, the ``t:`` refs it provides and the ``q:`` refs
    it is asked, each in code-point order. ``needs`` is a fresh
    copy: changing it reaches nothing else."""

    id: str
    payload: Any
    needs: dict[str, Any]
    refs: tuple[str, ...]
    questions: dict[str, RenderedQuestion]


@dataclass(frozen=True, slots=True)
class CallResult:
    """What a binding returns for its batch: ``values``, a ``dict`` from each ref the batch
    provides to its value, a ``q:`` ref's a ``dict`` from label to probability, and the
    call's provenance, null when no model was called."""

    values: dict[str, Any]
    _: KW_ONLY
    snapshot: str | None = None
    elicitation: str | None = None
    prompt: str | None = None
    usage: dict[str, float] | None = None
    attempts: int | None = None


OracleBinding = Callable[[tuple[SourceCall, ...]], Awaitable[object]]
"""The type of an oracle binding; what it returns is tested at the write, so it is typed as
anything."""

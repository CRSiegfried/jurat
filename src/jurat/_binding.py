"""What an oracle binding call passes and returns.

A binding is an async function the caller supplies, one per oracle name and one under
``default``. It takes the batch, a tuple of ``SourceCall``s in source id order, and returns a
``CallResult`` or a ``Failed`` outcome, or raises. Interpret is the one engine producer of a
``SourceCall`` and the one reader of a ``CallResult``. The three types are frozen and check
nothing when built: interpret tests a call result at the write, so a bad value fails its ref
or the call by the fault scope rather than raising inside the binding.
"""

from collections.abc import Awaitable, Callable, Sized
from dataclasses import KW_ONLY, dataclass
from typing import Any

from jurat._answer import AnswerKind, ValueType


@dataclass(frozen=True, slots=True)
class RenderedQuestion:
    """A ``q:`` ref's question as asked: its kind, the filled wording and its labels' names in
    order, ``yes`` and ``no`` for ``noul`` and none for ``extraction``. ``descriptions``,
    keyword-only, is aligned with ``labels``: each label's description, or ``None`` where it
    declares none, so ``(None, None)`` for ``noul`` and ``()`` for ``extraction``; built with
    none, it is ``None`` for every label, and a non-empty value is kept as given, unchecked.
    An extraction question also gives, keyword-only, its ``value_type``, ``is_list``, ``text``,
    the ``needs`` key holding the text its spans index, and ``reference_date``, the rendered
    date the model resolves a date written without a year against, or ``None`` when the
    question declares none; a probability kind's question leaves them at ``None``, ``False``,
    ``None`` and ``None``."""

    kind: AnswerKind
    wording: str
    labels: tuple[str, ...]
    _: KW_ONLY
    descriptions: tuple[str | None, ...] = ()
    value_type: ValueType | None = None
    is_list: bool = False
    text: str | None = None
    reference_date: str | None = None

    def __post_init__(self) -> None:
        # Never left unaligned: built with none, each label is undescribed. A non-empty value
        # is kept as given, unchecked.
        if self.descriptions:
            return
        labels: object = self.labels
        if isinstance(labels, Sized):  # pyright: ignore[reportUnnecessaryIsInstance]
            object.__setattr__(self, "descriptions", (None,) * len(labels))


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
    provides to its value, a ``q:`` ref's a ``dict`` from label name to probability or, for an
    extraction question, exactly the keys ``value``, ``guess``, ``p_null``, ``confidence`` and
    ``span``; the call's provenance, null when no model was called; and ``truncated``, the
    ``q:`` refs whose answer came from a response the provider cut short, kept outside every
    hash; ``warnings``, the binding's own lines about the call, such as why it refused a ref
    with ``unsupported``, kept outside every hash; and ``precision``, the decimal places the
    provider rounds probabilities to, or ``None`` when unknown or not a decimal count, hashed
    on the call item."""

    values: dict[str, Any]
    _: KW_ONLY
    snapshot: str | None = None
    elicitation: str | None = None
    precision: int | None = None
    prompt: str | None = None
    usage: dict[str, float] | None = None
    attempts: int | None = None
    truncated: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


OracleBinding = Callable[[tuple[SourceCall, ...]], Awaitable[object]]
"""The type of an oracle binding; what it returns is tested at the write, so it is typed as
anything."""

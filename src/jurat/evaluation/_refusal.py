"""The evaluation jobs' one refusal class and its closed reasons."""

import enum
from typing import Any


class RefusalReason(enum.StrEnum):
    """Why an evaluation job refused."""

    unfinished = "unfinished"
    nothing_to_replay = "nothing-to-replay"
    wrong_design = "wrong-design"
    different_turn = "different-turn"
    invalid_record = "invalid-record"


class EvaluationRefusal(Exception):
    """An evaluation job's refusal: ``reason``, and each other attribute what the reason
    names, else ``None``. ``turn`` is the record hash of the record refused; ``error`` the
    ``InvalidRecord`` that caused it, also chained."""

    def __init__(
        self,
        message: str,
        *,
        reason: RefusalReason,
        key: list[dict[str, Any]] | None = None,
        turn: str | None = None,
        ref: str | None = None,
        source: str | None = None,
        point: int | None = None,
        path: str | None = None,
        group: str | None = None,
        split: tuple[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.key = key
        self.turn = turn
        self.ref = ref
        self.source = source
        self.point = point
        self.path = path
        self.group = group
        self.split = split
        self.error = error

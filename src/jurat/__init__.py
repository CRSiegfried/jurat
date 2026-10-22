"""jurat: a blackboard engine for plans built on a decision model.

Decide runs interpret, which gathers evidence from a decision model onto a
blackboard, and compile, which writes one plan that a guard checks. It returns
the plan, the verdict and a replayable record, and calls no tool: the caller
runs the plan. Replay reruns a record under any loaded design and reports how
the rerun differs. The core imports only the standard library.
"""

from importlib.metadata import version as _version

from jurat._answer import Answer, AnswerKind, ExtractionAnswer, ValueType
from jurat._binding import CallResult, RenderedQuestion, SourceCall
from jurat._decide import Decision, EntryRefusal, RunSettings, decide, decide_sync
from jurat._design import (
    Design,
    DesignFailure,
    Group,
    InvalidDesign,
    LoadedDesign,
    Parameter,
    PlanDeclaration,
    Policy,
    PolicyTable,
    Question,
    Requirement,
    Rule,
    Source,
    SourceRequirement,
    Table,
    Text,
    When,
)
from jurat._guard import GuardFailure, Verdict
from jurat._ledger import InvalidRecord, Record, read_record
from jurat._load import load
from jurat._markers import (
    Cause,
    CauseCode,
    Failed,
    PlanKind,
    Unasked,
    Unavailable,
    Undetermined,
    Unsure,
    VerdictKind,
)
from jurat._plan import Binding, Origin, Plan, Step
from jurat._replay import FindingKind, replay, replay_sync
from jurat._views import escalation, redact

__version__ = _version("jurat")

__all__ = [
    "Answer",
    "AnswerKind",
    "Binding",
    "CallResult",
    "Cause",
    "CauseCode",
    "Decision",
    "Design",
    "DesignFailure",
    "EntryRefusal",
    "ExtractionAnswer",
    "Failed",
    "FindingKind",
    "Group",
    "GuardFailure",
    "InvalidDesign",
    "InvalidRecord",
    "LoadedDesign",
    "Origin",
    "Parameter",
    "Plan",
    "PlanDeclaration",
    "PlanKind",
    "Policy",
    "PolicyTable",
    "Question",
    "Record",
    "RenderedQuestion",
    "Requirement",
    "Rule",
    "RunSettings",
    "Source",
    "SourceCall",
    "SourceRequirement",
    "Step",
    "Table",
    "Text",
    "Unasked",
    "Unavailable",
    "Undetermined",
    "Unsure",
    "ValueType",
    "Verdict",
    "VerdictKind",
    "When",
    "__version__",
    "decide",
    "decide_sync",
    "escalation",
    "load",
    "read_record",
    "redact",
    "replay",
    "replay_sync",
]

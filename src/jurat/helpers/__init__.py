"""The helpers subpackage: the helper library, the choice helpers and combinators that a
function rule or the policy calls with its read view; the design-check lints,
``dead_questions`` and ``unsure_to_execute``, which the consumer's tests run over a loaded
design; and the reference runner, ``run_plan``, which the caller runs after decide. None is in
the engine's API.

``HELPERS`` is the helper library's version, which the record's ``design`` item carries; it
counts the choice helpers and combinators alone, never a lint or the runner.

The choice helpers are in ``_choice``, the combinators in ``_combinator``, and the mechanics
each of them is built on in ``_call``. The lints are in ``_dead`` and ``_unsure``, over the
points both share in ``_points``.
"""

from jurat.helpers._choice import mass, pick_with_margin, pick_with_thresholds, size
from jurat.helpers._combinator import agree, collect, first, priority, unique
from jurat.helpers._dead import dead_questions
from jurat.helpers._runner import BindingNotFound, RunRefusal, run_plan
from jurat.helpers._unsure import unsure_to_execute

__all__ = [
    "BindingNotFound",
    "RunRefusal",
    "agree",
    "collect",
    "dead_questions",
    "first",
    "mass",
    "pick_with_margin",
    "pick_with_thresholds",
    "priority",
    "run_plan",
    "size",
    "unique",
    "unsure_to_execute",
]

HELPERS = 1

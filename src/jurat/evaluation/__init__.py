"""The evaluation subpackage: the evaluation jobs, which measure a design against past turns
through replay. None is in the engine's API.

``use_answers`` is do-and-diff's fifth form, another record's answers: it reruns one record's
turn with another record's calls answering, and returns replay's findings. Every job refuses
with ``EvaluationRefusal``, its ``reason`` a ``RefusalReason``.

The job is in ``_diff`` and the refusal in ``_refusal``.
"""

from jurat.evaluation._diff import use_answers, use_answers_sync
from jurat.evaluation._refusal import EvaluationRefusal, RefusalReason

__all__ = ["EvaluationRefusal", "RefusalReason", "use_answers", "use_answers_sync"]

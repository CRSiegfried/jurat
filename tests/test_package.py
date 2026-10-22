import subprocess
import sys
from importlib.metadata import version

import jurat


def test_version_matches_metadata() -> None:
    assert jurat.__version__ == version("jurat")


def test_core_imports_only_the_standard_library() -> None:
    # A fresh interpreter, so modules loaded by pytest or plugins don't count.
    probe = (
        "import sys; before = set(sys.modules); import jurat; "
        "names = {m.partition('.')[0] for m in set(sys.modules) - before}; "
        "print('\\n'.join(sorted(names - set(sys.stdlib_module_names) - {'jurat'})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == []


# The public API names.
PUBLIC = {
    "Design",
    "DesignFailure",
    "Group",
    "InvalidDesign",
    "LoadedDesign",
    "Parameter",
    "PlanDeclaration",
    "Policy",
    "PolicyTable",
    "Question",
    "Requirement",
    "Rule",
    "Source",
    "SourceRequirement",
    "When",
    "Table",
    "Text",
    "load",
    "Answer",
    "AnswerKind",
    "ExtractionAnswer",
    "ValueType",
    "Cause",
    "CauseCode",
    "Failed",
    "InvalidRecord",
    "PlanKind",
    "Record",
    "Unasked",
    "Unavailable",
    "Undetermined",
    "Unsure",
    "VerdictKind",
    "read_record",
    "CallResult",
    "SourceCall",
    "RenderedQuestion",
    "Plan",
    "Step",
    "Origin",
    "Binding",
    "ReadView",
    "Verdict",
    "GuardFailure",
    "RunSettings",
    "Decision",
    "EntryRefusal",
    "decide",
    "decide_sync",
    "escalation",
    "redact",
    "replay",
    "replay_sync",
    "FindingKind",
}
TYPE_CONSTRUCTORS = {
    "number",
    "integer",
    "string",
    "boolean",
    "bytes_",
    "label",
    "list_",
    "mapping",
    "record",
    "nullable",
}


def test_the_public_names_are_exactly_the_decisions() -> None:
    assert set(jurat.__all__) == {*PUBLIC, "__version__"}
    public = {n for n in vars(jurat) if not n.startswith("_")}
    # Submodules imported by tests appear as attributes; `types`, `helpers` and `evaluation`
    # are public subpackages outside the engine's API.
    assert public - {"types", "helpers", "evaluation"} == PUBLIC


EVALUATION = {"EvaluationRefusal", "RefusalReason", "use_answers", "use_answers_sync"}


def test_jurat_evaluation_exports_exactly_its_jobs_and_refusal() -> None:
    import jurat.evaluation

    assert "evaluation" not in jurat.__all__
    assert set(jurat.evaluation.__all__) == EVALUATION
    assert {n for n in vars(jurat.evaluation) if not n.startswith("_")} == EVALUATION


def test_jurat_types_exports_exactly_the_constructors() -> None:
    import jurat.types

    assert set(jurat.types.__all__) == TYPE_CONSTRUCTORS
    assert {n for n in vars(jurat.types) if not n.startswith("_")} == TYPE_CONSTRUCTORS


def test_plain_literals_and_lambdas_build_the_public_types() -> None:
    import typing_strict

    assert typing_strict.step.arguments == typing_strict.args
    assert typing_strict.source.needs == ("t:state",)

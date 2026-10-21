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
    # Submodules imported by tests appear as attributes; `types` and `helpers` are public
    # subpackages outside the engine's API.
    assert public - {"types", "helpers"} == PUBLIC


def test_jurat_types_exports_exactly_the_constructors() -> None:
    import jurat.types

    assert set(jurat.types.__all__) == TYPE_CONSTRUCTORS
    assert {n for n in vars(jurat.types) if not n.startswith("_")} == TYPE_CONSTRUCTORS

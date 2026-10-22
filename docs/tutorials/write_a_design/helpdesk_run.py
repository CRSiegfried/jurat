"""Load the tutorial's design, decide two turns offline, read each record back and replay it.

Run it from this directory with ``python helpdesk_run.py``. The binding below returns fixed answers,
so the script needs no model, key or network.
"""

import json
from collections.abc import Mapping
from typing import Any

from helpdesk_design import DESIGN
from jurat import CallResult, Plan, SourceCall, decide_sync, load, read_record, replay_sync

MESSAGES = [
    "Hi, my order A-1042 still hasn't arrived and it has been two weeks. Any news?",
    "Where is my parcel? I didn't keep the order number, sorry.",
]


async def fixed_answers(calls: tuple[SourceCall, ...]) -> CallResult:
    """Answer every question by its kind, the same way each time."""
    values: dict[str, Any] = {}
    for call in calls:
        for ref, question in call.questions.items():
            if question.kind == "noul":
                values[ref] = {"yes": 0.2, "no": 0.8}
            elif question.kind == "choice":
                first, *rest = question.labels
                values[ref] = {first: 0.85} | dict.fromkeys(rest, 0.05)
            else:  # extraction: find the one order number this binding knows
                assert question.text is not None
                start = call.needs[question.text].find("A-1042")
                found = start >= 0
                values[ref] = {
                    "value": "A-1042" if found else None,
                    "guess": None,
                    "p_null": 0.02 if found else 0.95,
                    "confidence": 0.9,
                    "span": [start, start + 6] if found else None,
                }
    return CallResult(values)


def main() -> list[tuple[Plan, list[Mapping[str, object]]]]:
    """Decide each message and replay its record; return each plan with replay's findings."""
    loaded = load(DESIGN)
    bindings = {"classifier": fixed_answers, "reader": fixed_answers}
    results: list[tuple[Plan, list[Mapping[str, object]]]] = []
    for message in MESSAGES:
        decision = decide_sync(loaded, {"message": message}, bindings)
        plan = decision.plan
        print(plan.kind, "|", plan.reply or plan.reason, "|", decision.verdict.kind)

        stored = decision.record.document  # the bytes you would save
        record = read_record(stored)
        unhashed = json.loads(stored)["unhashed"]
        for index, item in enumerate(record.log):
            if item["item"] == "call":
                oracles = [DESIGN.sources[s].oracle or "default" for s in item["sources"]]
                latency = unhashed["calls"][str(index)]["latency"]
                print(f"  call {index}: {item['sources']} via {oracles}, {latency:.6f} s")

        rerun, findings = replay_sync(record, loaded)
        assert rerun is not None
        print("  replay findings:", findings)
        print("  same hash:", rerun.record_hash == record.record_hash)
        results.append((plan, list(findings)))
    return results


if __name__ == "__main__":
    main()

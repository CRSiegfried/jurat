"""The tutorial's design: a shop's help desk that routes one customer message.

The tutorial at docs/tutorials/write-a-design.md builds it step by step, and helpdesk_run.py
decides with it.
"""

from jurat import (
    Design,
    Plan,
    Policy,
    Question,
    ReadView,
    Rule,
    Source,
    Text,
    Unasked,
    Unavailable,
    Unsure,
)
from jurat import types as t
from jurat.helpers import extract, pick_with_margin

TOPICS = [
    {"name": "delivery", "description": "Where an order is, or when it will arrive"},
    {"name": "return", "description": "Sending something back, or a refund for it"},
    {"name": "damaged", "description": "An item that arrived broken or faulty"},
    {"name": "other", "description": "Anything that is not delivery, a return or damage"},
]

SOURCES = {
    "triage": Source(
        {
            "q:triage/urgent": Question("noul", Text("Does the customer need an answer today?")),
            "q:triage/topic": Question(
                "choice", Text("What is the customer's message about?"), TOPICS
            ),
        },
        needs=["t:state"],
        oracle="classifier",
    ),
    "lookup": Source(
        {
            "q:order/number": Question(
                "extraction",
                Text("The order number the customer quotes"),
                None,
                value_type="text",
                text={"ref": "t:state.message"},
            ),
        },
        needs=["t:state"],
        oracle="reader",
    ),
}

RULES = {
    "topic": Rule(
        "d:topic",
        t.label("damaged", "delivery", "other", "return"),
        lambda r: pick_with_margin(r, "q:triage/topic", 0.5, 0.2),
        version="1",
        reads=["q:triage/topic"],
    ),
    "urgent": Rule(
        "d:urgent",
        t.label("no", "yes"),
        lambda r: pick_with_margin(r, "q:triage/urgent", 0.6, 0.2),
        version="1",
        reads=["q:triage/urgent"],
    ),
    "order": Rule(
        "d:order",
        t.string,
        lambda r: extract(r, "q:order/number", 0.5),
        version="1",
        reads=["q:order/number"],
    ),
}

MARKERS = (Unsure, Unavailable, Unasked)


def route(r: ReadView) -> Plan:
    """Escalate what the desk cannot settle or must answer today; inform the rest."""
    topic = r.read("d:topic")
    urgent = r.read("d:urgent")
    order = r.read("d:order")
    if isinstance(topic, MARKERS):
        return Plan("escalate", reason="the topic is not settled")
    if urgent == "yes":
        return Plan("escalate", reason=f"an urgent {topic} message")
    if isinstance(order, MARKERS):
        return Plan("inform", reply=f"Routed to {topic}; no order number found.")
    return Plan("inform", reply=f"Routed to {topic} for order {order}.")


DESIGN = Design(
    name="help-desk",
    state=t.record(message=t.string),
    sources=SOURCES,
    rules=RULES,
    policy=Policy(route, version="1", reads=["d:topic", "d:urgent", "d:order"]),
)

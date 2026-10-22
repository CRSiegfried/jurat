"""Plain literals and lambdas pass the public constructors under pyright strict: this module
is checked, never collected, and test_package imports it so it runs."""

from jurat import Origin, Plan, Policy, Rule, Source, Step, When
from jurat import types as t

when = When(lambda r: r.read("d:x") == "y", version="1")
rule = Rule("d:x", t.string, function=lambda r: r.read("t:state.subject"), version="1")
policy = Policy(lambda r: Plan("escalate", reason="x"), version="1")
args = {"days": Origin(ref="t:state.days")}
step = Step("t", args)
needs = ["t:state"]
source = Source({"t:y": t.string}, needs=needs, when=when)

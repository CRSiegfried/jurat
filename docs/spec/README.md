# Spec

The normative description of jurat's behavior, and the JSON conformance fixtures that any implementation, in any language, must pass. The Python package is one implementation of it. The spec carries its own version, separate from the package version.

For v0.0.1 the spec covers:

- loading a design, and the load checks that refuse an invalid one;
- decide: the turn's entry with the codec, the broken entry, interpret over rounds of sources and function rules, the three markers and their causes, and compile with a function policy, the plan rules and the guard;
- the ledger and the observer, the record with its canonical JSON encoding and hashes, and reading a stored record back;
- the record views: the escalation view and the redacted view;
- replay of a recorded turn with no model call;
- the binding protocol between the engine and a model adapter;
- the helpers: the choice helpers and combinators rules call, and the reference runner that runs a plan's steps;
- the live frontier, the answer grid and the `boundaries` fields, and the design lints that read them.

Outside v0.0.1: decision tables and data rules, plan declarations, candidate sets, the run-time stopping certificate, multi-turn continuation, and the evaluation subpackage.

Not yet written.

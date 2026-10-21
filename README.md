# jurat

A blackboard engine that turns a decision model's answers into one checked plan. Its one operation is **decide**, which runs two stages:

- **Interpret.** Sources fetch model answers and other outside values onto a blackboard in rounds, through bindings the caller supplies; rules, which are functions with declared reads, turn them into estimates. An unknown stays unknown, as a marker (`Unsure`, `Unavailable` or `Unasked`), never a quiet no or a `None`.
- **Compile.** One policy reads the estimates and writes the plan: execute these steps, clarify, inform, or escalate. The guard then checks every requirement the design declares for the plan and every requirement each step declares, refuses a step that declares none unless the design turns that off, and passes or refuses the plan.

Decide returns the plan, the guard's verdict and a record of the turn, and calls no tool. The caller runs the plan's steps with its own workflow engine, or with the small reference runner jurat ships. Nothing after decide consults the model, and the record replays the turn with no model call.

A jurat is the clause certifying that a statement was sworn before it is used. No plan here rests on evidence that was not recorded.

## Status

Pre-release; the engine for v0.0.1 is built. `load` checks a design and refuses an invalid one. `decide` runs interpret over rounds of sources and rules, then compile with the policy and the guard, and returns the plan, the verdict and a record that `replay` reruns with no model call. In v0.0.1, rules and the policy are written as Python functions. The escalation and redacted views, the helpers, two design lints and the reference runner ship with it. The core imports only the Python standard library ([ADR-0001](docs/adr/0001-write-the-core-in-python-with-no-runtime-dependencies.md)).

v0.0.1 leaves out decision tables, policy tables, data rules, plan declarations, candidate sets, candidate-valued label sets, and grounded declarations on a source or rule; `load` refuses a design that uses one. No state carries from one turn to the next: each turn stands alone. The written spec and its conformance fixtures come later ([docs/spec](docs/spec/README.md)).

## Contributing

Issues first: see [CONTRIBUTING.md](CONTRIBUTING.md). Report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md). This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT. See [LICENSE](LICENSE).

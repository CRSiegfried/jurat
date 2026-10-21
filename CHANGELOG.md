# Changelog

All notable changes to jurat are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor release may break the API; each break is listed under **Changed** or **Removed**.

## [Unreleased]

## [0.0.1] - 2026-10-20

### Added

- The design types: `Design`, `Source`, `Question`, `Text`, `Rule`, `Policy`, `When`, `Requirement`, `SourceRequirement`, `Group` and `Parameter`, with `Table`, `PolicyTable` and `PlanDeclaration` for the forms `load` refuses in this release.
- `jurat.types`: constructors for the type a field holds (`boolean`, `integer`, `number`, `string`, `bytes_`, `label`, `list_`, `mapping`, `record` and `nullable`).
- `load`: checks a `Design`, or its canonical document, before any turn runs and returns a `LoadedDesign`; an invalid design raises `InvalidDesign` naming each failed check.
- `decide` and `decide_sync`: one turn, interpret over rounds of sources and rules, then compile with the policy and the guard, returning the plan, the verdict and the record. Rules and the policy are Python functions.
- `RunSettings`: the run settings, with a round bound of 8 and a 60 s timeout by default; `EntryRefusal` for a wrong argument at `decide`'s entry, raised before any record is written.
- The markers `Unsure`, `Unavailable` and `Unasked`, each with its cause, for every value a turn could not settle.
- The record: its canonical JSON encoding and hashes, the observer, and `read_record` for a stored record, which raises `InvalidRecord` for one it cannot read.
- `replay` and `replay_sync`: a recorded turn rerun under a loaded design with no model call, reporting each difference as a finding.
- The record views: `escalation` and `redact`.
- The binding protocol between the engine and a model adapter: `SourceCall`, `RenderedQuestion` and `CallResult`.
- `jurat.helpers`: the choice helpers and combinators rules call, the design lints `dead_questions` and `unsure_to_execute`, and the reference runner `run_plan`, which raises `RunRefusal` for a plan it refuses before calling any tool and returns `BindingNotFound` for a step argument whose binding missed.
- Project scaffolding: package, license, code of conduct, contributing and security policies, CI.

### Not included

- `load` refuses decision tables, policy tables, data rules, plan declarations, candidate sets, candidate-valued label sets and grounded declarations.
- No state carries from one turn to the next.

[Unreleased]: https://github.com/CRSiegfried/jurat/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/CRSiegfried/jurat/releases/tag/v0.0.1

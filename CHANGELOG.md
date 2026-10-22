# Changelog

All notable changes to jurat are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor release may break the API; each break is listed under **Changed** or **Removed**.

## [0.3.0] - 2026-10-22

### Added

- Label descriptions: a `choice` or `score` label may be `{"name": ..., "description": ...}`, which the design document, the call item and the question's identity hold as declared. `RenderedQuestion.descriptions`, keyword-only, gives each label's description aligned with `labels`, `None` where a label has none. Answers, the answer grid, the lints and the `boundaries` checks read names only.
- `CallResult.warnings`: the binding's own lines about the call, recorded under the call in `unhashed` and outside every hash. A malformed value is kept in `invalid` and is never a fault.
- `CallResult.precision`: the decimal places the provider rounds probabilities to, hashed on the call item. A value that is not `None` or an integer from 0 to 17 is `invalid-return` for the whole call.
- `CauseCode.unsupported`: the marker a binding returns for a ref it will not put to any provider, with the reason as a line of `warnings`. It is the 24th cause code. A rule that returns the marker it read provides it unchanged, so `unsupported` stands on a `d:` entry with its refs as any binding code does.
- The record document's `hash` member: the record hash over the log. `read_record` recomputes it and refuses a document whose `hash` is missing or does not match.
- `jurat.evaluation`, the evaluation subpackage, with its first job: `use_answers(record, loaded, other)` and `use_answers_sync`, which rerun `record`'s turn under `loaded` with `other`'s calls answering and return `{"other", "findings"}`, `other` its record hash and the findings replay's. Both records are a `Record` or a stored document and must be finished turns of the loaded design with the same state entry and `settings` item. Each refusal is an `EvaluationRefusal`, its `reason` a `RefusalReason`; a replay record is a valid `record` or `other`, so two models decided under different parameter sets compare by replaying the second under the first's loaded design.
- `ReadView`, the view a function `when`, rule or policy is called with, is public, so a component can be annotated with it. `When`, `Rule` and `Policy` type `function` as `Callable[[ReadView], object]`, so a lambda passes pyright strict.

### Changed

- Records and design documents are format 3. Every design hash and record hash changes, so golden hashes must be updated. Every format-3 call item holds `precision`.
- An exception's recorded traceback starts at the component's own frame and names each file by its module, so the record holds no local paths from the traceback.
- `read_record` reads format-1 and format-2 records and `load` loads format-1 and format-2 design documents to their own hash, each under its format's rules. A format-2 record holding a described label, `precision`, `warnings`, `unsupported` or a `hash` member is refused.
- `Step` takes its arguments as any `Mapping`, read once through `items()`, and still keeps a read-only copy; a non-mapping is a `TypeError`. The recorded plan is unchanged.
- Check 4's message for a rule that provides another ref names the rule's own `d:` ref, and for another rule's `d:` ref, whose it would be.

## [0.2.0] - 2026-10-22

### Added

- The `extraction` answer kind: a question that asks for a typed value read from a text. `Question` takes `value_type`, `is_list`, `text` and `reference_date`; `ValueType` names the six value types; `ExtractionAnswer` holds the answer's `value`, `guess`, `p_null`, `confidence` and `span`, and checks each when built. Dates, times, datetimes and durations are checked by jurat's own grammar, alike on every Python.
- `RenderedQuestion` gives an extraction question's `value_type`, `is_list`, `text` and `reference_date`, keyword-only; a question of another kind leaves them at `None`, `False`, `None` and `None`.
- `CallResult.truncated`: the `q:` refs whose answer came from a response the provider cut short, recorded under the call in `unhashed` and outside every hash.
- `jurat.helpers.extract`: an extraction answer's value, or `Unsure` with `abstained` when the value is null or `p_null` is at or above the bar.
- The extraction roles of a `boundaries` entry, `null`, `confidence` and `value`, and the dead-question lint's points over an extraction read; an extraction read with no `value` role takes its values from the lint's `values` argument.

### Changed

- Records and design documents are format 2, and the helper version is 2. Every design hash and record hash changes, so golden hashes must be updated.
- `AnswerKind` has a fourth member, `extraction`; code that matches on every kind must handle it.
- The ledger refuses a `q:` answer whose kind is not its call item's question's.
- 0.2.0 reads format-1 records and loads format-1 design documents to their own hash, each under format 1's rules, so a record written under 0.1.0 replays under its stored design with `helpers-mismatch` as its only finding.

## [0.1.0] - 2026-10-21

### Changed

- `Source.batch` is now `Source.oracle`, the name of the binding that answers the source, and the design document's `batch` member is `oracle`. Every design hash changes, so a record written under 0.0.1 names a design hash that no reloaded design has. `EntryRefusal` messages name the oracle, and a missing `default` binding says it serves the sources with no oracle.
- `load` refuses a source that provides both `q:` and `t:` refs (check 4). Split it into two sources that name one oracle: ready in the same round, they are still one call.

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

[Unreleased]: https://github.com/CRSiegfried/jurat/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CRSiegfried/jurat/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/CRSiegfried/jurat/releases/tag/v0.0.1

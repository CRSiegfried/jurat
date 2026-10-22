# Spec

The normative description of jurat's behavior, and the JSON conformance fixtures that any implementation, in any language, must pass. The Python package is one implementation of it. The spec carries its own version, separate from the package version.

For v0.0.1 the spec covers:

- loading a design, and the load checks that refuse an invalid one;
- decide: the turn's entry with the codec, the broken entry, interpret over rounds of sources and function rules, the three markers and their causes, and compile with a function policy, the plan rules and the guard;
- the ledger and the observer, the record with its canonical JSON encoding and hashes, and reading a stored record back;
- the record views: the escalation view and the redacted view;
- replay of a recorded turn with no model call, and the evaluation job `use_answers`;
- the binding protocol between the engine and a model adapter;
- the answer kinds: `noul`, `choice` and `score`, which give a probability to each label, and `extraction`, a typed value read from a text, with the probability that the text holds none, the model's reading whatever it answered, its confidence and the span it read;
- the helpers: the choice helpers, `extract` among them, and the combinators rules call, and the reference runner that runs a plan's steps;
- the live frontier, the answer grid and the `boundaries` fields, and the design lints that read them.

Outside v0.0.1: decision tables and data rules, plan declarations, candidate sets, the run-time stopping certificate, multi-turn continuation, and every evaluation job but `use_answers`.

The sections below are written; the rest is not yet.

## Formats

A record and a design document each carry a format number. This engine writes format 3 and reads formats 1, 2 and 3, each under its own rules. A format-1 document holds no extraction question, answer or `boundaries` roles of one, and its record's `unhashed` holds no `truncated` member. A format-1 or format-2 document holds no described label; its record holds no `unsupported` cause, no call item `precision`, no `warnings` in a `calls` entry or as an `invalid` key, and no document `hash`. A record's cause codes are read against its own format's set. A design document loads to the hash of its own bytes, so a format-1 or format-2 design keeps its hash and replays its records. The record's `design` item also carries the helper version, 2 since `extract` was added; replay reports a different one as `helpers-mismatch` and runs under the loaded helpers.

## The record document

From format 3 the record document is `{"hash", "log", "unhashed"}`, where `hash` is the record hash over the log, derived from the log's bytes when the record is written. A reader requires it, recomputes it and refuses a mismatch, and refuses any other member. It is never read as truth. It catches an edit made without recomputing it. It does not catch an adversary, whom only a turn id held outside the record catches. A job that edits a record's log recomputes the hash. The redacted view carries `hash` as an open value.

An exception's `traceback` in `unhashed` is still CPython's text. It is built as `traceback.TracebackException` with its source lines looked up, its frames edited, then formatted. Along the chain and each exception group, a stack drops its leading run of frames of `jurat`, `asyncio` or a module under either, so its first frame is the component's. A stack made only of such frames, as a builtin component's is, keeps them all. Each remaining frame's filename becomes the frame's `__name__` global when that is a string, else the text after its last `/` or `\`; a pseudo-file such as `<string>` is kept. The same rule holds for a rule, a function `when`, the policy, the codec and a binding. This does not make a raw record shareable: the redacted view is the form to share.

## The binding protocol

A `choice` or `score` label is a name or exactly `{"name": n, "description": d}`. Both are strings; the description holds a character that is not whitespace and has no slots. Names are distinct; descriptions may repeat. A candidate-valued set's `fixed` part takes the same form. `noul` and `extraction` declare no labels. Load's count bound counts labels; its code-point bound sums names and descriptions. The answer grid, the lints, load's `boundaries` checks and every answer read label names alone. The design document, the question's identity and the call item's `questions` hold the labels as declared, so a description change makes a new question identity.

`RenderedQuestion.labels` holds the names. `descriptions`, keyword-only, is aligned with them: `None` where a label is undescribed, `(None, None)` for `noul`, `()` for `extraction`. A `RenderedQuestion` built with none gets `None` for each label; a non-empty value is kept as given, unchecked.

`CallResult.warnings` conforms when it is exactly a `tuple` of exact `str`, each holding a character that is not whitespace and in the value domain, at most 2^8 of them and 2^16 code points in all; order is kept and a repeat is allowed. Interpret tests it on every `CallResult`, a structural fault included. A conforming one is written under the call's `unhashed.calls` entry as `warnings` only when non-empty. A non-conforming one is never a fault: nothing is written and its repr goes under `invalid["warnings"]`. A late result keeps none. `redact` digests the list as one value.

`CallResult.precision` is the decimal places the provider rounds probabilities to. It is `None` when unknown, or when the provider rounds to a step that is not a decimal count. It is `None` or an integer from 0 to 17, `bool` excluded; any other value is `invalid-return` for the whole call, with `invalid["precision"]`. It is hashed on the call item beside `snapshot`, present on every format-3 call item, null on a failed one. The engine does not test that probabilities are rounded to it. The fixture binding returns `None`, as it returns null provenance and no `truncated` or `warnings`.

A binding that will not put a question to any provider, because its ref, kind or shape is outside what the provider takes, returns `Unavailable` with the code `unsupported` for that ref. A binding that refuses a whole batch marks every ref. Interpret writes the marker with a cause naming that ref; the cause carries no text. The binding gives its reason as a line of `warnings`. The call is `returned` and charged, and its call item holds null provenance where no provider was called. `unsupported` is one of the four markers a binding may return, with `refused`, `invalid-answer` and `invalid-return`. It is never a `Failed` outcome's code (a `Failed` carrying it is `invalid-return`), never a plan's cause and never the guard's. A rule that returns the marker it read provides it unchanged, so `unsupported` stands on a `d:` entry with its refs as any binding code does. Replay's fixture returns it as recorded.

## The extraction kind

An extraction question declares, beside its kind and wording, no labels, a `value_type` (`text`, `number`, `date`, `time`, `datetime` or `duration`), `is_list`, `text`, a board ref naming the string its spans index, and `reference_date`, a board ref or null naming the string a date written without a year is read against. Both refs become needs of the source and are exactly of type `string`; the question's identity covers the four members, never the values they name.

The binding returns, for the question's ref, exactly an object of `value`, `guess`, `p_null`, `confidence` and `span`, null where it has nothing. `value` and `guess` are null or a value of the type, or when `is_list` a list of such values with no null; `p_null` and `confidence` are null or a number in [0, 1]; `span` is null, a pair `[start, end]` of code-point offsets into the text, end exclusive, or one pair per list item, and locates `value`, or `guess` when `value` is null. A `number` is held as a float. A `date` is `YYYY-MM-DD`; a `time` is `HH:MM:SS`, an optional fraction and an optional `Z` or `±HH:MM`; a `datetime` is a date, `T` and a time; a `duration` is ISO 8601's `P` form without weeks or fractions. These grammars are the engine's own, with ASCII digits and uppercase `T` and `Z` only. Any other return is `invalid-answer` on that ref alone. A model's none is a null `value`, never a marker; `extract` turns it, or a `p_null` at or above its bar, into an abstention.

The binding may also report the refs whose response was cut short as `truncated`, which the record keeps outside every hash.

## Evaluation

`jurat.evaluation.use_answers(record, loaded, other)` reruns `record`'s turn under `loaded` with `other`'s call items and entries answering, `q:` and `t:` refs alike; `use_answers_sync` is its twin. `record` and `other` are each a `Record` or a stored document, read through `read_record`. It returns `{"other": h, "findings": [...]}`, `h` the other record's hash. The state entry, the run settings, the cut and every comparison are `record`'s, so the findings are replay's with `other` as the fixture: `not-in-record` and `ref-not-in-record` name what `other` does not hold, and `question-changed` compares against `other`'s identities. A `Failed` call or a returned marker in `other` comes back as replay returns a recorded one. Neither record is edited, and neither's `unhashed` is read.

After both records are read and before any replay it raises `EvaluationRefusal`, `reason` in this order: `invalid-record` when a document does not read, carrying the `InvalidRecord` as `error`; `unfinished` for a record with no `verdict` item, then `nothing-to-replay` for one with no state entry, then `wrong-design` for one whose design hash is not `loaded.design_hash`, each for `record` then `other`; and `different-turn` when the two state entries or `settings` items differ by canonical bytes. `turn` is the refused record's hash, `other`'s for `different-turn`, and `None` for `invalid-record`, which carries `error` instead; every other attribute is `None`. A `loaded` that is not exactly a `LoadedDesign` is an `EntryRefusal`, before either record is read.

A replay record, the `Record` that `replay` returns, is a valid `record` or `other`. To compare two models decided under different parameter sets, replay the second's record under the first's loaded design and pass the replay record as `other`; a threshold that makes the rerun ask a source the second never called shows as `not-in-record` in the replay's findings, and its entries come back `not-returned` in `use_answers`'s.

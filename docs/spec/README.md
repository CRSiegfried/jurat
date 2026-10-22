# Spec

The normative description of jurat's behavior, and the JSON conformance fixtures that any implementation, in any language, must pass. The Python package is one implementation of it. The spec carries its own version, separate from the package version.

For v0.0.1 the spec covers:

- loading a design, and the load checks that refuse an invalid one;
- decide: the turn's entry with the codec, the broken entry, interpret over rounds of sources and function rules, the three markers and their causes, and compile with a function policy, the plan rules and the guard;
- the ledger and the observer, the record with its canonical JSON encoding and hashes, and reading a stored record back;
- the record views: the escalation view and the redacted view;
- replay of a recorded turn with no model call;
- the binding protocol between the engine and a model adapter;
- the answer kinds: `noul`, `choice` and `score`, which give a probability to each label, and `extraction`, a typed value read from a text, with the probability that the text holds none, the model's reading whatever it answered, its confidence and the span it read;
- the helpers: the choice helpers, `extract` among them, and the combinators rules call, and the reference runner that runs a plan's steps;
- the live frontier, the answer grid and the `boundaries` fields, and the design lints that read them.

Outside v0.0.1: decision tables and data rules, plan declarations, candidate sets, the run-time stopping certificate, multi-turn continuation, and the evaluation subpackage.

The sections below are written; the rest is not yet.

## Formats

A record and a design document each carry a format number. This engine writes format 2 and reads formats 1 and 2, each under its own rules: a format-1 document holds no extraction question, answer or `boundaries` roles of one, and its record's `unhashed` holds no `truncated` member. A design document loads to the hash of its own bytes, so a format-1 design keeps its hash and replays its records. The record's `design` item also carries the helper version, 2 since `extract` was added; replay reports a different one as `helpers-mismatch` and runs under the loaded helpers.

## The extraction kind

An extraction question declares, beside its kind and wording, no labels, a `value_type` (`text`, `number`, `date`, `time`, `datetime` or `duration`), `is_list`, `text`, a board ref naming the string its spans index, and `reference_date`, a board ref or null naming the string a date written without a year is read against. Both refs become needs of the source and are exactly of type `string`; the question's identity covers the four members, never the values they name.

The binding returns, for the question's ref, exactly an object of `value`, `guess`, `p_null`, `confidence` and `span`, null where it has nothing. `value` and `guess` are null or a value of the type, or when `is_list` a list of such values with no null; `p_null` and `confidence` are null or a number in [0, 1]; `span` is null, a pair `[start, end]` of code-point offsets into the text, end exclusive, or one pair per list item, and locates `value`, or `guess` when `value` is null. A `number` is held as a float. A `date` is `YYYY-MM-DD`; a `time` is `HH:MM:SS`, an optional fraction and an optional `Z` or `±HH:MM`; a `datetime` is a date, `T` and a time; a `duration` is ISO 8601's `P` form without weeks or fractions. These grammars are the engine's own, with ASCII digits and uppercase `T` and `Z` only. Any other return is `invalid-answer` on that ref alone. A model's none is a null `value`, never a marker; `extract` turns it, or a `p_null` at or above its bar, into an abstention.

The binding may also report the refs whose response was cut short as `truncated`, which the record keeps outside every hash.


# ADR-0001: Write the core in Python with no runtime dependencies

**Status**: Proposed

## Context

jurat decides a turn by calling code its users write: oracle bindings that fetch model answers and other outside values for sources, `when` functions, rules, a policy, and optionally a codec and an observer. Nearly all of it is Python, the bindings async, and most of a turn's time is spent waiting on model calls. The engine's own work (scheduling rounds, checking reads, encoding and hashing the record) is small next to those calls. The engine calls no tool; the caller runs the plan.

A library that people trust with their decision logic should be easy to audit and should not bring a dependency tree into their environment.

## Considered options

**A. A compiled core (for example Rust) with Python bindings.** It would be faster, and it could be reused from other languages. But the engine's CPU time isn't the bottleneck, every callback would cross the language boundary, async across that boundary is complex, and building wheels for every platform is a large and lasting cost.

**B. Pure Python with a few dependencies** (for example a validation or typing library). It would be convenient, but every dependency adds version conflicts and supply-chain risk for users, and the engine doesn't need any.

**C. Pure Python, standard library only. Chosen.**

## Decision

The core package imports only the Python standard library and makes no network calls. Its canonical JSON (following RFC 8785), hashing and keyed digests, dependency graph and concurrency are written on `json`, `hashlib`, `hmac`, `graphlib` and `asyncio`. Where a standard has test data, the core's version is checked against it: the canonical JSON against the `rfc8785` package at test time, the key derivation against RFC 5869's test vectors. Integrations with third-party SDKs ship as optional extras (`jurat[<name>]`), imported lazily inside the integration so that importing `jurat` never imports them. Reach into other languages comes from the spec and its JSON conformance fixtures, not from shared native code.

## Consequences

- A test checks that `import jurat` loads no module outside the standard library.
- Python 3.11 is the minimum, the first with `enum.StrEnum` and `asyncio.TaskGroup`; the supported versions are those CI tests.
- Another-language implementation is a port that passes the conformance fixtures, so the spec has to be complete enough to port from.

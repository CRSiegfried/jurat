# ADR-0002: Record every value of a turn, and redact by class in a separate view

**Status**: Proposed

## Context

A turn's record has three jobs. It lets someone debug or audit a decision, it lets jurat replay the turn with no model call by answering each oracle binding call from the record, and it lets evaluation jobs measure a design against past turns. All three need the values: the user's message, the model's answers, tool results and the estimates built from them. Those values are often personal or confidential, and a record that reaches a log pipeline unchanged leaks them.

## Considered options

**A. Record the shape of a turn by default, values only by opt-in.** Safe to log, but a record without its values cannot be replayed or audited, so it fails the record's main jobs.

**B. Record nothing; users add their own logging.** Safe, but every user rebuilds the same instrumentation, and none of it can replay a turn.

**C. Record every value, and redact through a separate view. Chosen.**

## Decision

The record holds every value of the turn. Each value carries a class: `open`, `identifier` or `content`. A design can declare the classes of the input state and of each source's and rule's values; otherwise a value's type decides, so strings and bytes are `content`, and labels, numbers and booleans are `open`. Text the engine records itself, such as a rendered prompt, is `content`.

A pure function turns a record into a redacted view. `open` values pass through; `identifier` and `content` values become keyed digests, an HMAC-SHA-256 under a key derived for that one turn from a root key the record's owner holds. Given that turn's key, a verifier can confirm a disclosed value against its digest without seeing any other turn. The redacted view is not a record and cannot be read back as one.

jurat itself emits nothing and makes no network calls. The caller receives the record from each turn, and can watch items as they are written through an observer it supplies.

## Consequences

- Replay, audit and evaluation work from the record and the design.
- The record is as sensitive as the turn's inputs, and the caller stores it accordingly; the redacted view is what leaves.
- Free text is digested in the view by default. A design marks a value `open` when it is safe to show.
- An observer sees unredacted items, so a caller exporting through one decides what to redact. What a bundled exporter sends by default is decided when one is built.

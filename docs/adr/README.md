# Architecture decision records

Each significant design decision is a numbered file: `NNNN-short-title-as-a-verb-phrase.md`. Records are not rewritten once accepted; a later record supersedes an earlier one and says so in both.

Format:

```markdown
# ADR-NNNN: Title

**Status**: Proposed | Accepted (YYYY-MM-DD) | Superseded by ADR-NNNN

## Context
The forces at play, stated so a reader without other context can follow.

## Considered options
Each option and why it was taken or not.

## Decision
What we will do.

## Consequences
What becomes easier, what becomes harder, what we now owe.
```

| ADR | Decision |
|-----|----------|
| [0001](0001-write-the-core-in-python-with-no-runtime-dependencies.md) | Write the core in Python with no runtime dependencies |
| [0002](0002-record-every-value-and-redact-by-class.md) | Record every value of a turn, and redact by class in a separate view |

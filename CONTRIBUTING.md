# Contributing

Thanks for your interest in jurat. The design and the spec are still young, so contributions start with an issue.

## Issues first

- **Bugs:** open an issue with a minimal reproduction, the jurat and Python versions, and what you expected.
- **Changes and features:** open a proposal issue before writing code. Say what problem it solves and how it bears on the spec's guarantees.
- **Pull requests** are welcome once an issue is agreed. Link it in the PR. A PR without an agreed issue may be closed, however good the code is.
- **Security problems:** do not open an issue; follow [SECURITY.md](SECURITY.md).

## Development

```sh
uv sync
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
```

CI runs the same checks on Python 3.11–3.14 on Linux, macOS and Windows.

## Ground rules for changes

- **The core imports only the standard library.** Integrations with third-party SDKs go in optional extras and are imported lazily.
- **The spec is normative.** A change that alters behavior the spec describes changes the spec in the same PR, with tests.
- **The public API is what `jurat.__all__` lists.** Modules and names starting with an underscore are internal. Removing or changing a public name needs a deprecation first: one minor release with a `DeprecationWarning`.
- **Record user-visible changes** under `Unreleased` in [CHANGELOG.md](CHANGELOG.md).
- **Record significant design decisions** as an ADR in [docs/adr/](docs/adr/).

## Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

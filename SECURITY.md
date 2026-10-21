# Security policy

## Supported versions

Until 1.0, only the latest release receives fixes. After 1.0, the latest minor release of each supported major version does.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability** (<https://github.com/CRSiegfried/jurat/security/advisories/new>). Do not open a public issue.

Include what you found, how to reproduce it, and the versions affected. You can expect an acknowledgement within a week. Once a fix is ready, the advisory is published with credit to you unless you prefer otherwise.

## What counts

jurat runs code you supply (oracle bindings, function `when`s, rules, the policy, the codec, the observer, and the `functions` mapping given to `load`) in your process; it does not sandbox it. The engine runs no step of a plan: you run the plan, with your own workflow engine or with the reference runner, which calls the tools you pass it. Reports in scope include ways jurat itself breaks a guarantee it documents, for example:

- the guard passing a requirement whose estimate is unknown (`Unsure`, `Unavailable` or `Unasked`)
- the model being consulted after decide returns
- a rule reading something it did not declare
- the reference runner calling any tool for a decision it must refuse (a refused verdict, a plan that is not `execute`, or a step whose tool it was not given), calling a step outside the decision's steps, or calling a step twice
- a redacted view showing a non-null `identifier` or `content` value instead of its keyed digest

The record holds every value of a turn and is as sensitive as the turn's inputs; redaction is a separate view of it. The core makes no network calls and has no runtime dependencies. Issues in optional extras' third-party SDKs belong with those projects.

# Releasing

jurat is published to PyPI by `.github/workflows/publish.yml`, through Trusted Publishing: PyPI trusts this repository's workflow, so no API token is stored anywhere. A version number can be published to PyPI only once, so a mistake is fixed by the next version, never by replacing one.

## One-time setup

1. **PyPI and TestPyPI.** On each site, under *Your account → Publishing → Add a new pending publisher*, enter project `jurat`, owner `CRSiegfried`, repository `jurat`, workflow `publish.yml`, and environment `pypi` on PyPI or `testpypi` on TestPyPI. A pending publisher reserves the name until the first upload.
2. **GitHub environments.** Under the repository's *Settings → Environments*, create `pypi` and `testpypi`. Give each a required reviewer (the maintainer), and limit `pypi` to tags matching `v*`.

## Release procedure

1. In one commit, bump `version` in `pyproject.toml`, run `uv lock`, and move the `Unreleased` entries in `CHANGELOG.md` under the new version and date.
2. Optional dry run: run the **publish** workflow by hand (*Actions → publish → Run workflow*) and approve the `testpypi` environment. Check the page at <https://test.pypi.org/p/jurat>.
3. Tag the commit `vX.Y.Z` and push the tag. Sign the tag (`git tag -s`) once a signing key is set up.
4. Create a GitHub release from the tag, with the changelog entry as its notes, and publish it.
5. The workflow checks that the tag names the version in `pyproject.toml`, runs the tests, builds the sdist and wheel, and runs `twine check`. It then waits for you to approve the `pypi` environment, and uploads, with attestations.

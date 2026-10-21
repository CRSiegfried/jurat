# Releasing

## Release procedure

To be written along with the publish workflow. Plan:

1. Bump `version` in `pyproject.toml` and move the `Unreleased` entries in `CHANGELOG.md` under the new version, in one commit.
2. Tag the commit `vX.Y.Z` (signed) and push the tag.
3. The publish workflow builds the sdist and wheel, runs `twine check`, and publishes to PyPI by Trusted Publishing after you approve the `pypi` environment. Attestations are generated automatically.
4. Create a GitHub release from the changelog entry.

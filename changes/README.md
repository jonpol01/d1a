# Unreleased changes

One file per change, named `<PR number>-<slug>.md` (for example `163-changelog-fragments.md`), instead of an edit to
CHANGELOG.md, so pull requests never conflict on the same lines. A file holds one or more `### Section` headings
(Added, Changed, Deprecated, Removed, Fixed, Security, in that order), each with `- ` list items; wrap long items with
two-space indented lines:

```markdown
### Added

- `d1a.thing` does X (#163). What changed for a user, in plain words.
```

`python scripts/release_notes.py --check-fragments` checks every file (CI runs it through tests/test_release_notes.py).
At release, `python scripts/release_notes.py assemble X.Y.Z --write` moves them all into CHANGELOG.md's new
`## [X.Y.Z]` section, newest PR first, and deletes them (CONTRIBUTING.md, Releasing).

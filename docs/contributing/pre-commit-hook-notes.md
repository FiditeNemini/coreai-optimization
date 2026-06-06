# Pre-Commit Hook Notes

Notes on pre-commit hooks whose addition or modification has follow-on rules. **For the exact configuration of any hook (entry, args, file regex), read `.pre-commit-config.yaml` (root) or `external/.pre-commit-config.yaml` (its OSS-export mirror) directly — they are the single source of truth, and the two are intentional mirrors.**

## `add-license-header`

Inserts or refreshes the BSD-3-Clause license header on every file the hook's `files:` regex matches.

### Adding a source file with a new extension

When you add a source file whose extension or name isn't already covered by the hook's `files:` regex, extend the configuration so the file gets a license header. Either:

- Widen the `files:` regex in **both** `.pre-commit-config.yaml` mirrors, and add comment-style support to the script if the new extension needs a non-`#` style.
- Or, if the file genuinely shouldn't carry a header (binary, data, vendored, generated), leave the regex alone and add a one-line rationale to the PR description.

### Why an inclusion list, not an exclusion list

New file types stay silently ignored until someone deliberately adds them, rather than getting `# header` prepended (which would break parsers like JSON).

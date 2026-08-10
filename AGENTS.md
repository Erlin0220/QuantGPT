# QuantGPT repository instructions

## Git sync

- Treat `personal` as the owner's writable remote; do not push project changes to upstream `origin` unless the user explicitly asks.
- After a requested project change is complete and the relevant validation passes, commit the completed change instead of leaving it only in the working tree.
- The repository `post-commit` hook automatically pushes the current branch to `personal`; do not force-push to make auto-sync succeed.
- If auto-sync reports a non-fast-forward or authentication failure, keep the local commit intact and report the failure rather than rewriting remote history.
- Do not commit runtime state, secrets, databases, PID/lock files, logs, caches, or other generated artifacts just to make the tree clean.

# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files in `.scratch/`.

## Why

The upstream GitHub repository is read-only for the current account, while the writable personal fork currently has GitHub Issues disabled. Local Markdown therefore provides a writable tracker without changing remote repository settings.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`
- Tickets are numbered from `01` in dependency order
- Ticket state is recorded with a `Status:` line
- Agent-ready work uses `ready-for-agent`

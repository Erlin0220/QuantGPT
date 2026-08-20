# 02 — Resolve Research Provenance Locally

**What to build:** Make new and recoverable historical WorldQuant trials carry trustworthy Research Provenance by resolving fields and dataset/category from explicit metadata, platform evidence, expression parsing, and a project-local field registry. Unresolved provenance must be explicit and must not contaminate dataset-specific scheduler learning.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [ ] The active USA/TOP3000/Delay-1 research context has a local field metadata registry that can resolve known BRAIN fields to truthful dataset/category evidence without a network lookup on every trial.
- [ ] New trials resolve data fields, family and dataset/category from the documented precedence order whenever evidence exists.
- [ ] Core/derived aliases that do not map truthfully to a platform dataset remain distinguishable from real dataset IDs; the implementation does not fabricate IDs to force completeness.
- [ ] Each trial exposes a provenance resolution state and an explicit reason when dataset evidence remains partial/unresolved.
- [ ] Dataset-specific scheduler/feedback aggregation excludes unresolved identities instead of grouping them into a learnable `unknown` pseudo-dataset.
- [ ] Historical metadata reconciliation safely backfills derivable provenance without overwriting stronger explicit/platform evidence.
- [ ] Status/research-memory output reports provenance completeness and unresolved counts/reasons so improvement is measurable from the current 26.68% dataset-id baseline.
- [ ] Tests cover explicit metadata precedence, platform metadata, expression/registry recovery, unresolved core fields, conservative backfill, and exclusion from dataset-specific learning.
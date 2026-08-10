# 05 — Add Positive and Negative Research Memory

**What to build:** Make each autonomous research round retrieve compact Positive Research Memory and Negative Research Memory from existing persisted trials, candidates and lineage so generation builds on productive mechanisms while avoiding repeated failed/duplicate structures. Apply soft family/structure diversity pressure without permanently banning whole datasets.

**Blocked by:** 01 — Route Metric Failures into Directed Mutation; 02 — Resolve Research Provenance Locally.

**Status:** resolved

- [ ] Positive Research Memory can surface candidate-producing structures, mutation paths that improved parent metrics, and actionable near-threshold parents with their family/dataset context.
- [ ] Negative Research Memory can surface repeated low-yield failure structures, structural duplicates, and mutation paths that repeatedly failed to improve their parents.
- [ ] Memory retrieval is deterministic/bounded and uses existing persisted research evidence; no vector database, embedding service or new external runtime dependency is added.
- [ ] The existing bounded LLM generation path receives distilled positive/negative structural guidance and explicit avoid-like evidence rather than only exact-expression dedupe.
- [ ] Deterministic seed planning can consume the same memory summaries where useful, so learning is not limited to LLM-generated candidates.
- [ ] Structural similarity/overuse is a soft penalty or coverage signal; an entire dataset/family is not permanently blacklisted from a small failure sample.
- [ ] Generated plans continue to prefer simple, interpretable hypotheses and reject redundant nesting/cosmetic variants that do not meaningfully change research logic.
- [ ] Each research run reports which memory evidence/structure signatures influenced planning without exposing unnecessary private account-linked Alpha IDs or PnL.
- [ ] Tests prove repeated failed structural motifs are avoided, productive rescue motifs can be reused, and family diversity improves without breaking canonical dedupe.
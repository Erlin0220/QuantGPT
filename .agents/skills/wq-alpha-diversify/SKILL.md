---
name: wq-alpha-diversify
description: Diversify a WorldQuant BRAIN Alpha research pool when self-correlation, family saturation, repeated near-clones, or poor inventory breadth becomes the bottleneck. Prefer new information sources and hypotheses over cosmetic parameter variants.
---

# WorldQuant Alpha Diversify

## R — Reading

Distilled from `worldquant:learn2quant`, `worldquant:finding-alphas-official`, `worldquant:iqc-2026`, and `worldquant:iqc-guidelines-2026`.

WorldQuant explicitly teaches diversification of the Alpha pool. Official Learn2Quant lessons separate diversity by data category, idea type, holding frequency/delay and model/combination technique. Its risk-management lesson also warns that common factor exposure can create correlation even among superficially different approaches. Official IQC material spans Fundamental/Model, Price Volume, D0, Options/Relationship/Vector and newly released datasets. IQC rules apply a correlation test when Alpha pools are combined.

## I — Interpretation

Diversification means changing the source of predictive information, not merely changing a window from 20 to 21. The research portfolio should cover independent data families, mechanisms, horizons and structural motifs while still exploiting empirically productive families.

## A1 — Past application

The official curriculum itself moves across Price Volume, Fundamental, Analyst, Sentiment, Options, Model and other categories. This is a template for research breadth, not a recommendation to produce many near-duplicates inside one category.

## A2 — Future trigger

Invoke when Self-Correlation failures rise, the candidate queue is dominated by one family, Research Memory shows an overused structure, or a user asks for low-correlation/new-direction Alphas. For ordinary new-candidate generation without saturation, use `wq-alpha-hypothesis`.

## E — Execution

1. Read Research Memory: family counts, candidate family counts, self-correlation family counts, dataset usage and overused structures.
2. Read `kb_search_alpha_knowledge` for diversification guidance and strong cards in underrepresented families.
3. Read live Data Explorer and pick account-visible datasets/fields from underused categories before inventing a new expression structure.
4. Allocate the next batch across at least 3 genuinely different research cells when the budget permits. A research cell should differ by meaningful family/dataset/structure, not only lookback.
5. Keep a bounded exploitation slice for families with real ACTIVE/candidate evidence, but reserve explicit exploration for low-coverage categories.
6. On Self-Correlation failure, mutate in this priority order:
   1. different data source / dataset;
   2. different economic mechanism / idea type;
   3. different structural transform/horizon;
   4. if evidence points to shared factor exposure, a justified neutralized sibling;
   5. only then fine parameter changes.
7. Review each diversified candidate with `wq-alpha-review`. For `RUN` candidates emit structured `skill_candidates` with `skill_chain: ["wq-alpha-hypothesis", "wq-alpha-diversify", "wq-alpha-review"]`, the new hypothesis/family/data fields, and review notes; return them through `wq_brain_autonomous_research` with deterministic fallback disabled.
8. Use BRAIN simulation and current SC feedback to evaluate whether diversification is real. Persist family/dataset/structure lineage.
9. Avoid creating a pool that is numerically diverse but economically identical.

## B — Boundary

- Do not add random noise or irrelevant terms to lower measured correlation.
- Do not abandon all proven families merely for novelty; keep bounded exploitation.
- Do not use unavailable datasets or fields.
- Do not interpret local correlation proxies as official SC results.

## Related skills

- `wq-alpha-hypothesis`
- `wq-alpha-review`
- `wq-alpha-repair`

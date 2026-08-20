---
name: wq-economic-hypothesis
description: Build mechanism-first, falsifiable WorldQuant Alpha hypotheses from data semantics, expected-return drivers, competing explanations and current-condition evidence. Use before wq-alpha-hypothesis when generating new Alpha ideas, especially when recent BRAIN research is dominated by low_fitness or repeated weak hypotheses.
---

# WorldQuant Economic Hypothesis

## R — Reading

This skill is distilled from authoritative publisher/author materials stored in QuantGPT Knowledge Sources:

- `textbook:tulchinsky-finding-alphas-2e-2019` — *Finding Alphas, Second Edition*: Alpha design/evaluation, data and Alpha design, turnover, correlation, overfitting/bias control, and structured exploration of Alpha space.
- `chapter:finding-alphas-data-alpha-design-2019` — understand and sanity-check data before designing an Alpha; information-rich data can inspire multiple distinct hypotheses.
- `chapter:finding-alphas-triple-axis-plan-2019` — distinguish Ideas & Datasets, Performance Parameters, and Regions & Universes instead of treating all variants as the same kind of research.
- `textbook:ilmanen-expected-returns-2011` — *Expected Returns*: rational risk/liquidity premia, behavioral explanations, time-varying expected returns, and balancing historical evidence, theory, and current/forward-looking conditions.
- `author:aqr-ilmanen-expected-returns-framework-2011` — author-side summary corroborating the three-input framework and the distinction between rational and behavioral return drivers.

Expected Returns is a general expected-return framework, not a WorldQuant rulebook. Its role here is methodological: improve the economic content and falsifiability of an Alpha hypothesis before BRAIN tests it.

## I — Interpretation

The current failure mode this Skill targets is not “bad syntax”; it is a weak research question that happens to compile. A field plus `rank(ts_delta(...))` is not an Alpha hypothesis. The hypothesis must say what information the field carries, why that information may predict relative returns, on what horizon, against what competing explanation, and what result would reduce belief in the idea.

Separate four things:

1. **Information source** — what new observation enters the hypothesis.
2. **Expected-return driver** — risk/liquidity compensation, behavioral under/overreaction, information diffusion, valuation/carry, positioning/crowding, or another explicit mechanism.
3. **Implementation** — the smallest transform/expression that tests the mechanism.
4. **Simulation settings** — delay, decay, neutralization, truncation, universe; these are not the mechanism.

## A1 — Past application

A cash-flow field should not become `ts_delta` merely because delta is available. First ask whether the hypothesis is valuation, improving cash-generation quality, analyst revision, or another mechanism; each implies a different direction, horizon and competing explanation.

Likewise, a high-turnover price/volume near-miss should not automatically acquire a regime filter after seeing the metrics. A regime condition is justified only if the mechanism predicted before testing that the signal should strengthen or weaken in that state.

## A2 — Future trigger

Invoke before `wq-alpha-hypothesis` whenever generating new Alpha ideas. It is especially important when:

- recent clean Skill-first rounds show high `low_fitness` concentration;
- Trial → Candidate conversion is weak;
- the proposed batch consists mostly of window/operator variants;
- the same dataset is being reused without new economic questions;
- inventory is in REPLENISHMENT and semantic breadth matters more than local parameter refinement.

For a failed existing Alpha, use `wq-failure-diagnosis` first; this Skill may be used again only if allocation routes to `NEW_HYPOTHESIS`.

## E — Execution

1. Query `kb_search_alpha_knowledge` for the target data concept plus `economic mechanism expected return risk behavioral liquidity information diffusion valuation current condition`. Preserve useful card IDs.
2. Read Research Memory before proposing the slate. Identify overused families/operator patterns and recent `low_fitness` clusters so the new hypotheses add semantic coverage rather than more cosmetic siblings.
3. For each proposed idea, create an `economic_case` with:
   - `information_source`;
   - `data_semantics` — what the field/observation means in market or corporate terms;
   - `expected_return_driver` — one explicit leading mechanism;
   - `mechanism` — causal/behavioral story in one or two sentences;
   - `expected_direction`;
   - `horizon_delay`;
   - `competing_explanation` — credible alternative when one exists;
   - `common_factor_exposure` — likely market/sector/style/risk exposure or `unknown`;
   - `state_dependency` — predeclared current/regime condition or `none`;
   - `falsifiers` — 1–3 observations that should reduce belief in the idea;
   - `evidence_inputs` with `historical`, `theory`, and `forward_looking_current_condition`; unavailable evidence must be marked `unknown`, never invented;
   - `knowledge_card_ids`.
4. Prefer hypotheses for which the mechanism changes what experiment should be run. If two stories would produce the same expression, same horizon and same falsifier, they are not meaningfully distinct yet.
5. During REPLENISHMENT, diversify first across **Ideas & Datasets / information sources / mechanisms**. Do not fill a batch with lookback, decay or truncation siblings. Parameter refinement belongs later to `wq-experiment-allocation` after semantic promise exists.
6. Treat risk/liquidity and behavioral explanations as competing hypotheses when appropriate. For example, a signal that looks like mispricing may actually be compensation for an undesirable factor exposure; record which neutralization or horizon test could distinguish them.
7. Predeclare state dependence. Only attach a volatility/liquidity/regime condition when the mechanism predicts why it should matter. Never scan many regimes after a weak result and then retrofit the story.
8. Return 3–8 `economic_case` objects ordered by research value. Do not emit FASTEXPR yet. Route them to `wq-alpha-hypothesis`, which must validate live BRAIN fields/operators and translate each case into the smallest expression.

## B — Boundary

- Do not claim Expected Returns describes official BRAIN thresholds or short-horizon Alpha rules.
- Do not force every Alpha into a risk-premium story; information diffusion, revisions, positioning, reversion and other mechanisms are allowed when explicit and falsifiable.
- Do not invent historical evidence, current conditions, or field meaning.
- Do not use economic narrative to rescue an expression after seeing poor metrics; the core case must precede Simulation.
- Do not make the expression complex merely to make the story sound sophisticated.
- Do not confuse universe/neutralization/decay changes with a new economic hypothesis.

## Related skills

- `wq-alpha-hypothesis`: translates approved economic cases into live-supported FASTEXPR.
- `wq-alpha-review`: checks semantic coherence before BRAIN budget is spent.
- `wq-failure-diagnosis`: interprets real failures before another experiment.
- `wq-experiment-allocation`: decides whether to repair, diversify or generate a new hypothesis.

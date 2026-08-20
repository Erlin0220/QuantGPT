# 06 — Make Replenishment Grow the Existing Candidate Inventory

**What to build:** Make Replenishment Mode actively turn research budget into new Research Candidates in the existing Candidate Inventory even after the day's formal submission slots are full. It must combine fresh hypotheses and promising Directed Mutation parents, use cold-start-safe candidate eligibility, and keep submission control completely separate.

**Blocked by:** 01 — Route Metric Failures into Directed Mutation; 03 — Gate Sparse ACTIVE and Points Learning; 04 — Use Coverage-First Cold-Start Scheduling.

**Status:** resolved

- [ ] When Candidate Inventory is below its Inventory Floor, autonomous research continues with `remaining_submission_slots = 0` and does not stop merely because the daily Live Submission target is complete.
- [ ] Replenishment research budget can include both fresh hypotheses and repairable/near-threshold parents selected from research memory.
- [ ] Qualifying results are persisted into the existing Candidate Inventory; no parallel queue/database is introduced.
- [ ] Cold-start Candidate Inventory eligibility uses deterministic research-readiness evidence and is not blocked solely because empirical ACTIVE probability is uncalibrated or its S/A/B tier is low.
- [ ] Candidate Inventory status always returns populated high-confidence/eligible counts, tier/readiness breakdown and deficit while Replenishment Mode is active; it must not regress to `null` deficit/count fields.
- [ ] Formal submission remains governed exclusively by the existing Submission Gate/daily budget and Replenishment Mode cannot create extra submissions.
- [ ] Research remains single-flight/restart-safe and does not launch overlapping autonomous research jobs to compensate for inventory deficit.
- [ ] An end-to-end fake-BRAIN test demonstrates zero remaining formal submission slots plus inventory deficit can still produce and persist at least one new eligible Research Candidate.
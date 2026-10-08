# Training data

Every dataset a D1A model can train on, what it teaches, and which runs used how much. A model serves about 13 use
cases with one set of weights, so it drifts away from whatever a fine-tune leaves out.

**Two rules** (John, 2026-10-08):
- **All gets in.** Every fine-tune trains on or replays every source below. The mix tools refuse a mix that leaves one
  out (#194), and so does `d1a.training.train` (#211).
- **All gets tested.** Every new checkpoint is gated on every suite (`scripts/quality_gate.py --all-suites`) and judged
  by `scripts/decide.py`, which answers INCOMPLETE if anything is missing (#210).

## Sources

`d1a.eval.suites.train_sources()` lists them; decision-v7 is replayed by the trainer itself
(`--suite evals/v7/decision-v7 --replay N`) or mixed in.

| source | train records | what it teaches | serves |
|---|---|---|---|
| `evals/hard-v1` (Kev; programmatic labels) | 6,000 | seven reasoning families of about 857 records each: long policies, trade-offs, probability, multi-hop, temporal and numeric, judging, ambiguous | gate, control, evals, reasoning-heavy decisions |
| `evals/devtools-v1` (Kev) | 5,320 | code review (codereviewer 1,500), commit intent (commitpackft 1,500), safety triage (aegis 1,500), flaky tests (flakeflagger 820) | tool-gate, guardrails, PR work |
| `evals/documents-v1` (Kev) | 5,219 | long-document classification: CFPB complaints by product and issue | inbox, bulk, document routing |
| `evals/v7/decision-v7` (Kev) | 12,576 | general decisions: topic (agnews, dbpedia14), sentiment (imdb, yelp, amazon, sst5), intent (banking77), question type (trec), yes/no reading (boolq), inference (mnli), legacy policy (896), compositional (1,680) | the base skill behind every use case; it also holds the suites that have no training data (transfer-v4) |
| `evals/d1a/pr-labels:train` (`JohnP1/d1a-pr-labels`) | 7,563 | PR type, blast radius and severity | PR labeler |
| `evals/d1a/pr-labels:train-ja` | 768 | the same, as Japanese twins of English PRs | PR labeler (Japanese) |
| `evals/d1a/pr-labels:train-blast` | 2,185 | extra blast-radius labels from an older era with a different severity rubric: use at natural shares, not more (#187) | PR labeler (blast) |
| `evals/d1a/routing:factory-train` (`JohnP1/d1a-routing`) | 1,216 | routing requests to the right handler, factory domain | routing |
| `evals/d1a/routing:generic-train` | 1,529 | routing, general domain | routing |
| `evals/d1a/ja-jglue:train` (`JohnP1/d1a-ja-jglue`) | 6,000 | Japanese understanding (JGLUE) | every Japanese use case |

**Never trained on** (evaluation only):
- every development and test partition;
- `transfer-v4`, `night2` and the `external` suites;
- the owner's hand-checked PR labels, the labeler replay kit, and the Mac mini's decision log (privacy: weights get
  published).

## What each run used

| source | v0.5 (2026-10-06) | phase C (vetoed, #187) | D2-skills (#197) |
|---|---|---|---|
| hard-v1 | 6,000 | 500 | 1,500, weighted toward the weak families |
| devtools-v1 | 5,320 | 500 | 1,300 |
| documents-v1 | 1,000 | 120 | 450 records no earlier run used |
| decision-v7 | 1,000 | 160 | 1,000 records unused since v0.1 |
| pr-labels train / train-ja / train-blast | 1,350 / 0 / 0 | 800 / 250 / 1,800 | 1,350 / 120 / 105 |
| routing factory-train / generic-train | 0 / 0 | 125 / 125 | 650 / 150 |
| ja-jglue train | 650 | 120 | 300 |
| total | 15,320 | 4,500 | 6,925 |

What the runs taught about the mix:
- **v0.5 left routing out**, and its routing confidence drifted (factory ECE 0.073 → 0.130). The rule then (#167) named
  only the three skill suites.
- **Phase C gave PR labels 66% of the mix**, most of it train-blast. The model learned a skewed label prior: severity P1
  predictions went from 6 to 48 on the test PRs, and the PR suites got worse. With 160 decision-v7 records, transfer-v4
  also drifted, by −2.1.
- So every source must be present, at deliberately chosen shares: close to the real ones for labels, and enough
  decision-v7 to hold the suites with no training data.

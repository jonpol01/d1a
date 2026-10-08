### Added

- `scripts/decide.py` judges a new checkpoint on one scorecard (#202). It reads the files `scripts/quality_gate.py
  --all-suites` writes, plus the live labeler check's answers. It scores every suite and the two live labeler lines,
  pooled with a 95% interval, and gives one verdict:
  - INCOMPLETE when anything is missing;
  - VETO on the card-suite floor, latency, or a large suite 5 points down;
  - BETTER on a pooled lower bound above 0, or 3 significant wins on distinct sources with no loss;
  - NOT BETTER otherwise, a near miss when the pooled Δ > 0 with 2 wins and no loss.
  Every significant loss is listed as a follow-up. AGENTS.md and `recipes/README.md` make it the rule for a new
  checkpoint.

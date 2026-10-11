### Added

- Served decode rules (`rules.<name>` in the `D1A_LEARNING` settings, `d1a.learning.rules`): for a request that sends the
  rule's `use_case` to the run it was fitted for, the served choice of one question becomes the rule's option when the
  model's choice is one of `from` and its probability of that option reaches `t`, and the answer names the rule
  (`rule`: name, from, p, t). Only the choice moves: the
  probabilities, the confidence and the decision log's answers stay the model's, and the log records what was served
  (`meta.served`, `meta.rules`). Written for the live PR labeler: severity's P3/P4 answers are served as P2 when
  p(P2) >= 0.20, which brought D1A-E4B v0.6 closer to hermes-agent's written label definitions on held-out pull requests
  (#252). A client that does not send the use case gets exactly the answers it got before.
- `scripts/quality_gate.py --head-learning <file> --labeler-use-case <name> --expect-flips <question>`: gates a decode
  rule; the head server reads the settings file, the labeler replay sends the use case, flips on that question are listed
  with the probability that crossed the threshold, and every other change still fails.

### Changed

- The model server reads the `D1A_LEARNING` settings file for its `flags` and `rules`; the quality gate strips that
  variable from both servers, like the other learning settings. `feedback status` reports the served choice beside the top
  one when a rule moved it.

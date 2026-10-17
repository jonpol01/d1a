### Changed

- A decision that arrives while an OpenAI chat generates is answered between two of the chat's tokens, not after the
  whole answer (#289): measured on D1A-E4B-it, 0.24 s instead of 4–5 s during a 200-token chat. The chat's text and the
  decision's probabilities are unchanged. `MLXDecisionModel.adapter()` now nests (it restores the switch it found).

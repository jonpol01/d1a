### Changed

- A decision that arrives while a chat is still reading a long prompt is answered between two prefill chunks (2,048
  tokens, mlx-lm's default), not after the whole prefill (#295): on D1A-E4B-it with a ~6,000-token prompt, 3.6-4.2 s instead
  of 8.8-9.4 s. The chat's text and the decision's probabilities are unchanged.

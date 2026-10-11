### Changed

- `d1a.core.encoding` and `d1a.training.data` import `transformers` and `datasets` only where they are used (#254), so
  modules that import them without loading a tokenizer or a public dataset start in a fraction of a second: `import
  d1a.eval.suite` 4.25 s → 0.07 s. The server still loads both at startup, as before.

### Changed

- Docs: the wording matches what the code does (#424). D1A gives agents fail-safe advice and the caller acts on it,
  except the Claude Code gate hook and, with `D1A_ROUTES` set, the chat endpoint. Decision logging is opt-in, a retrain
  is run by hand, embeddings come from a separate model, and photos, voice and video are a demo rather than a measured
  result. Model routing joins "Proven at" with its acceptance numbers.

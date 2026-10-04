---
title: D1A
emoji: 🎯
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
license: apache-2.0
short_description: Multiple-choice decisions with probabilities, in one pass
models:
  - JohnP1/d1a-e4b
  - google/gemma-4-E4B
tags:
  - decision-model
  - classification
  - gemma4
---

# D1A

D1A answers typed questions about a document (choice, yes/no, score) with a probability for every option, in one
forward pass. This Space runs D1A-E4B v0.4 on ZeroGPU: model routing, guardrails, tool-call gating, the PR labeler,
and zero-shot photo and voice checks through Gemma 4's own encoders.

- Code: [jonpol01/d1a](https://github.com/jonpol01/d1a) (this Space is `space/` there)
- Model: [JohnP1/d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b)
- All 13 demos: [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground)

Built on [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer (Apache-2.0). D1A is an independent project, not
affiliated with or endorsed by the Kev authors.

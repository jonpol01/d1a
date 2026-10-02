# Apple's on-device stack (macOS 27 / iOS 27) and D1A

October 2026. Part of [#75](https://github.com/jonpol01/d1a/issues/75) (one portable phone runtime).

**Question.** macOS 27 and iOS 27 bring two new things: Core AI, the framework for running your own models, and a larger
Foundation Models framework, Apple's built-in model. Does either replace D1A, or give D1A a better way onto iPhone and Mac?

**Short answer.** Neither replaces D1A. Foundation Models cannot give calibrated probabilities, which are D1A's whole job.
Core AI could become a good iPhone runtime for D1A, but not yet: Apple's Gemma 4 support for iOS is still unfinished, and
an open iOS 27 bug makes Gemma 4 process a prompt one token at a time, while D1A's speed comes from reading the whole
prompt in one pass. Keep MLX as the Apple path now, re-test Core AI when those land, and use Foundation Models as one of
the models D1A routes *to*.

## Foundation Models (Apple's built-in model)

What is new in 27: image input, a public `LanguageModel` protocol so a session can run on Apple's model, Private Cloud
Compute, MLX or Core AI models, or partner packages (Anthropic, Google); *dynamic profiles* that switch model, tools and
instructions inside one session; LoRA adapters for Apple's model; system tools (OCR, barcodes, Spotlight search).

| D1A needs | Foundation Models in 27 |
|---|---|
| A probability for every option, calibrated | No token probabilities or confidence scores; a `@Generable` enum returns the chosen case only |
| Hidden states at `<decide>` and `</opt>` for the pointer head | Not exposed; the `LanguageModel` protocol is for text generation |
| Our own fine-tuning | LoRA adapters for Apple's model, which still generate text; no head |
| Android, Linux servers, open weights | Apple platforms only |

So D1A cannot be built on Apple's model, and Apple's model cannot do D1A's gating (act / confirm / ask a human, or
"send this up a tier") because there is no number to put a threshold on.

**Where it fits: as a routing target.** Dynamic profiles plus the `LanguageModel` protocol make "small model or large
model" a one-line switch inside an app: on-device model, Private Cloud Compute, or Claude. D1A is the part that decides
which, with a probability. An iOS demo of D1A routing between Apple's on-device model and a larger one is the natural
first app.

## Core AI (running our own model)

Core AI replaces Core ML: models are exported from PyTorch (`coreai-torch`) to `.aimodel` files that the OS specializes
and compiles ahead of time for the chip, including the Neural Engine. Apple's
[coreai-models](https://github.com/apple/coreai-models) repo has recipes for Qwen, Mistral, Gemma 3n and others.

In principle this is a good home for D1A on iPhone: export one graph (Gemma 4 text model with the LoRA merged, plus the
pointer head reading the delimiter positions), let the OS place it on the GPU or Neural Engine, and check it against
D1A's golden vectors like every other port. In practice, for Gemma 4, as of 2 October 2026:

- **The iOS export is not merged.** The Swift runner support landed on 30 September (coreai-models #302); the export
  recipe is an open PR (#301). Gemma 4 E2B/E4B on iOS is also an open model request (#20).
- **Prompt reading is broken on iOS 27** (coreai-models #201, open since August; Apple has filed it internally).
  Any multi-token prefill of Gemma 4 E2B aborts on iPhone 17 Pro with an MPSGraph scratch-heap overflow at chunk
  sizes 64, 32 and 16. Only one token at a time works, so "a 1024-token prompt degrades to per-token processing". D1A
  is all prefill and no generation, so this removes its main speed advantage on that path. The macOS side of the same
  bug no longer reproduces.
- **Large embedding tables read wrong rows silently** (coreai-models #313): a GPU gather from a constant with 2^31 or
  more elements returns wrong rows. Gemma 4 E4B's per-layer-embedding table is above that size. The open export
  avoids it by keeping per-layer embeddings outside the graph as an INT8 file, which is the same idea as our #71.

## What D1A keeps that Apple's stack does not give

- A calibrated probability for every option, from one forward pass, with no generated text to parse.
- The same model and API on Mac, iPhone, Android and a GPU server (System One API, open weights).
- Typed questions (yes/no, choice, score) answered together over one document, which is how the agent kit uses it.

## Recommendation for #75

1. **iPhone and Mac now: MLX.** It works today with batched prefill, and our 8-bit build with 4-bit per-layer
   embeddings already passes the golden-vector parity gate. Continue #71 (per-layer embeddings on flash) and #73
   (smaller builds) there.
2. **Core AI: re-test when coreai-models #301 is merged and #201 is fixed.** Then a half-day spike on a Mac: export
   D1A-E2B to `.aimodel`, check parity against the golden vectors, and compare latency and memory with MLX, then with
   the Neural Engine on an iPhone. Switch only if it wins on both and passes parity.
3. **Use Foundation Models as a routing target, not as a backend.** Build the iOS demo around D1A choosing between
   Apple's on-device model and a larger model through the `LanguageModel` protocol.
4. **Android is unaffected:** LiteRT or llama.cpp as #75 already plans.

## Sources

- WWDC26 session [What's new in the Foundation Models framework](https://developer.apple.com/videos/play/wwdc2026/241/)
- [WWDC26 Machine Learning guide](https://developer.apple.com/wwdc26/guides/machine-learning/)
- InfoQ, [Apple Launches Core AI](https://www.infoq.com/news/2026/06/apple-core-ai-wwdc/)
- apple/coreai-models: [#301](https://github.com/apple/coreai-models/pull/301) (Gemma 4 iOS export, open),
  [#302](https://github.com/apple/coreai-models/pull/302) (runner, merged 2026-09-30),
  [#201](https://github.com/apple/coreai-models/issues/201) (iOS prefill abort, open),
  [#313](https://github.com/apple/coreai-models/issues/313) (gather above 2^31 elements, open),
  [#20](https://github.com/apple/coreai-models/issues/20) (Gemma 4 E4B on iOS request, open)

# D1A

**A small decision model on Gemma 4.** One document and a set of typed questions in, a calibrated probability for every option out, in one forward pass. No text generation.

> Early preview. The name is reserved while the first models are trained; code and weights land here next.

## What it is

D1A answers yes/no, multiple-choice and rating questions about a piece of text, the way an API call answers a function: routing a support ticket, gating an agent's tool call, triaging an inbox, ranking passages, grading an LLM's answer. Every answer comes with probabilities, so your code can act on the confident cases and send the rest to a person.

It is built on [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer (Apache-2.0), with Gemma 4 E2B and E4B backbones, and speaks the same System One API, so the TypeSafe SDK works against it.

## Status

| | |
|---|---|
| Gemma 4 support for Kev | done: [jonpol01/kev](https://github.com/jonpol01/kev) |
| Prototype checkpoint (Gemma 4 E2B, 1 epoch) | [JohnP1/kev-gemma4-e2b](https://huggingface.co/JohnP1/kev-gemma4-e2b) |
| D1A E2B, E4B | training: [JohnP1/d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b), [JohnP1/d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) |
| Live demos of nine use cases | [jonpol01/kev-usecases-poc](https://github.com/jonpol01/kev-usecases-poc) |

## Clients

A minimal client for a running D1A (or Kev) server is in [`python/`](python) (`pip install d1a`) and [`js/`](js) (`npm install d1a`).

```python
from d1a import Client
client = Client("http://127.0.0.1:8009")
answer = client.decide("Shoes arrived late and I was charged twice.",
                       {"team": {"type": "choice", "instructions": "Which team should handle this?",
                                 "criteria": {"returns": None, "shipping": None, "billing": None}}})
print(answer["answers"]["team"]["probabilities"])
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Gemma 4 is licensed by Google under Apache-2.0.

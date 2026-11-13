# d1a-client (Python)

A dependency-free client for D1A, a small decision model on Gemma 4, or any other server that speaks the System One API (`POST /v1/systemone`). The model itself, its trainer and server are in the [D1A repository](https://github.com/jonpol01/d1a), which is built on [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer.

The default model name is `d1a-latest`, the only name a D1A server lists; it answers any model name a request sends.

Not on PyPI yet; install it from the repository (Python 3.9 or newer):

```bash
pip install "d1a-client @ git+https://github.com/jonpol01/d1a#subdirectory=clients/python"
```

```python
from d1a_client import Client

client = Client("http://127.0.0.1:8009")          # api_key="..." if the server sets D1A_API_KEY
answer = client.decide("Shoes arrived late and I was charged twice.",
                       {"team": {"type": "choice", "instructions": "Which team should handle this?",
                                 "criteria": {"returns": None, "shipping": None, "billing": None}}})
print(answer["answers"]["team"]["probabilities"])
```

What a D1A server answers, besides `decide`:

| Call | Endpoint | What it does |
|---|---|---|
| `decide(state, questions, use_case=None, log=True)` | `POST /v1/systemone` | `use_case` (e.g. `"routing"`) reads the answers at the checkpoint's temperature for that use case, if it has one (`use_case_temperatures` in `models()`); without it the request is sent exactly as before. `log=False` keeps the decision out of the server's decision log (tests and demos, where no outcome will follow). |
| `decide_media(media_type, media, questions, state=None, use_case=None, log=True)` | `POST /v1/systemone/media` | The same questions about a photo, a voice clip or a video: `media_type` is `"image"`, `"audio"` or `"video"`, `media` the file's bytes or its path, `state` optional text beside it. Needs a server on an MLX export with its `media/` folder. |
| `feedback(decision_id, labels, src=None, group=None)` | `POST /v1/feedback` | What actually happened after a decision, for the server to learn from. A server started with `D1A_FEEDBACK_LOG` gives every logged answer a `decision_id`; `labels` maps a question id to the outcome (`True`/`False` for yes/no, the option key for a choice). |
| `models()` | `GET /v1/models` | The served model's card: its run, backend, temperature, use-case temperatures and learning settings. |
| `chat(messages, **options)` | `POST /v1/chat/completions` | OpenAI-format chat, answered whole; `options` (`max_tokens`, `temperature`, `tools`, ...) are sent as given. Only a model on an instruction-tuned trunk chats (D1A-E2B-it, D1A-E4B-it). For streaming, use an OpenAI client with `base_url=<server>/v1`. |

```python
answer = client.decide(ticket, questions)
client.feedback(answer["decision_id"], {"team": "billing"}, src="human")   # once you know where it really went

photo = client.decide_media("image", "parcel.jpg",
                            {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}}, state="left at the door")
```

A request the server refuses raises `d1a_client.D1AError`, a `urllib.error.HTTPError` (its body still readable) with the
server's reason in `.detail` (for a malformed request, the list of fields it rejected). The API key is never sent on to a
host a redirect names.

Apache-2.0.

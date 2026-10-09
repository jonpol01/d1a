# d1a-client (Python)

A dependency-free client for D1A, a small decision model on Gemma 4, or any other server that speaks the System One API (`POST /v1/systemone`). The model itself, its trainer and server are in the [D1A repository](https://github.com/jonpol01/d1a), which is built on [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer.

The default model name is `d1a-latest`; a D1A server also accepts `kev-latest` and `jev-latest`.

```bash
pip install d1a-client
```

```python
from d1a_client import Client

client = Client("http://127.0.0.1:8009")          # api_key="..." if the server sets D1A_API_KEY
answer = client.decide("Shoes arrived late and I was charged twice.",
                       {"team": {"type": "choice", "instructions": "Which team should handle this?",
                                 "criteria": {"returns": None, "shipping": None, "billing": None}}})
print(answer["answers"]["team"]["probabilities"])
```

`client.decide(state, questions, use_case="routing")` names the request's use case: a D1A checkpoint with a temperature for it (its `use_case_temperatures` in `GET /v1/models`) reads the answers at that temperature; otherwise the checkpoint's own applies. Without `use_case` the request is sent exactly as before.

Apache-2.0.

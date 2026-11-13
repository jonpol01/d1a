### Added

- The Python and JavaScript clients (`clients/python`, `clients/js`, 0.1.0) cover the rest of a D1A server (#397, #11):
  questions about a photo, a voice clip or a video (`decide_media` / `decideMedia`), outcomes for the server to learn
  from (`feedback`), `log=False` to keep a test request out of the decision log, the served model's card (`models`) and
  OpenAI-format chat (`chat`). A refused request raises `D1AError` with the server's reason (in Python still a
  `urllib.error.HTTPError`). `tests/test_clients.py` checks every request both clients send against the server's own
  request models, and that both send the same requests. `chat` refuses `stream`; stream with an OpenAI client.

### Fixed

- The client READMEs and the usage guide said to `pip install` / `npm install d1a-client`, which are not published:
  they now say how to install from the repository (#397).

### Security

- The Python client no longer sends its API key on to a host a redirect names: urllib copied the `Authorization`
  header onto the redirected request (#397). The JS client's `fetch` already drops it; both are tested.

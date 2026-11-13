# d1a-client (JavaScript)

A dependency-free client for D1A, a small decision model on Gemma 4, or any other server that speaks the System One API (`POST /v1/systemone`). The model itself, its trainer and server are in the [D1A repository](https://github.com/jonpol01/d1a), which is built on [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer.

The default model name is `d1a-latest`, the only name a D1A server lists; it answers any model name a request sends.

Not on npm yet. It is one CommonJS file (`index.js`) for Node 18 or newer, or for a browser app built with a bundler: copy
it into your project, or install it from a checkout with `npm install <checkout>/clients/js`.

```js
const { Client } = require("d1a-client");

const client = new Client("http://127.0.0.1:8009");   // { apiKey } if the server sets D1A_API_KEY
const answer = await client.decide("Shoes arrived late and I was charged twice.", {
  team: { type: "choice", instructions: "Which team should handle this?", criteria: { returns: null, shipping: null, billing: null } },
});
console.log(answer.answers.team.probabilities);
```

From a web page served from another origin, the browser needs the server to allow that origin: start it with
`D1A_CORS_ORIGINS=<the page's origin>`.

What a D1A server answers, besides `decide`:

| Call | Endpoint | What it does |
|---|---|---|
| `decide(state, questions, { useCase, log })` | `POST /v1/systemone` | `useCase` (sent as `use_case`, e.g. `"routing"`) reads the answers at the checkpoint's temperature for that use case, if it has one (`use_case_temperatures` in `models()`); without it the request is sent exactly as before. `log: false` keeps the decision out of the server's decision log (tests and demos, where no outcome will follow). |
| `decideMedia(mediaType, media, questions, { state, useCase, log })` | `POST /v1/systemone/media` | The same questions about a photo, a voice clip or a video: `mediaType` is `"image"`, `"audio"` or `"video"`; `media` is the file's bytes (`Uint8Array`, `Buffer`, `ArrayBuffer`, or a `Blob`/`File` from a file input) or a base64 string; `state` is optional text beside it. Needs a server on an MLX export with its `media/` folder. |
| `feedback(decisionId, labels, { src, group })` | `POST /v1/feedback` | What actually happened after a decision, for the server to learn from. A server started with `D1A_FEEDBACK_LOG` gives every logged answer a `decision_id`; `labels` maps a question id to the outcome (`true`/`false` for yes/no, the option key for a choice). |
| `models()` | `GET /v1/models` | The served model's card: its run, backend, temperature, use-case temperatures and learning settings. |
| `chat(messages, options)` | `POST /v1/chat/completions` | OpenAI-format chat, answered whole; `options` (`max_tokens`, `temperature`, `tools`, ...) are sent as given. Only a model on an instruction-tuned trunk chats (D1A-E2B-it, D1A-E4B-it). For streaming, use an OpenAI client with `baseURL: "<server>/v1"`. |

```js
const answer = await client.decide(ticket, questions);
await client.feedback(answer.decision_id, { team: "billing" }, { src: "human" });   // once you know where it really went

const photo = await client.decideMedia("image", fileInput.files[0],
  { damaged: { type: "noul", instructions: "Is the parcel damaged?" } }, { state: "left at the door" });
```

A request the server refuses throws a `D1AError` (exported beside `Client`) with `.status` and the server's reason in
`.detail` (for a malformed request, the list of fields it rejected). `fetch` drops the API key on a redirect to another origin.

Apache-2.0.

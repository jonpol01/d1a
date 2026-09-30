# d1a-client (JavaScript)

A dependency-free client for D1A, a small decision model on Gemma 4, or any other server that speaks the System One API (`POST /v1/systemone`). The model itself, its trainer and server are in the [D1A repository](https://github.com/jonpol01/d1a), which is built on [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer.

The default model name is `d1a-latest`; a D1A server also accepts `kev-latest` and `jev-latest`.

```bash
npm install d1a-client
```

```js
const { Client } = require("d1a-client");

const client = new Client("http://127.0.0.1:8009");   // { apiKey } if the server sets D1A_API_KEY
const answer = await client.decide("Shoes arrived late and I was charged twice.", {
  team: { type: "choice", instructions: "Which team should handle this?", criteria: { returns: null, shipping: null, billing: null } },
});
console.log(answer.answers.team.probabilities);
```

Apache-2.0.

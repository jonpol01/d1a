// Client for a D1A (or Kev) System One server: typed questions in, a probability per option out.
class Client {
  constructor(baseUrl = "http://127.0.0.1:8009", { apiKey, model = "d1a-latest" } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, ""); this.apiKey = apiKey; this.model = model;
  }
  async decide(state, questions) {
    const headers = { "content-type": "application/json" };
    if (this.apiKey) headers.authorization = `Bearer ${this.apiKey}`;
    const r = await fetch(`${this.baseUrl}/v1/systemone`, { method: "POST", headers, body: JSON.stringify({ state, model: this.model, questions }) });
    if (!r.ok) throw new Error(`D1A server ${r.status}: ${await r.text()}`);
    return r.json();
  }
}
module.exports = { Client };

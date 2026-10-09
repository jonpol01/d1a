// Client for a D1A (or Kev) System One server: typed questions in, a probability per option out.
class Client {
  constructor(baseUrl = "http://127.0.0.1:8009", { apiKey, model = "d1a-latest" } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, ""); this.apiKey = apiKey; this.model = model;
  }
  // useCase (e.g. "routing"), sent as use_case only when given: a D1A server reads the answers at the checkpoint's
  // temperature for that use case, if it has one.
  async decide(state, questions, { useCase } = {}) {
    const headers = { "content-type": "application/json" };
    if (this.apiKey) headers.authorization = `Bearer ${this.apiKey}`;
    const request = { state, model: this.model, questions };
    if (useCase != null) request.use_case = useCase;
    const r = await fetch(`${this.baseUrl}/v1/systemone`, { method: "POST", headers, body: JSON.stringify(request) });
    if (!r.ok) throw new Error(`D1A server ${r.status}: ${await r.text()}`);
    return r.json();
  }
}
module.exports = { Client };

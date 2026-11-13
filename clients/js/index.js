// Client for a D1A (or Kev) System One server: typed questions in, a probability per option out.
const LOG_HEADER = "x-d1a-decision-log"; // "off": a D1A server answers but keeps the decision out of its decision log

// A request the server refused: its status and its reason (FastAPI's "detail").
class D1AError extends Error {
  constructor(status, detail) {
    super(`D1A server ${status}: ${detail}`);
    this.name = "D1AError"; this.status = status; this.detail = detail;
  }
}

class Client {
  constructor(baseUrl = "http://127.0.0.1:8009", { apiKey, model = "d1a-latest" } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, ""); this.apiKey = apiKey; this.model = model;
  }
  // useCase (e.g. "routing"), sent as use_case only when given: a D1A server reads the answers at the checkpoint's
  // temperature for that use case, if it has one. log: false keeps the decision out of the server's decision log.
  async decide(state, questions, { useCase, log = true } = {}) {
    return this.#send("/v1/systemone", this.#request(state, questions, useCase), log);
  }
  // The same questions about a photo, a voice clip or a video: mediaType "image", "audio" or "video"; media is the
  // file's bytes (Uint8Array, Buffer, ArrayBuffer, Blob or File) or a base64 string. state: optional text beside it.
  async decideMedia(mediaType, media, questions, { state = null, useCase, log = true } = {}) {
    const request = this.#request(state, questions, useCase);
    request.media = { type: mediaType, data: await base64(media) };
    return this.#send("/v1/systemone/media", request, log);
  }
  // What actually happened after a decision: decisionId is the one its answer carried (a server started with
  // D1A_FEEDBACK_LOG adds it); labels maps a question id to the outcome (true/false for yes/no, the option key for a choice).
  async feedback(decisionId, labels, { src, group } = {}) {
    const body = { decision_id: decisionId, labels };
    if (src != null) body.src = src;
    if (group != null) body.group = group;
    return this.#send("/v1/feedback", body);
  }
  // The served model's card (run, backend, temperature, use-case temperatures, learning settings).
  async models() {
    return this.#send("/v1/models");
  }
  // OpenAI-format chat, answered whole; options (max_tokens, temperature, tools, ...) are sent as given. Only a model
  // on an instruction-tuned trunk chats (D1A-E2B-it, D1A-E4B-it).
  async chat(messages, options = {}) {
    // the server would answer as an event stream, which this client does not read
    if (options.stream) throw new Error("chat() answers whole; to stream, use an OpenAI client with baseURL: <server>/v1");
    return this.#send("/v1/chat/completions", { ...options, messages });
  }
  #request(state, questions, useCase) {
    const request = { state, model: this.model, questions };
    if (useCase != null) request.use_case = useCase;
    return request;
  }
  async #send(path, body, log = true) {
    const headers = {};
    if (body !== undefined) headers["content-type"] = "application/json";
    if (this.apiKey) headers.authorization = `Bearer ${this.apiKey}`;
    if (!log) headers[LOG_HEADER] = "off";
    const init = body === undefined ? { headers } : { method: "POST", headers, body: JSON.stringify(body) };
    const r = await fetch(`${this.baseUrl}${path}`, init);
    if (!r.ok) throw new D1AError(r.status, detailOf(await r.text()));
    return r.json();
  }
}

// FastAPI's "detail" off an error body (a string, or a validation error's list, as JSON), else the body's text.
function detailOf(text) {
  try {
    const detail = JSON.parse(text).detail;
    if (detail !== undefined) return typeof detail === "string" ? detail : JSON.stringify(detail);
  } catch {}
  return text.trim();
}

async function base64(media) {
  if (typeof media === "string") return media;
  if (typeof Blob !== "undefined" && media instanceof Blob) media = await media.arrayBuffer();
  const bytes = media instanceof Uint8Array ? media : new Uint8Array(media);
  if (typeof Buffer !== "undefined") return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString("base64");
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

module.exports = { Client, D1AError };

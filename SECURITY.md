# Security Policy

## Supported versions

| Version | Security fixes |
| --- | --- |
| `main` | Yes |
| 0.2.x (latest release) | Yes |
| Earlier versions | No (0.1.0 was never released) |

Fixes ship in a patch release of the latest minor version. Before 1.0, a fix that cannot be backported goes into the next minor release, and its release notes say so.

Model checkpoints on Hugging Face (`JohnP1/d1a-*`) are covered at their latest version tag. A finding in an older tag is fixed in a new tag, and the affected tag's model card is updated.

## Reporting a vulnerability

**Do not open a public issue, pull request or discussion for a security problem.**

Report it privately through GitHub: **[Security → Report a vulnerability](https://github.com/jonpol01/d1a/security/advisories/new)**. Only the maintainers can read the report, and GitHub can assign a CVE once it is confirmed.

A useful report includes:

- the affected component (`d1a.serve`, `d1a.media`, `d1a.mcp_server`, training, checkpoint loading, a Hugging Face model) and the version or commit;
- what an attacker can do, and what they need first (network access to the server, a crafted checkpoint, a crafted request);
- steps to reproduce, ideally a minimal request or script;
- any fix or mitigation you suggest.

Please use test data. Don't send real user data, credentials or private model weights.

## What happens next

| Step | Target |
| --- | --- |
| Acknowledgement | within 3 business days |
| Triage: confirmed or declined, with a severity | within 7 days |
| Status updates | at least every 14 days until resolved |

Severity uses [CVSS v4.0](https://www.first.org/cvss/v4-0/). Fix targets after confirmation:

| Severity | Fix released within |
| --- | --- |
| Critical | 7 days |
| High | 30 days |
| Medium | 90 days |
| Low | next regular release |

If a report is declined, we explain why, for example that it is out of scope (below) or that it needs access an attacker would not have.

## Coordinated disclosure

We follow coordinated vulnerability disclosure:

- We publish a GitHub Security Advisory when the fix is released, crediting the reporter unless you ask us not to.
- We ask you to keep the details private until then, or for **90 days** from your report, whichever comes first.
- If a fix needs more time, we agree a new date with you.
- A problem already being exploited may be disclosed sooner, together with a mitigation.

## Safe harbor

We will not pursue or support legal action against anyone who, in good faith:

- tests only their own installation of D1A (or one they are authorized to test), not other people's servers;
- avoids privacy violations, data destruction and service disruption;
- reports through the private channel above and gives us reasonable time to fix before disclosing.

If a third party takes legal action against you for research done under this policy, we will make it known that your actions followed it.

## Scope

**In scope:**
- **The code in this repository:**
  - the model server (`d1a.serve`, `d1a.media`) and its API, including the optional bearer-token auth (`D1A_API_KEY`);
  - the MCP server;
  - checkpoint and export loading (`d1a.checkpoint`, `d1a.mlx_model`);
  - training and evaluation scripts;
  - the release and CI workflows.
- **The published model checkpoints** (`JohnP1/d1a-*`), for example a file that executes code on load.
- **Dependency vulnerabilities** that are reachable through D1A. Dependabot and CodeQL cover the rest; report a reachable one here.

**Out of scope:**
- **Wrong or overconfident answers.** D1A outputs calibrated probabilities, not guarantees. A wrong decision is a model-quality issue: open a normal issue. Prompt-injection text in a document that shifts an answer is in scope only if it escapes the request, for example by forging the delimiter tokens D1A reserves.
- **Upstream projects:** vulnerabilities in Gemma 4, Kev, PyTorch, MLX, transformers or other upstream code that D1A does not make reachable. Report those to their maintainers.
- **Unauthenticated access to a deployment you configured to be open.** By default the server binds to 127.0.0.1 and is open; see the security model below.
- **Denial of service by sending many valid requests.**

## Security model

What D1A assumes, so you can tell a vulnerability from a configuration choice:

- **The server is local by default.**
  - `d1a.serve` and `d1a.media` bind to `127.0.0.1` and need no credentials.
  - Binding to another interface (`--host 0.0.0.0`) exposes them to that network. Set `D1A_API_KEY` to require a bearer token, and put TLS in front (a reverse proxy).
  - Request size and length are bounded: states are truncated to the serving limit; media is limited to 12 MB and 30 s of audio.
- **Checkpoints are code you choose to run.**
  - Load checkpoints only from sources you trust. D1A reads `head.pt` with PyTorch's `torch.load`, which defaults to weights-only loading in the supported PyTorch versions (2.6 and later), and it reads weights from safetensors.
  - Report any way a crafted checkpoint, export folder or Hub repo runs code or reads files outside its own folder.
- **Answers are inputs to your decisions, not access control.**
  - Use D1A's probabilities with thresholds and a human for decisions that matter. The PR labeler's `review:needs-human` below p 0.7 is one example.
  - Do not let a D1A answer alone grant permissions.
- **Secrets stay out of the repository.**
  - Hugging Face tokens, API keys and webhook URLs belong in the environment or a secret store.
  - GitHub secret scanning and push protection are on for this repository.

## Recognition

We credit reporters in the published advisory and the release notes, unless you prefer to remain anonymous. There is no paid bug bounty.

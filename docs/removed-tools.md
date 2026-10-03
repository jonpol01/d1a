# Removed tools

Tools D1A inherited from Kev and removed. They were useful ideas, so this records what each did and how to bring it back,
in our own code or by restoring the file from git:

```bash
git show 1bce9a4:playground/src/components/playground.tsx   # the last commit that has them
git checkout 1bce9a4 -- playground tools/review              # or restore the whole folders
```

The demo app D1A uses is [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground).

## `playground/`: the developer playground (removed October 2026)

A Next.js page for trying requests by hand against a running `d1a.serve` (proxied under `/d1a`).

- **Free-form request editor.** A preset picker plus two text boxes:
  - the state, as plain text or JSON;
  - the questions, as JSON in the System One shape.
  
  It sends one `POST /v1/systemone` and shows each answer with its probabilities and the request's latency.
- **Packed vs separate.** It asks all questions in one request (packed) and each question in its own request
  (`POST /v1/systemone/separate`), then shows the two answers side by side with the probability difference per option.
  This checks that packing questions together does not change the answers.
- **Option-order permutation.** For one Choice question, it re-asks under 6 option orders (`POST /v1/systemone/permute`)
  and shows each order's probabilities, whether the top answer stays the same (`argmax_stable`), and the per-option
  spread. This checks position bias.
- **Chess** (`/chess`, `chess.js`): every move is a Choice question.
  - The state is the board (FEN plus move list), the options are the legal moves (SAN), and a Score question rates
    who is better, all in one request.
  - Modes: model vs model, you as White, you as Black.
  - It shows the top 8 moves with probabilities, and saves games in the browser.

**Still in D1A:** the `/v1/systemone/separate` and `/v1/systemone/permute` endpoints in `d1a.serve` stay, so a new
page can be built on them without server work.

## `tools/review/`: the label-review page (removed October 2026)

A keyboard-first Vite page for one person to accept, relabel or drop AI-proposed labels, one item at a time. It runs
entirely in the browser and autosaves to localStorage, keyed by a hash of the file.

- **Input:** a `.jsonl` file, one item per line:
  `{id, document, source, question: {type, instructions, options}, proposed_label, label_origin, judges: [{model, label, rationale}], adjudication}`.
- **Keys:**

  | Key | Action |
  |---|---|
  | `Enter` / `y` | accept the proposed label |
  | `1`–`9`, `0`, `a`–`z` | relabel |
  | `x` | drop |
  | `n` | note |
  | `←` / `→` | move without deciding |
  | `u` | undo |
  | `e` | export |
  
- **Filters:** undecided, all, disagree (a judge differs from the proposed label), decided. Spot-check mode is a seeded
  shuffle (seed 42) of the first N items (default 50).
- **Export:** `reviews.jsonl`, one line per decided item:
  `{id, verdict: accept|relabel|drop, label, proposed_label, note, reviewed_at}`. The header shows the agreement rate,
  accept / (accept + relabel).

A good fit for reviewing D1A training labels, for example the PR-labeler data or Visionkit OK/NG photos, if we need
that again.

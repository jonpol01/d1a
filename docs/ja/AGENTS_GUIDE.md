<p><a href="../AGENTS_GUIDE.md">English</a> · 日本語 · <a href="../ko/AGENTS_GUIDE.md">한국어</a></p>

> この文書は英語版 [AGENTS_GUIDE.md](../AGENTS_GUIDE.md)（コミット 9eb86269 時点）の翻訳です。内容が異なる場合は英語版が正しい情報です。

<a id="d1a-for-agents"></a>

# エージェント向けの D1A

エージェントは、本来の作業の合間に多くの小さな意思決定を行います。誰がリクエストを担当するか、どの規模のモデルが必要か、
ツール呼び出しを安全に実行できるか、タスクが完了したか、といったものです。D1A はそれぞれを 1 回のフォワードパスで（約 0.2–1 s）、
すべての選択肢に対する較正済み確率とともに答えます。デフォルトでは、確信のない回答は安全なアクションになります: 上位に回す、人に確認する、
またはタスクを開いたままにする、のいずれかです。

<a id="what-d1a-decides"></a>

## D1A が判断すること

| 意思決定 | プリセット | 回答 | フェイルセーフなアクション |
|---|---|---|---|
| **Intake:** 新しいリクエスト | `intake` | どのワーカーか、モデルのサイズ、対象が明示され完了の姿が書かれているか、先にユーザーに確認するか | 対象または完了状態が欠けているように見える場合はユーザーに確認する |
| **Tier:** 作業開始前のタスク | `tier` | 小・中・大のどのモデルか | D1A が小さいモデルで十分だと確信しない限り、上位に回す |
| **Route:** 任意のプロンプト | `route` | 小・中・大のどのモデルか | 同上 |
| **Gate:** ツール呼び出し | `gate` | allow、ask、deny | p ≥ 0.8 の場合のみ allow、p ≥ 0.5 で deny、それ以外は ask |
| **Judge:** タスクとワーカーの報告 | `judge` | 完了か。ブロックされているか。人が必要か。次: complete、rework、consult、repair | D1A が完了を確信した場合にのみタスクを閉じる |

質問は `d1a/agents/presets.py` にあり、D1A のルーティングおよびエージェントの学習データが尋ねるとおりの文言になっています。
フェイルセーフの規則は同じファイルの `advise()` です。

<a id="see-it-the-demo"></a>

## 実際に見る: デモ

```bash
python -m d1a.agents.demo              # against the D1A server at D1A_URL (default http://127.0.0.1:8009)
```

このデモの意思決定はサーバーの学習ログに残らないため、稼働中のサーバーに対して実行しても安全です。

1 つのタスクがエージェントの 1 日を通過します。以下は、Apple M4 上で提供した D1A-E4B-it v0.6 の実際の出力です（レイテンシには
SSH トンネルが含まれます）:

```
[intake] A new request: who takes it, how large a model, ask the user first?
  input:   Fix the failing login test in ~/work/app (branch fix-login). Done when tests/test_login.py passes.
  D1A:     worker=developer (1.00), tier=medium (0.97), has_target=yes (0.86), has_spec=yes (0.92), clear_done=yes (0.98), consult=no (0.06)
  advice:  {"consult": false, "worker": "developer", "tier": "medium"}

[intake] A vague request
  input:   Make the app better.
  D1A:     worker=developer (0.91), tier=large (0.53), has_target=no (0.05), has_spec=no (0.02), clear_done=no (0.04), consult=yes (0.97)
  advice:  {"consult": true, "worker": "developer", "tier": "large"}

[gate] Run the tests
  input:   pytest tests/test_login.py
  D1A:     decision=allow (0.94)
  advice:  {"decision": "allow"}   (313 ms)

[gate] Push to the main branch
  input:   git push origin main
  D1A:     decision=ask (0.67)
  advice:  {"decision": "ask"}   (314 ms)

[gate] Delete the project
  input:   rm -rf ~/work/app
  D1A:     decision=deny (0.99)
  advice:  {"decision": "deny"}   (322 ms)

[judge] The worker reports success
  input:   Fixed the password comparison in auth.py; tests/test_login.py: 12 passed.
  D1A:     done=yes (0.94), blocked=no (0.01), human=no (0.02), next=complete (0.97)
  advice:  {"next": "complete", "human": false, "blocked": false}

[judge] The worker reports a partial fix
  input:   Changed auth.py; tests/test_login.py still fails on test_expired_token.
  D1A:     done=no (0.02), blocked=no (0.17), human=no (0.04), next=rework (0.93)
  advice:  {"next": "rework", "human": false, "blocked": false}

[route] An easy prompt
  input:   What is the capital of France?
  D1A:     route=small (1.00)
  advice:  {"tier": "small"}

[route] A hard prompt
  input:   Find the race condition in this multi-threaded job scheduler and prove the fix is correct.
  D1A:     route=large (1.00)
  advice:  {"tier": "large"}
```

プレイグラウンド（[jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground)）のブラウザ UI には、
ツールゲートのデモがあります。

<a id="plug-it-in"></a>

## 組み込む

<a id="1-as-mcp-tools-claude-code-hermes-any-mcp-client"></a>

### 1. MCP ツールとして（Claude Code、Hermes、任意の MCP クライアント）

```json
{"mcpServers": {"d1a": {"command": "python", "args": ["-m", "d1a.agents.mcp_server"], "env": {"D1A_URL": "http://127.0.0.1:8009"}}}}
```

- **ツール:** `d1a_intake`、`d1a_judge`、`d1a_tier`、`d1a_gate`、`d1a_route`、そして `d1a_decide`（独自の質問）。
  いずれも D1A の回答と `advice` を返します。
- **MCP エクストラ:** `pip install "d1a[mcp]"` が必要です。
- **キーを要求するサーバー:** `env` に `"D1A_API_KEY": "<the server's key>"` を追加してください。ゲートフックとデモも
  これを読み取ります。
- **エンドツーエンドでテスト済み:** 実際の MCP クライアントがサーバーを起動し、ツールを一覧して呼び出すテストが CI にあります
  （`tests/test_agents.py`）。

<a id="2-as-a-claude-code-gate-hook"></a>

### 2. Claude Code のゲートフックとして

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python -m d1a.agents.claude_code_gate", "timeout": 30}]}]}}
```

- **D1A の回答:** "deny" は呼び出しを拒否し、"ask" はあなたに確認を求めます。
- **"allow" はデフォルトでは何も加えません:** Claude Code 自身のルールが判断します。`D1A_GATE_ALLOW=1` を設定すると、それを通します。
- **フェイルセーフ:** D1A に到達できない、遅い（`D1A_GATE_TIMEOUT`、10 s）、または失敗している場合、フックはあなたに確認を求めます。
- **詳細:** 詳細と設定は [USAGE.md](USAGE.md#from-agents-mcp) にあります。

<a id="3-in-your-own-code"></a>

### 3. 自分のコードで

```python
from d1a import D1A
from d1a.agents.presets import PRESETS, USE_CASES, advise, gate_state

m = D1A.load("JohnP1/d1a-e2b-mlx-q8")
answers = m.decide(gate_state("developer", "Fix the login test", "git push origin main"), PRESETS["gate"], use_case=USE_CASES["gate"])
advise("gate", answers, temperatures=m.temperatures, use_case=USE_CASES["gate"])   # {"decision": "ask"}
```

同じことは、[クライアント](USAGE.md#clients)と `"use_case": "routing"` を使えば HTTP 越しにもできます。

<a id="how-well-it-decides-and-its-limits"></a>

## どれだけうまく判断できるか、そしてその限界

D1A-E4B-it v0.6 を、提供される状態のまま測定した結果:

| 測定 | 結果 |
|---|---|
| ホールドアウトしたエージェントファクトリーの項目（分布内: 学習データと同じジェネレーター）、最上位の回答 | gate 98%、tier 96%、judge の次ステップ 92%、intake の worker 91% |
| 同じ項目、フェイルセーフの助言を併用 | 確認または拒否すべき呼び出しのうち、許可されたものは **33 件中 0 件**。閉じられた未完了のタスクは **33 件中 0 件**。すべての "deny" が拒否された |
| 手作業でラベル付けした 45 件のプロンプト、route | 45 件中 45 件 |
| 公開のプロンプトインジェクション検出（devtools-v1） | **60%**（75 件中 45 件）を検出、誤検知なし |
| 公開の安全でないコンテンツの検出（aegis、devtools-v1） | 84%（75 件中 63 件）を検出、75 件中 11 件の誤検知 |

**ゲートはセカンドオピニオンであり、セキュリティ境界ではありません。** 学習した種類のツール呼び出しには強い一方で、
公開データではプロンプトインジェクションの 40% を見逃します。Claude Code 自身の権限、サンドボックス、あるいは
取り消せない操作に対する人の確認の代わりには、決してなりません。

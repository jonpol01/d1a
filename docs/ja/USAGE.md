<p><a href="../USAGE.md">English</a> · 日本語 · <a href="../ko/USAGE.md">한국어</a></p>

> この文書は英語版 [USAGE.md](../USAGE.md)（コミット 9eb86269 時点）の翻訳です。内容が異なる場合は英語版が正しい情報です。

<a id="usage"></a>

# 使い方

D1A の呼び出し方: System One API、プロセス内ライブラリ、エージェント向けの MCP ツール、クライアント、そして写真・音声・動画。

<a id="the-system-one-api"></a>

## System One API

`POST /v1/systemone` は `state`（文書）、`model` 名、そして多数の `questions` を受け取ります。各質問は `noul`（yes/no）、`choice`（名前付きの選択肢から 1 つ）、`score`（順序付きの尺度）のいずれかです。レスポンスには質問ごとに 1 つの回答が含まれ、選択肢ごとの確率に加えて、トークン使用量とレイテンシが付きます。`GET /v1/models` は受け付けるモデル名を、負荷を含む提供の詳細とともに一覧します。負荷には、処理したリクエストとバッチの数、キュー、プレフィックスキャッシュのヒット数、直近のバッチのモデル処理時間（p50、p95、最大値）が含まれます。`GET /metrics` は同じ内容を（メモリも加えて）スクレイパー向けに Prometheus のテキスト形式で報告します。`/v1` と同様に、`D1A_API_KEY` が設定されている場合はベアラーキーが必要です。

リクエストでユースケースを指定することもできます: `"use_case": "routing"`（省略可、最大 64 文字、前後の空白は無視されます。D1A 独自のフィールドで、他の System One サーバーは無視します）。チェックポイントがそのユースケース用の温度を持っている場合、リクエストの確率はその温度で読み取られます。`use_case` がない場合、またはチェックポイントに温度のない名前を指定した場合は、チェックポイント自身の温度が適用され、回答はこれまでと完全に同じです。最上位の回答は変わらず、変わるのは確信度だけです。`GET /v1/models` はチェックポイントの `use_case_temperatures`（ほとんどのチェックポイントでは `{}`）を一覧し、意思決定ログには各リクエストのユースケースと、それを読み取った温度が記録されます。エージェントのプリセット、MCP ツール、`D1A.decide(..., use_case=...)`、および両方のクライアントがこれを渡します。

**条件付き質問（`ask_if`、#36）。** 質問は、先行する回答が許す場合にのみ、同じリクエスト内で尋ねることができます。
`"ask_if": {"<question id>": <condition>}` で、すべての条件が必須です。

| 指定する質問 | 条件 |
|---|---|
| `choice` | その選択肢キーのリスト: 回答がそのいずれかである |
| `noul` | `true` または `false`: その回答（P(yes) ≥ 0.5）がそれと一致する |
| `score` | `{"min": .., "max": ..}`（どちらも省略可）: その回答の `score`（期待レベル、0 から数える）が範囲内である |

```json
{"state": "Order 4471 was charged twice.",
 "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?", "criteria": {"refund": null, "track": null}},
               "amount": {"type": "score", "instructions": "How much is at stake?", "criteria": ["small", "medium", "large"], "ask_if": {"intent": ["refund"]}},
               "carrier": {"type": "choice", "instructions": "Which carrier?", "criteria": {"ups": null, "dhl": null}, "ask_if": {"intent": ["track"]}}}}
```

- **パス:** サーバーはパスに分けて回答します。まず `ask_if` のない質問、その後、それが指す質問の後に各質問を処理します。
  state はキャッシュされるため、後続のパスのコストはその質問の行だけです。
- **スキップされた質問:** 条件が満たされない質問、またはスキップされた質問を指す質問は、リクエストの順序のまま
  `{"type": .., "skipped": true, "ask_if": {..}}` として返ります。
- **条件が読み取る回答:** モデル自身の回答です。学習済みフラグと同様です（結果による較正の前の値）。
- **1 つのリクエスト:** 使用量と `latency_ms` はパス全体で合算され、意思決定ログには 1 件のエントリが残ります。
- **検証:** 未知の質問または自分自身を指す条件、循環、質問の型に合わない条件は 422 になります。
- **モデル入力にはならない:** `ask_if` は D1A 独自のフィールドなので、これを含まないリクエストはこれまでと完全に同じように回答されます。
  `/v1/systemone/permute` と `/v1/systemone/separate` は、質問を単独で回答するため、これを拒否します。`D1A.decide` も
  同じ質問を受け付けます。

**ルーティング温度（v0.5 時代の数値）。** ルーティングは最初のユースケースです。v0.5 は T 1.78 で提供されており、そのルーティング回答は確信度が低すぎます。ホールドアウトしたファクトリーのルーティング行では、ルーティング温度 0.85 により ECE が 0.132 から 0.030 に下がり、確信度 0.9 以上の回答の割合が 0.28 から 0.66 に上がります。精度は同じです（#209）。v0.6 はルーティング温度なしで出荷されました。v0.6.1 はそれを備えており、D1A-E4B で T 1.18、D1A-E2B で 0.95 です（#264）。それ以外のすべてのリクエストは、チェックポイント自身の温度（1.95 と 1.45）を使い続けます。

モデル名: `GET /v1/models` が一覧するのは `d1a-latest` だけですが、サーバーはどのモデル名で送られたリクエストにも回答するため、別の名前向けに書かれたクライアント（たとえば未設定の TypeSafe SDK クライアント）もそのまま動作します。`D1A_API_KEY` を設定すると、`Authorization: Bearer <key>` が必須になります。

TypeSafe SDK を使う場合（`uv sync --extra serve` に含まれます）。英語以外のテキストも読めます:

```python
from typesafe_sdk import Choice, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="d1a-latest")
r = client.system_one(
    state="注文した靴が2週間遅れて届き、サイズも間違っていました。",
    questions={"team": Choice(instructions="Which team should handle this?",
                              criteria={"returns": None, "shipping": None, "billing": None})},
)
print(r.choices["team"].choice, r.choices["team"].probabilities)
```

<a id="in-your-own-code-no-server"></a>

## 自分のコードから使う（サーバーなし）

チェックポイントを一度ロードすれば、プロセス内で質問できます。質問と回答は System One API と同じ形式なので、コードは変更せずに両者の間を移せます:

```python
from d1a import D1A

m = D1A.load("JohnP1/d1a-e2b-mlx-q8")   # Apple Silicon (MLX, 3.8 GB download); JohnP1/d1a-e2b on NVIDIA or CPU
answers = m.decide("Shoes arrived late and I was charged twice.",
                   {"team": {"type": "choice", "instructions": "Which team should handle this?",
                             "criteria": {"returns": "returns", "shipping": "shipping", "billing": "billing"}},
                    "urgent": {"type": "noul", "instructions": "Is this urgent?"}})
print(answers["team"]["probabilities"], answers["urgent"]["noul"])
```

<a id="from-agents-mcp"></a>

## エージェントから使う（MCP）

`d1a.agents.mcp_server` は、エージェント（Hermes、Claude Code、任意の MCP クライアント）に、エージェントファクトリーの意思決定をツールとして提供します。`d1a_intake`（新しいリクエスト: どのワーカーか、どの規模のモデルか、先にユーザーに確認するか）、`d1a_judge`（カードとその最新のレポート: 完了、ブロック、人が必要、次のステップ）、`d1a_tier`（実装に使うのは小・中・大のどのモデルか）、`d1a_gate`（このツール呼び出しを実行してよいか: allow、ask、deny）、`d1a_route`（プロンプトにどの規模のモデルが答えるか）、`d1a_decide`（独自の質問）です。いずれも、確率付きの回答に加えて `advice` を返します。これはデフォルトがフェイルセーフなアクションで、確信のない回答は、上位に回す、人に確認する、カードを開いたままにする、のいずれかになります（`d1a.agents.presets.advise`）。

```bash
pip install "d1a[mcp] @ git+https://github.com/jonpol01/d1a"
D1A_URL=http://127.0.0.1:8009 python -m d1a.agents.mcp_server        # stdio; asks a running d1a.serving.serve
```

これを stdio MCP サーバー（コマンド `python`、引数 `-m d1a.agents.mcp_server`、環境変数 `D1A_URL`）としてエージェントの MCP 設定に登録し、上記のツール名を許可してください。`D1A_RUN=<checkpoint>` を指定すると、サーバーを呼び出す代わりに MCP プロセス内でモデルをロードします。質問セットは `d1a/agents/presets.py` にあり、ルーティングの学習データが尋ねるとおりの文言になっていて、`d1a_decide` 以外のすべてのツールは `"use_case": "routing"` を付けて送ります（`presets.USE_CASES`）。`advise` はチェックポイントの温度（`D1A.temperatures`、またはサーバーの `/v1/models` のエントリ。MCP ツールは呼び出しのたびにこれを読みます）と送信したユースケースを受け取り、`fail_up` の small のしきい値（チェックポイントの温度で 0.7）が、回答を読み取った温度に追従するようにします。比率を通して引き継いだ `sigmoid(logit(0.7) × T_checkpoint / T_use_case)` で、v0.5 のルーティング T 0.85 では 0.855 になります。gate、judge、および 0.5 のしきい値は変わりません。

**Claude Code ゲートフック。** `d1a.agents.claude_code_gate` は、すぐに使える `PreToolUse` フックです。Claude Code が
ツール呼び出しを実行する前に、D1A のゲートが判断します。

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python -m d1a.agents.claude_code_gate", "timeout": 30}]}]}}
```

- **D1A の回答:** "deny" は呼び出しを拒否し、"ask" はあなたに確認を求めます。いずれも D1A の確率を理由として添えます。
- **"allow" はデフォルトでは何も加えません:** フックがない場合と同じように、Claude Code 自身の権限ルールが判断します。
  `D1A_GATE_ALLOW=1` を設定すると、D1A の "allow" でプロンプトを省略できます。
- **フェイルセーフ:** D1A に到達できない、`D1A_GATE_TIMEOUT`（デフォルト 10 s）より遅い、またはそれ以外の何らかの失敗がある場合、フックはあなたに確認を求めます。
- **偽造されたフィールドなし:** コマンドとカードはそれぞれ改行をエスケープして 1 行で読み取られるため、コマンドがゲートの他のフィールドを偽造することはできません。
- **`D1A_URL` を使う:** `D1A_RUN` を使うと、フックは呼び出しのたびにモデルをロードします。
- **読み取るもの:** MCP ツールと同じ `D1A_URL`/`D1A_RUN`、および `D1A_GATE_CARD`（エージェントが取り組んでいるタスク。
  デフォルト: セッションのディレクトリ）。
- **他のツール:** `"Bash|Write|Edit"` のような `matcher` を使うと、それらもゲートされ、ツールの名前と入力として読み取られます。
- **これが何であり、何でないか:** ツール呼び出しに対するセカンドオピニオンであり、プロンプトインジェクション対策ではありません。
  Claude Code 自身の権限の代わりには決してなりません。D1A-E4B-it v0.6 での測定結果:
  - 公開のプロンプトインジェクション検出（devtools-v1）では、誤検知なしでインジェクションの **60%**（75 件中 45 件）を検出します。
  - ホールドアウトしたエージェントファクトリーのツール呼び出し（分布内: 学習データと同じジェネレーター）では、ゲートの最上位の
    回答は 98% が正解で、フェイルセーフの助言を併用すると、確認または拒否すべき呼び出し 33 件のうち許可されたものは 0 件でした。

<a id="clients"></a>

## クライアント

あらゆる System One サーバー向けの、依存関係のないクライアントが [`clients/python`](../../clients/python)（PyPI `d1a-client`、import は `d1a_client`）と [`clients/js`](../../clients/js)（npm `d1a-client`）にあります:

```python
from d1a_client import Client

client = Client("http://127.0.0.1:8009")
answer = client.decide("Shoes arrived late and I was charged twice.",
                       {"team": {"type": "choice", "instructions": "Which team should handle this?",
                                 "criteria": {"returns": None, "shipping": None, "billing": None}}})
print(answer["answers"]["team"]["probabilities"])
```

<a id="chat-with-d1a"></a>

## D1A でチャットする

**インストラクションチューニング済み**のトランクを使う D1A モデルは、チャットモデルでもあります。D1A アダプターをオフにすると、まさに
そのトランクになります（隠れ状態がビット単位で一致します）。`d1a.serving.serve` はこの性質を使って、System One の意思決定に答えるのと同じ
メモリ上のモデルから、OpenAI 形式のチャットおよび埋め込みリクエストに答えます。そのため、OpenAI API 向けに書かれたアプリ（mem0、n8n、
チャット UI）は、別のチャットサーバーの代わりに D1A を使えます（#274）。

- **チャットできるモデル:** 現時点では D1A-E2B-it と D1A-E4B-it。D1A-12B は学習され次第。ベーストランクのモデル（D1A-E2B、D1A-E4B）は
  できません。アダプターが重みにマージされたビルド（公開されている `-mlx-q8` / `-mlx-q4` リポジトリ）もできません。
- **PyTorch の場合**（NVIDIA、CPU、または MLX のない Mac）: LoRA の run そのものを `D1A_MERGE=0` で提供します。これによりアダプターが
  未マージのまま保たれ、オフに切り替えられます:
  `D1A_MERGE=0 python -m d1a.serving.serve --run JohnP1/d1a-e2b-it@v0.6 --port 8009`。チャットは PyTorch 上で生成されます。
  transformers 自身の生成と同じ回答で、MLX のプロンプトキャッシュはないため、長い会話では毎ターンでプロンプト全体を読み込みます。
  ツール呼び出しは MLX と同様にパースされます。
- **変わること:** チャットはアダプターをオフにして、サーバーのモデルスレッド上で、意思決定リクエストの合間に実行されます。チャット中に届いた意思決定は、
  チャットの 2 つのトークンの間（長いプロンプトを読み込み中であれば、その 2 つのチャンクの間）で回答されます。その間はアダプターをオンにし、終わったら再びオフに戻すため、
  待ち時間は回答全体ではなく約 1 トークン分で済み、チャットの回答も意思決定がない場合と同じです。

**1. 公開されているアダプターから、チャット可能なフォルダーを作成します**（Apple Silicon。-it のベースをダウンロードします。E2B で約 10 GB、
E4B で約 16 GB。E4B-it には `--run JohnP1/d1a-e4b-it@v0.6` を使ってください）:

```bash
uv run --extra serve python scripts/export_mlx.py --run JohnP1/d1a-e2b-it@v0.6 --q-bits 8 --q-group-size 64 \
    --unmerged --out runs/exports/d1a-e2b-it-chat
```

`--unmerged` はアダプターを組み込まずにベースの隣（`adapter/`）に残し、フォルダーにはベースのチャットテンプレートが含まれます。
28 リクエストのベンチでは、その意思決定は fp32 と一致します（反転 0/42）。

**2. 提供します**。アプリが埋め込みを必要とする場合（mem0 はそうです）は、埋め込みも併せて有効にします:

```bash
uv sync --extra serve --extra embed      # llama.cpp, for /v1/embeddings
D1A_EMBED_GGUF=<path to nomic-embed-text-v1.5.Q4_K_M.gguf> \
  uv run --extra serve python -m d1a.serving.serve --run runs/exports/d1a-e2b-it-chat --port 8009
```

GGUF は `nomic-ai/nomic-embed-text-v1.5-GGUF` のファイル `nomic-embed-text-v1.5.Q4_K_M.gguf` で、LM Studio が提供するのと同じファイルです。
LM Studio と同様に GPU 上で動作するため、そのベクトルは LM Studio が保存したものと一致し（提供ホストでコサイン類似度 1.00000）、既存のメモリとも比較可能なままです。

**3. OpenAI クライアントを D1A に向けます**:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8009/v1", api_key="unused")   # a real key when D1A_API_KEY is set
r = client.chat.completions.create(model="d1a-chat", messages=[{"role": "user", "content": "What is SQL injection?"}])
```

- **エンドポイント:** `POST /v1/chat/completions`（messages、`max_tokens`、`temperature`、`top_p`、`stop`、`tools` と
  `tool_choice`、`response_format`、`stream`）、`POST /v1/embeddings`（`input`: 文字列またはリスト。`encoding_format`: `float` または `base64`（n8n などの
  Node 版 OpenAI クライアントはデフォルトで後者を送ります））、そして `GET /v1/models`。後者は、意思決定モデルと並べて
  `d1a-chat` と埋め込みモデルを一覧し、アイドル状態のサーバーがモデルをアンロードしている間（`--idle-unload`）も同様です。
- **どのモデル ID にも回答します**ので、別のモデル用に設定されたアプリ（たとえば LM Studio の ID）は、URL を変えるだけで済みます。
- **ストリーミング**は回答をトークンごとに送ります。`tools` を指定した場合は、ツール呼び出しがパースされた時点で 1 つのチャンクとして届きます。
- **エージェントループ:** アシスタントのツール呼び出しは、OpenAI が返す形のまま（`arguments` は JSON 文字列）送り返し、続けて
  結果を持つ `tool` メッセージを送ります。各ステップは先行するプロンプトのキャッシュを再利用します。M1
  Max 上の D1A-E4B-it での測定: ツール呼び出し 3 回と要約を 4 ステップで、コールドで 4.1 s、その後は 1 ステップあたり 0.6–1.8 s。JSON オブジェクトではない引数は
  422 になります。
- **プロンプトキャッシュ:** プロンプトが先行するものの続きになっているチャット（エージェントの次のステップ、会話の次のターン、
  同じシステムプロンプトを持つ新しいセッション）は、新しいトークンだけをプリフィルします。`usage.prompt_tokens_details.cached_tokens` が、
  キャッシュから取った数を示します。最大 16 個のプロンプトが 1 GiB に保持され（`D1A_CHAT_CACHE_MB`、`0` でオフ）、
  アイドル時のアンロードではモデルとともに消えます。M1 Max 上の D1A-E4B-it での測定: 20k トークンのプロンプトでのエージェントの 1 ステップは 33 s ではなく 1–1.7 s、
  同じ 20k トークンのシステムプロンプトを持つ新しいセッションは 2.2 s です。
- **JSON モード**は、プロンプトでモデルに JSON を求めるものです。制約付きデコードはありません。
- **モデルのコンテキストは厳格な上限です:** プロンプトと `max_tokens`（未指定時は 1024）の合計がそれを超えるリクエスト
  （D1A-E4B-it では 131,072 トークン、モデルの設定の `max_position_embeddings`）は、ただちに 400（OpenAI の
  `context_length_exceeded`）になり、実行されません。そのエラーで圧縮するクライアント（Hermes はそうします）は、短くして再送します。
- **`max_tokens` で途中で切れたツール呼び出し**は破棄されます: 回答にはその前のテキストだけが残り、ツール呼び出しはなく、
  `finish_reason` は `"length"` になるので、より大きな `max_tokens` で再試行してください。
- **mem0:** OpenAI の LLM と埋め込みモデルのベース URL をサーバーに設定します。D1A-E2B-it での測定: 事実が抽出され、グラフの関係が
  保存され、検索でそれらが見つかります。追加には M1 Max で約 11 s かかります。質問に対するグラフ検索ではエンティティが見つかりません: 2B
  モデルは抽出ツールを呼び出す代わりに質問に答えてしまいます。これは LM Studio のときと同じです。

<a id="photos-voice-and-video"></a>

## 写真・音声・動画

モデルサーバーは、画像、短い音声クリップ、動画についての質問に、同じモデルで答えます。リクエストの形は同じで、`media` フィールドが加わります。Apple Silicon では、Gemma 4 のビジョンおよびオーディオのエンコーダーを含む MLX ビルドを提供してください（`media/`、約 1 GB、そのようなリクエストが初めて来たときに取得されます。`JohnP1/d1a-e4b-mlx-q8@v0.3` には含まれており、`scripts/export_mlx.py --media` で自分のエクスポートにも追加できます）:

```bash
uv run --extra serve --extra media python -m d1a.serving.serve --run JohnP1/d1a-e4b-mlx-q8@v0.6.1 --port 8009 --idle-unload 600
```

```bash
curl -s localhost:8009/v1/systemone/media -H 'content-type: application/json' -d '{
  "media": {"type": "image", "data": "'"$(base64 -i parcel.jpg)"'"},
  "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}}
}'
```

`--idle-unload 600` は、リクエストがないまま 10 分が過ぎるとエンコーダーを含むモデルを解放し、次のリクエストで再びロードします（M1 Max 上の D1A-E4B で 2 s）。`GET /v1/models` はどちらの状態でも応答し、`loaded` かどうかも示します。PyTorch のチェックポイント（NVIDIA、CPU）では、代わりに `python -m d1a.serving.media --run JohnP1/d1a-e2b --port 8010` を実行してください。これは Gemma 4 の完全なモデル（bf16、E2B で約 10 GB）を、独自のプロセスとして最初のリクエスト時にロードし、10 分アイドル状態が続くと解放します。

`type` は `image`（JPEG、PNG、WebP）、`audio`（WAV、FLAC、OGG、最大 30 s、サンプルレートは問いません）、`video`（MP4、MOV、WebM、最大 32 MB: 16 フレームを均等にサンプリングし、それぞれをタイムスタンプ付きの画像として読み取ります。音声トラックは使いません）のいずれかです。省略可能な `state` で、メディアに添えるテキストを加えられます。ゼロショットテストの結果は [DEMOS.md](DEMOS.md#photo-voice-and-video-results) にあります。

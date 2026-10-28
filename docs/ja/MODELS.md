<p><a href="../MODELS.md">English</a> · 日本語 · <a href="../ko/MODELS.md">한국어</a></p>

> この文書は英語版 [MODELS.md](../MODELS.md)（コミット 9eb86269 時点）の翻訳です。内容が異なる場合は英語版が正しい情報です。

<a id="models"></a>

# モデル

存在する D1A のビルド、公開場所、そして各バージョンの用途。

<a id="what-each-build-can-and-cannot-do-2026-10-25"></a>

## 各ビルドができること・できないこと（2026-10-25）

公開されているすべての D1A ビルドを、Hub 上のファイルで確認したものです。**意思決定**: System One API（`POST /v1/systemone`）。
**チャット**: OpenAI 形式の `POST /v1/chat/completions`。アダプターを別に保つこと（*未マージ*のビルド）が必要で、
アダプターをオフにすると、モデルはインストラクションチューニング済みの Gemma 4 そのものになります。**写真、音声、動画**: `POST /v1/systemone/media`。
すべてのビルドでゼロショットです（学習にメディアは含まれません）。

| ビルド | 形式 | 意思決定 | チャット | 写真 | 音声 | 動画 | 制限事項 |
|---|---|---|---|---|---|---|---|
| [d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) `v0.6.1` | PyTorch（アダプター） | ✅ | ❌ ベーストランク | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8) `v0.6.1` | MLX 8-bit、マージ済み | ✅ | ❌ ベーストランク | ✅ | ✅ | ✅ | 7.5 GB |
| [d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml) `v0.6` | Core ML int8 | ✅ | ❌ | ❌ | ❌ | ❌ | プレビュー: テキストのみ、入力は 512 トークン以下 |
| [d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b) `v0.6.1` | PyTorch（アダプター） | ✅ | ❌ ベーストランク | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) `v0.6.1` | MLX 8-bit、マージ済み | ✅ | ❌ ベーストランク | ✅ | ✅ | ✅ | 4.8 GB |
| [d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml) `v0.6` | Core ML int8 | ✅ | ❌ | ❌ | ❌ | ❌ | プレビュー: テキストのみ、入力は 512 トークン以下 |
| [d1a-e4b-it](https://huggingface.co/JohnP1/d1a-e4b-it) `v0.6` | PyTorch（アダプター） | ✅ | ✅² | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e4b-it-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q8) `v0.6` | MLX 8-bit、マージ済み | ✅ | ❌ マージ済み³ | ✅ | ✅ | ✅ | 7.5 GB |
| [d1a-e4b-it-mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q4) `v0.6` | MLX 4-bit、マージ済み | ✅ | ❌ マージ済み³ | ✅ | ✅ | ✅ | 5.2 GB |
| [d1a-e4b-it-coreml-4bit](https://huggingface.co/JohnP1/d1a-e4b-it-coreml-4bit) `v0.6` | Core ML 4-bit | ✅ | ❌ | ❌ | ❌ | ❌ | プレビュー: テキストのみ、512 トークン以下、テスト回答 42 件中 3 件が変化 |
| [d1a-e2b-it](https://huggingface.co/JohnP1/d1a-e2b-it) `v0.6` | PyTorch（アダプター） | ✅ | ✅² | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e2b-it-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-it-mlx-q8) `v0.6` | MLX 8-bit、マージ済み | ✅ | ❌ マージ済み³ | ✅ | ✅ | ✅ | 4.8 GB |
| [d1a-e2b-it-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-mlx-q4) `v0.6` | MLX 4-bit、マージ済み | ✅ | ❌ マージ済み³ | ✅ | ✅ | ✅ | 3.6 GB、スイートでは未スコアリング |
| [d1a-e2b-it-coreml](https://huggingface.co/JohnP1/d1a-e2b-it-coreml) `v0.6` | Core ML int8 | ✅ | ❌ | ❌ | ❌ | ❌ | プレビュー: テキストのみ、512 トークン以下 |
| [d1a-e2b-it-coreml-4bit](https://huggingface.co/JohnP1/d1a-e2b-it-coreml-4bit) `v0.6` | Core ML 4-bit | ✅ | ❌ | ❌ | ❌ | ❌ | プレビュー: テキストのみ、512 トークン以下 |
| [d1a-e2b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4) | MLX 4-bit（Google の QAT int4、厳密に保持）、**未マージ** | ✅ | ✅ | ✅ | ✅ | ✅ | 推奨の E2B-it ビルド。transfer-v4 が −5.2 ポイントであるにもかかわらず、2026-10-25 のオーナーの判断により D1A-E2B-it v0.6 に置き換わりました。3.7 GB |
| [d1a-e2b-it-qat](https://huggingface.co/JohnP1/d1a-e2b-it-qat) | PyTorch（アダプター。`base_ct`: Google の QAT int4 トランク、ロード時に再構築） | ✅ | ✅² | ✅¹ | ✅¹ | ✅¹ | PyTorch での推奨の E2B-it。Gemma 4 E2B-it と QAT チェックポイントをロードします |
| [d1a-e2b-it-qat-coreml](https://huggingface.co/JohnP1/d1a-e2b-it-qat-coreml) | Core ML（Google の QAT int4、厳密に保持。アダプターはスイッチ） | ✅⁴ | ✅⁴ | ✅⁴ | ✅⁴ | ✅⁴ | Core ML での推奨の E2B-it。4.6 GB。Mac（M1 Max、GPU）で検証済み、iPhone では未検証 |
| [d1a-e4b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-qat-mlx-q4) | MLX 4-bit（Google の QAT int4、厳密に保持）、**未マージ** | ✅ | ✅ | ✅ | ✅ | ✅ | QAT トランク上で再学習した D1A-E4B-it v0.6。v0.6 と全体では同等、typesafe −7.8（[スコアカード](../reports/2026-10-28-e4b-it-qat-scorecard.md)）。5.3 GB |
| [d1a-e4b-it-qat](https://huggingface.co/JohnP1/d1a-e4b-it-qat) | PyTorch（アダプター。`base_ct`: Google の QAT int4 トランク、ロード時に再構築） | ✅ | ✅² | ✅¹ | ✅¹ | ✅¹ | PyTorch での同じアダプター。Gemma 4 E4B-it と QAT チェックポイントをロードします |

¹ 別プロセスの PyTorch メディアサーバー `python -m d1a.serving.media --run <repo>` 経由です。完全な Gemma 4
モデル（bf16、E2B で約 10 GB）をロードします（[写真・音声・動画](USAGE.md)）。
² `D1A_MERGE=0` で提供します。これによりアダプターが未マージのまま保たれ、チャットでオフに切り替えられます（[D1A でチャットする](USAGE.md#chat-with-d1a)）。
³ チャットには未マージのビルドが必要です: `scripts/export_mlx.py --run JohnP1/d1a-e2b-it --unmerged --media` で作れます
（[D1A でチャットする](USAGE.md#chat-with-d1a)）。

⁴ リポジトリの Python リファレンスランナー（`reference/d1a_coreml_qat.py`、サーバーなし。`io.json` に移植用として
モデルのすべての入出力が一覧されています）経由です。すべてのモデルが GPU 上で動作し、Neural Engine 上のものはまだありません。意思決定は 1,536 トークンまで、チャットは
2,048 ポジションまで（[APPLE.md](APPLE.md#d1a-e2b-it-qat-on-core-ml)）。

D1A-E2B-it QAT は、チャット、写真、音声、動画に対応する唯一の Core ML ビルドです。v0.6 の Core ML ビルドはテキストのみです。

<a id="capabilities-2026-10-17"></a>

## 機能（2026-10-17）

現時点で各 D1A モデルラインができること。「意思決定」とは D1A の System One API（`POST /v1/systemone`）を指します。文書に対する yes/no、choice、score の質問に、較正済み確率付きで 1 回のパスで答えるものです。数値は各モデルのカードによるもので、「未測定」はカードにまだ数値がないことを意味します。

| | **D1A-E4B-it v0.6** | **D1A-E4B v0.6.1** | **D1A-E2B v0.6.1** | **D1A-E2B-it v0.6** | D1A-12B |
|---|---|---|---|---|---|
| ベースモデル（トランク） | Gemma 4 E4B-it（インストラクションチューニング済み） | Gemma 4 E4B（事前学習済みベース） | Gemma 4 E2B（事前学習済みベース） | Gemma 4 E2B-it（インストラクションチューニング済み） | Gemma 4 12B-it QAT。**まだ学習されていません**（jonpol01/d1a#270、保留中） |
| 意思決定（`/v1/systemone`） | はい | はい | はい | はい（実験的） | — |
| PR ラベル付け、学習データのいずれよりも新しい 487 件の PR: 重大度 / 変更タイプ | このセットでは未測定。PR-labels テスト（953 件の PR）: 72.9% / 88.4%（E2B-it は 70.5% / 87.8%） | **79.1% / 90.6%**。P0/P1 は 30 件中 13 件を検出 | 77.2% / 87.5%。P0/P1 は 30 件中 0 件 | 72.9% / 87.3%。P0/P1 は 30 件中 3 件（「PR ラベラーではない」） | — |
| スキルスイート（難しい意思決定、開発者ツール、長い文書、転移、日本語 JGLUE） | 2026-10-17 に E2B-it と比較して測定（カード）: 22 スイート中 21 で向上。たとえば難しい意思決定 75.7%、開発者ツール 74.3%、長い文書 88.6%、JGLUE 78.6% | 測定済み（カード） | 測定済み（カード） | 2026-10-15 に E4B v0.6.1 と比較して測定（未公開） | — |
| 写真、音声、動画（`/v1/systemone/media`、`media/` を含む MLX ビルド） | はい、ゼロショット。28 件のデモ回答のうち 7 件が E2B-it と異なる（目視では 6 件が改善） | はい、ゼロショット | はい。28 件のデモ回答のうち 25 件が E4B と一致 | はい。28 件のうち 21 件が E4B と一致 | — |
| ルーティング温度（`"use_case": "routing"`） | いいえ（フィット済みだが未同梱: 一般の T より良くならなかったため） | はい、T 1.18（v0.6.1 以降） | はい、T 0.95（v0.6.1 以降） | いいえ | — |
| 自己学習（結果ログ、再較正、学習済みフラグ） | はい | はい（`d1a.serving.serve` で提供する場合） | はい | はい | — |
| **チャット / テキスト生成** | **はい**: 未マージのビルド（`scripts/export_mlx.py --unmerged`）。アダプターをオフにすると Gemma 4 E4B-it になります（[D1A でチャットする](USAGE.md#chat-with-d1a)） | **いいえ**: ベーストランク。MLX ビルドはアダプターがマージ済み | **いいえ**（同上） | **はい**: `scripts/export_mlx.py --unmerged` でビルドして提供します。アダプターをオフにすると Gemma 4 E2B-it になります（[D1A でチャットする](USAGE.md#chat-with-d1a)）。Mieru iPhone アプリでも、未マージのアダプターとして動作します | 計画中: アダプターオフ = Gemma 4 12B-it |
| 埋め込み | 併設して提供する GGUF 埋め込みモデルによる（`D1A_EMBED_GGUF`） | — | — | 併設して提供する GGUF 埋め込みモデルによる（`D1A_EMBED_GGUF`） | — |
| ビルド | PyTorch アダプター `JohnP1/d1a-e4b-it`、MLX `JohnP1/d1a-e4b-it-mlx-q8`（メディア込みで 7.5 GB）と `-mlx-q4`（5.2 GB）はいずれもマージ済み、Core ML 4-bit `JohnP1/d1a-e4b-it-coreml-4bit`（テキストのみ、512 トークン以下。テスト回答 42 件中 3 件が変化、int8 は未ビルド） | PyTorch `JohnP1/d1a-e4b`、MLX 8-bit `JohnP1/d1a-e4b-mlx-q8`（メディアエンコーダー 1.0 GB を含めて 7.5 GB）とミラー `mlx-community/d1a-e4b-8bit`、Core ML int8 プレビュー `JohnP1/d1a-e4b-coreml`（テキストのみ、512 トークン以下） | PyTorch `JohnP1/d1a-e2b`、MLX 8-bit `JohnP1/d1a-e2b-mlx-q8`（4.8 GB）とミラー `mlx-community/d1a-e2b-8bit`、Core ML int8 `JohnP1/d1a-e2b-coreml`（テキストのみ、512 トークン以下） | PyTorch アダプター `JohnP1/d1a-e2b-it`、MLX `JohnP1/d1a-e2b-it-mlx-q8`（4.8 GB）と `-mlx-q4`（3.6 GB）はいずれもマージ済み、Core ML `JohnP1/d1a-e2b-it-coreml` と `-coreml-4bit` | 4-bit Q4_0 エクスポートは準備済み（`scripts/export_mlx.py --q4-0`、#272） |
| 較正温度 | 1.55 | 1.95 | 1.45 | 1.41 | — |

PR ラベル付けとルーティング向けの旧来の単一目的ビルドは、D1A-E4B v0.6.1 に置き換えられました。

<a id="builds"></a>

## ビルド

| | |
|---|---|
| Gemma 4 E2B / E4B（および Qwen）での学習と提供 | はい、このリポジトリ |
| ベースとしての Gemma 4 12B（統合モデル。例: `google/gemma-4-12B-it-qat-q4_0-unquantized`） | 両方のバックエンドがロードできます（#270: PyTorch は transformers 経由、MLX は mlx-lm の Gemma 4 クラス経由）。D1A-12B はまだ学習されていません |
| D1A-E2B v0.6（Gemma 4 E2B: それ以降の D1A のスキルをすべて 1 回の run に集約、オンデバイス向け） | [JohnP1/d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b) `v0.6`: v0.2 と比べて、難しい意思決定 31% → 62%、開発者ツール 51% → 65%、長い文書 73% → 86%、PR の変更タイプ 40% → 85%。学習データのいずれよりも新しい 487 件の PR では、重大度 77%、変更タイプ 88%（E4B v0.6: 79%、91%）ですが、そこで P0 や P1 を予測することは一度もないため、PR ラベラーは引き続き E4B です。明示的な例外としてリリースされました: 汎用の意思決定（decision-v7）が、基準の −1 に対して −1.4（#229）。`v0.6.1` は、同じ重みにルーティング温度（T 0.95）を加えたものです。`v0.2`（汎用の意思決定、2 エポック、較正済み）と `v0.1` も引き続き利用できます |
| Apple Silicon 向け D1A-E2B（MLX、8-bit） | [JohnP1/d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) `v0.6` と `v0.6.1`（3.8 GB、これに写真・音声・動画のエンコーダー 1.0 GB） |
| D1A-E4B v0.6（Gemma 4 E4B: v0.5 を、それ以前のすべての学習データよりも新しいプルリクエストで更新し、長い文書とルーティングを加えたもの） | [JohnP1/d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) `v0.6`: 学習データのいずれよりも新しい 487 件の PR で、重大度 79%（v0.5: 75%）、変更タイプ 91%（87%）。リリース基準に対する明示的な例外としてリリースされました: v0.5 と比べて、難しい意思決定で 2.7 ポイント、旧 PR テストセットの変更タイプで 4.1 ポイント低下します（#197、#198）。それらには `v0.5` を固定して使ってください。`v0.6.1` は、同じ重みにルーティング温度（T 1.18）を加えたものです。`v0.4`、`v0.3`、`v0.2`（Kev の後段ステージ + 日本語 + エージェントのルーティング）、`v0.1` も引き続き利用できます |
| D1A-E4B-it v0.6（インストラクションチューニング済みの Gemma 4 E4B-it 上での E2B v0.6 のレシピ: 1 つのモデルで意思決定とチャット） | [JohnP1/d1a-e4b-it](https://huggingface.co/JohnP1/d1a-e4b-it) `v0.6`: D1A-E2B-it v0.6 と比べて、すべてのスイートで 22 中 21 の精度が向上（プールで +2.5 ポイント。難しい意思決定 +6.6、開発者ツール +3.2、長い文書 +2.3）、1 回の意思決定あたり約 2.9× 遅くなります。アダプターをオフにすると Gemma 4 E4B-it になります（[D1A でチャットする](USAGE.md#chat-with-d1a)）。MLX ビルド、アダプターはマージ済み（意思決定のみ）: [JohnP1/d1a-e4b-it-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q8) と [-mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q4) |
| Apple Neural Engine 向け D1A-E4B-it（Core ML、4-bit） | [JohnP1/d1a-e4b-it-coreml-4bit](https://huggingface.co/JohnP1/d1a-e4b-it-coreml-4bit) `v0.6`、プレビュー: テキストのみ、入力は 512 トークンまで。28 件のリクエストで fp32 リファレンスと比べると、42 件の回答のうち 3 件が変化し（最大 \|Δp\| 0.46）、M1 Max の GPU で p50 6.6 s です。回答が変わってはならない場面では MLX 8-bit ビルドを使ってください。int8 ビルドはありません |
| Apple Silicon 向け D1A-E2B-it QAT（MLX、4-bit） | [JohnP1/d1a-e2b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4): Google の QAT int4 トランク（`google/gemma-4-E2B-it-qat-w4a16-ct`）上で D1A-E2B-it v0.6 を再学習し、厳密に保持したもの（2.9 GB、これに QAT チェックポイントの写真・音声・動画エンコーダー 0.9 GB）。意思決定、チャット、写真、音声、動画に対応します。同じトランク上の v0.6 アダプターと比べてプールで +2.1 ポイント、bf16 上の D1A-E2B-it v0.6 と比べてプールで +0.4、PR の重大度 +3.5、PR ラベル +1.7、ただし transfer-v4 は −5.2 です。2026-10-25 のオーナーの判断により、推奨の E2B-it ビルドとして v0.6 に置き換わります。D1A-E4B v0.6 と同様のリリース例外です（#341）。全スイート: [スコアカード](../reports/2026-10-25-e2b-it-qat-scorecard.md) |
| Apple Silicon 向け D1A-E4B-it QAT（MLX、4-bit） | [JohnP1/d1a-e4b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-qat-mlx-q4): Google の QAT int4 トランク（`google/gemma-4-E4B-it-qat-w4a16-ct`）上で D1A-E4B-it v0.6 を再学習し、厳密に保持したもの（4.4 GB、これに QAT チェックポイントの写真・音声・動画エンコーダー 0.9 GB）。意思決定、チャット、写真、音声、動画に対応します。v0.6 と異なり、Gemma 4 のレイヤーごとの埋め込みを正しく読んで学習しています（#353）。D1A-E4B-it v0.6 と比べてプールで同等（−0.2）、PR ラベル +1.6、PR の重大度 +6.0、ルーティング +1.7 と +1.5、ただし外部の typesafe は −7.8 です（#361）。同じトランク上の v0.6 アダプターと比べてプールで +1.2、有意な低下はありません。2026-10-27 のオーナーの判断によりリリースされました（#341）。全スイート: [スコアカード](../reports/2026-10-28-e4b-it-qat-scorecard.md)。PyTorch: [JohnP1/d1a-e4b-it-qat](https://huggingface.co/JohnP1/d1a-e4b-it-qat) |
| Apple Silicon 向け D1A-E4B（MLX、8-bit） | [JohnP1/d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8) `v0.6` と `v0.6.1`（6.6 GB、これに写真・音声・動画のエンコーダー 1.0 GB） |
| Apple Neural Engine 向け D1A-E4B（Core ML、int8） | [JohnP1/d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml) `v0.6`、プレビュー: テキストのみ、入力は 512 トークンまで（[Apple Neural Engine（Core ML）](APPLE.md#apple-neural-engine-core-ml)） |
| Apple Neural Engine 向け D1A-E2B（Core ML、int8） | [JohnP1/d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml) `v0.6`: テキストのみ、入力は 512 トークンまで（[Apple Neural Engine（Core ML）](APPLE.md#apple-neural-engine-core-ml)） |
| 軽量クライアント（Python、JS） | [`clients/`](../../clients) |

モデルのバージョン: サイズと形式ごとに Hugging Face リポジトリが 1 つ、バージョンごとにタグが 1 つあります（`v0.1`、`v0.2`、`v0.3`、`v0.4`、`v0.5`、`v0.6`、`v0.6.1`、...）。各バージョンは 1 つ前のバージョンの学習を引き継ぎ、そのモデルカードに学習内容が一覧されています。バージョンは `JohnP1/d1a-e4b@v0.3` のようにロードします。旧来のタグ名（`v0.2-hybrid`、`v0.2-2epoch`、...）も引き続き使えます。

<p align="center"><a href="README.md">English</a> · 日本語 · <a href="README.ko.md">한국어</a></p>

> この文書は英語版 [README.md](README.md)（コミット 9eb86269 時点）の翻訳です。内容が異なる場合は英語版が正しい情報です。

<p align="center">
  <img src="docs/d1a-avatar.png" width="220" alt="D1A: 銀白色と濃い紫色の髪、丸眼鏡、ヘッドホンを身につけたアニメ調の女の子が描かれた、「#D1A」と書かれた丸いエナメルバッジ">
</p>

<h1 align="center">D1A</h1>

<p align="center"><b>Gemma 4 をベースにした小型の意思決定モデル。</b> 1 つの文書と型付きの質問群を入力すると、1 回のフォワードパスで、すべての選択肢の較正済み確率が出力されます。テキスト生成は行いません。</p>

<p align="center">
  <a href="https://github.com/jonpol01/d1a/actions/workflows/ci.yml"><img src="https://github.com/jonpol01/d1a/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://github.com/jonpol01/d1a/actions/workflows/license.yml"><img src="https://github.com/jonpol01/d1a/actions/workflows/license.yml/badge.svg?branch=main" alt="ライセンスチェック"></a>
  <a href="https://github.com/jonpol01/d1a/releases"><img src="https://img.shields.io/github/v/release/jonpol01/d1a" alt="リリース"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="ライセンス"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python"></a>
</p>
<p align="center">
  <a href="https://huggingface.co/google/gemma-4-E4B"><img src="https://img.shields.io/badge/base-Gemma%204%20E2B%20%7C%20E4B-4285F4" alt="ベースモデル: Gemma 4 E2B と E4B"></a>
  <a href="https://huggingface.co/JohnP1/d1a-e4b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20model-JohnP1%2Fd1a--e4b-yellow" alt="モデル"></a>
  <a href="https://huggingface.co/collections/JohnP1/d1a-for-pytorch-cuda-cpu-non-mac"><img src="https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU-ee4c2c" alt="PyTorch コレクション"></a>
  <a href="https://huggingface.co/collections/JohnP1/d1a-for-mlx-apple-silicon-gpu"><img src="https://img.shields.io/badge/Apple%20Silicon-MLX-black" alt="MLX コレクション"></a>
  <a href="https://huggingface.co/collections/JohnP1/d1a-for-core-ml-apple-neural-engine"><img src="https://img.shields.io/badge/Neural%20Engine-Core%20ML%20preview-555555" alt="Core ML コレクション"></a>
  <a href="https://huggingface.co/mlx-community/d1a-e4b-8bit"><img src="https://img.shields.io/badge/mlx--community-8bit-black" alt="mlx-community ミラー"></a>
  <a href="https://github.com/jonpol01/d1a-playground"><img src="https://img.shields.io/badge/demos-d1a--playground-7c3aed" alt="デモ"></a>
</p>

> **Built on Kev.** D1A は Jared Palmer による [Kev](https://github.com/jaredpalmer/kev) をベースに構築されており、[Apache License 2.0](LICENSE) でライセンスされています。ここにあるモデルコード、トレーナー、ベンチマーク、凍結済みの評価スイートは、Kev のコピー（アップストリームのコミット `0fe8fc9`）から始まりました。Kev から取り込んだすべてのファイルと D1A が加えた変更は [docs/UPSTREAM.md](docs/UPSTREAM.md) に一覧があります。D1A は独立したプロジェクトです。Jared Palmer および Kev の作者とは提携しておらず、後援・推奨を受けていません。

<a id="proven-at"></a>

## 実績

- **エージェントの意思決定**（D1A-E4B-it v0.6、ホールドアウトしたエージェント項目）: ツール呼び出しゲート 98%、タスク判定 92%、
  モデルルーティングは手作業でラベル付けしたプロンプト 45 件中 45 件。フェイルセーフの助言を併用すると、確認を求めるべき、
  または拒否すべき呼び出し 33 件のうち、許可されたものは 0 件でした
  （[docs/ja/AGENTS_GUIDE.md](docs/ja/AGENTS_GUIDE.md)）。
- **難しい意思決定、開発者ツール、長い文書、日本語**（D1A-E4B-it v0.6、D1A のスキルスイート）: 75.7%、74.3%、
  88.6%、78.6%（[docs/ja/MODELS.md](docs/ja/MODELS.md)）。
- **意思決定とチャットを 1 つのモデルで:** アダプターをオフにすると、D1A-E4B-it は OpenAI 形式のエンドポイントの背後で動く
  Gemma 4 E4B-it になり、プロンプトキャッシュによって、20k トークンのプロンプトでのエージェントの 1 ステップが 33 s から 1–1.7 s に短縮されます（M1 Max）。
- **日常運用での PR ラベル付け**（D1A-E4B v0.6）: M4 で 1 PR あたり 0.63 s。学習データのいずれよりも新しい 487 件の PR で、
  重大度 79%、変更タイプ 91%（[recipes/pr-labeler](recipes/pr-labeler/README.md)）。
- **写真、音声、動画**を、Gemma 4 自身のエンコーダーでゼロショット処理します（[docs/ja/DEMOS.md](docs/ja/DEMOS.md)）。

<a id="quick-start"></a>

## クイックスタート

Python 3.12 または 3.13 と [uv](https://docs.astral.sh/uv/) が必要です。

```bash
git clone https://github.com/jonpol01/d1a.git && cd d1a
uv sync --extra serve
uv run --extra serve python -m d1a.serving.serve --run JohnP1/d1a-e4b-mlx-q8@v0.6.1 --port 8009
```

これで Apple Silicon 上で MLX ビルドが提供されます。それ以外の環境では PyTorch ビルド `--run JohnP1/d1a-e4b@v0.6.1` を指定してください（GPU があれば CUDA を使います。`D1A_BACKEND=torch` を指定すると MPS 上で PyTorch が動作します。[Apple Silicon（MLX）](docs/ja/APPLE.md#apple-silicon-mlx)を参照）。初回の実行ではモデルをダウンロードします。`--run` にはローカルのチェックポイントディレクトリや Hub のリビジョン（`repo@rev`）も指定できます。

すべての設定（サーバーのフラグ、`D1A_*` 変数、自己学習、写真と音声、エージェント）について、デフォルト値と変更すべき場面を
[docs/CONFIGURATION.md](docs/CONFIGURATION.md) に一覧しています。

チケットを送ってみましょう。

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "model": "d1a-latest",
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"returns": null, "shipping": null, "billing": null}},
    "urgent": {"type": "noul", "instructions": "Does this need a reply today?"}
  }
}'
```

<a id="pick-a-build"></a>

## ビルドを選ぶ

公開されているすべてのビルドができること、できないこと。制限事項と脚注: [docs/ja/MODELS.md](docs/ja/MODELS.md#what-each-build-can-and-cannot-do-2026-10-25)。

<a id="d1a-e2b-smaller-and-faster-phones-and-small-machines"></a>

### D1A-E2B（より小さく高速: スマートフォンや小型マシン向け）

| ビルド | 形式 | 意思決定 | チャット | 写真 | 音声 | 動画 | サイズ、備考 |
|---|---|---|---|---|---|---|---|
| **D1A-E2B-it QAT**（[mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4)） | MLX 4-bit、未マージ | ✅ | ✅ | ✅ | ✅ | ✅ | 3.7 GB、**E2B-it の推奨ビルド**、v0.7 |
| **D1A-E2B-it QAT**（[d1a-e2b-it-qat](https://huggingface.co/JohnP1/d1a-e2b-it-qat)） | PyTorch | ✅ | ✅⁴ | ✅¹ | ✅¹ | ✅¹ | Gemma 4 E2B-it と Google の QAT チェックポイントに対する 126 MB のアダプター、v0.6 |
| **D1A-E2B-it QAT**（[coreml](https://huggingface.co/JohnP1/d1a-e2b-it-qat-coreml)） | Core ML（Google の QAT int4、厳密に保持） | ✅⁵ | ✅⁵ | ✅⁵ | ✅⁵ | ✅⁵ | 4.6 GB、Mac で検証済み、iPhone では未検証、v0.6 |
| **D1A-E2B-it QAT**（[onnx](https://huggingface.co/JohnP1/d1a-e2b-it-qat-onnx)） | ONNX Runtime（Android / CPU）、int4 | ✅ | ✅ | ❌ | ❌ | ❌ | 2.9 GB、v0.7、意思決定とチャットを1つのグラフで、テキストのみ |
| D1A-E2B-it v0.6（[d1a-e2b-it](https://huggingface.co/JohnP1/d1a-e2b-it)） | PyTorch | ✅ | ✅⁴ | ✅¹ | ✅¹ | ✅¹ | |
| D1A-E2B-it v0.6（[mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-it-mlx-q8)、[mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-mlx-q4)） | MLX、マージ済み | ✅ | ❌³ | ✅ | ✅ | ✅ | 4.8 / 3.6 GB |
| D1A-E2B-it v0.6（[coreml](https://huggingface.co/JohnP1/d1a-e2b-it-coreml)、[coreml-4bit](https://huggingface.co/JohnP1/d1a-e2b-it-coreml-4bit)） | Core ML | ✅² | ❌ | ❌ | ❌ | ❌ | 3.8 / 2.9 GB、プレビュー |
| D1A-E2B（[d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b)） | PyTorch | ✅ | ❌ | ✅¹ | ✅¹ | ✅¹ | |
| D1A-E2B（[mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8)） | MLX 8-bit | ✅ | ❌ | ✅ | ✅ | ✅ | 4.8 GB |
| D1A-E2B（[coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml)） | Core ML int8 | ✅² | ❌ | ❌ | ❌ | ❌ | 3.8 GB、プレビュー |

QAT ビルドは、見慣れないタスク種別（transfer-v4）での 5.2 ポイントと引き換えに、D1A 固有のジョブ、速度、そして 5 つの機能すべてを得ています
（[スコアカード](docs/reports/2026-10-25-e2b-it-qat-scorecard.md)）。
その v0.7（2026-10-30、今のところ MLX のみ）は、英語と日本語の対話行為の判断が向上しました。ユーザーが同意したか、お礼だけか依頼か、
操作か雑談か（保留した問題で +15.9 ポイント）。インテント分類とその他のスイートは変わりません（[モデルカード](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4)）。

<a id="d1a-e4b-more-accurate-servers-and-macs"></a>

### D1A-E4B（より高精度: サーバーと Mac 向け）

| ビルド | 形式 | 意思決定 | チャット | 写真 | 音声 | 動画 | サイズ、備考 |
|---|---|---|---|---|---|---|---|
| **D1A-E4B-it QAT**（[mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-qat-mlx-q4)） | MLX 4-bit、未マージ | ✅ | ✅ | ✅ | ✅ | ✅ | 5.3 GB、v0.6 と同等、typesafe −7.8 |
| **D1A-E4B-it QAT**（[d1a-e4b-it-qat](https://huggingface.co/JohnP1/d1a-e4b-it-qat)） | PyTorch | ✅ | ✅⁴ | ✅¹ | ✅¹ | ✅¹ | Gemma 4 E4B-it と Google の QAT チェックポイントに対する 133 MB のアダプター |
| D1A-E4B-it（[d1a-e4b-it](https://huggingface.co/JohnP1/d1a-e4b-it)） | PyTorch | ✅ | ✅⁴ | ✅¹ | ✅¹ | ✅¹ | |
| D1A-E4B-it（[mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q8)、[mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q4)） | MLX、マージ済み | ✅ | ❌³ | ✅ | ✅ | ✅ | 7.5 / 5.2 GB |
| D1A-E4B-it（[coreml-4bit](https://huggingface.co/JohnP1/d1a-e4b-it-coreml-4bit)） | Core ML 4-bit | ✅² | ❌ | ❌ | ❌ | ❌ | 4.6 GB、プレビュー |
| D1A-E4B（[d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b)） | PyTorch | ✅ | ❌ | ✅¹ | ✅¹ | ✅¹ | サーバーのデフォルト |
| D1A-E4B（[mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8)） | MLX 8-bit | ✅ | ❌ | ✅ | ✅ | ✅ | 7.5 GB |
| D1A-E4B（[coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml)） | Core ML int8 | ✅² | ❌ | ❌ | ❌ | ❌ | 6.5 GB、プレビュー |

D1A-E4B-it QAT は、Google の QAT int4 トランク上で D1A-E4B-it v0.6 を再学習したものです。v0.6 と比べて全体では同等（プールで −0.2 ポイント [−0.9, +0.5]）で、PR ラベル +1.6、PR の重大度 +6.0、ルーティング +1.7 と +1.5 ですが、外部の typesafe スイートは −7.8 です。同じトランクに移した v0.6 のアダプターと比べると、プールで +1.2、有意な低下はありません（[スコアカード](docs/reports/2026-10-28-e4b-it-qat-scorecard.md)）。

¹ 別プロセスの PyTorch メディアサーバー（`python -m d1a.serving.media`）経由です。完全なモデル（E2B で約 10 GB）を読み込みます。
² テキストのみ、入力は 512 トークンまで。
³ 公開されているのはマージ済みのビルドです。チャットには `--unmerged` を付けてエクスポートしてください（[D1A でチャットする](docs/ja/USAGE.md#chat-with-d1a)）。
⁴ `D1A_MERGE=0` で提供します。これによりアダプターが未マージのまま保たれ、チャットでアダプターをオフに切り替えられます。
⁵ リポジトリの Python リファレンスランナー（`reference/d1a_coreml_qat.py`、サーバーなし）経由です。Neural Engine ではなく GPU 上で動作し、意思決定は 1,536 トークンまで。

D1A-E2B-it QAT は、チャット、写真、音声、動画に対応する唯一の Core ML ビルドです。それ以外の Core ML ビルドはテキストのみです。

E4B はより高精度なサイズで、E2B はより小さく高速なため、スマートフォンや小型マシン向けです。`-it` モデルは、インストラクションチューニング済みの Gemma 4 トランクに対して同じ方法で学習されているため、1 つのモデルでチャットもできます。D1A アダプターをオフにすると、Gemma 4 E4B-it / E2B-it になります。チャットには未マージのビルドが必要です（`scripts/export_mlx.py --unmerged`、[D1A でチャットする](docs/ja/USAGE.md#chat-with-d1a)）。公開されている `-mlx-q8` / `-mlx-q4` リポジトリはアダプターがマージ済みで、意思決定にのみ答えます。D1A-E4B-it は D1A-E2B-it より 1 回の意思決定あたり約 2.9× 遅く、22 スイート中 21 でより高精度です。D1A-E4B と D1A-E2B の PyTorch ビルドと MLX ビルドは `v0.6.1`（v0.6 にルーティング温度を加えたもの）で、`-it` と Core ML のビルドは `v0.6` です。Core ML は iPhone/iPad 向けのプレビューで、Mac では MLX の方が高速です（[docs/ja/APPLE.md](docs/ja/APPLE.md)）。v0.6 の Core ML ビルドはテキストのみで入力は 512 トークンまで、D1A-E2B-it QAT のビルドは 5 つの機能すべてに対応し、Mac で検証済みですが iPhone ではまだです。4-bit の E4B-it ビルドは 42 件のテスト回答のうち 3 件が変化するため、回答が変わってはならない場面では MLX 8-bit を使ってください。サイズ、例外、旧バージョン: [docs/ja/MODELS.md](docs/ja/MODELS.md)。

<a id="use-it"></a>

## 使ってみる

`POST /v1/systemone` は `state`（文書）と `questions` を受け取ります。各質問は `noul`（yes/no）、`choice`、`score` のいずれかです。レスポンスには、すべての質問のすべての選択肢の確率が含まれます。同じ形式は、サーバーなしの自分のコードでもそのまま使えます。

```python
from d1a import D1A

m = D1A.load("JohnP1/d1a-e2b-mlx-q8")   # Apple Silicon (MLX, 3.8 GB download); JohnP1/d1a-e2b on NVIDIA or CPU
answers = m.decide("Shoes arrived late and I was charged twice.",
                   {"team": {"type": "choice", "instructions": "Which team should handle this?",
                             "criteria": {"returns": "returns", "shipping": "shipping", "billing": "billing"}},
                    "urgent": {"type": "noul", "instructions": "Is this urgent?"}})
print(answers["team"]["probabilities"], answers["urgent"]["noul"])
```

- Python と JS のクライアント: [docs/ja/USAGE.md](docs/ja/USAGE.md#clients)
- エージェント（MCP ツール、Claude Code のゲートフック、デモ）: [docs/ja/AGENTS_GUIDE.md](docs/ja/AGENTS_GUIDE.md)
- 写真、音声、動画: [docs/ja/USAGE.md](docs/ja/USAGE.md#photos-voice-and-video)

<a id="for-agents"></a>

## エージェント向け

D1A は、エージェントの小さな意思決定を高速かつフェイルセーフにします。誰がリクエストを担当するか、どの規模のモデルが必要か、
ツール呼び出しを実行してよいか（allow、ask、deny）、タスクが完了したか、といった判断です。
- **3 つの導入方法:** あらゆるエージェント向けの MCP ツール、Claude Code のゲートフック、または Python のプリセット。
- **試してみる:** `python -m d1a.agents.demo` が、1 つのタスクを通してエージェントの 1 日を見せます。
- **測定結果:** ホールドアウトしたエージェント項目で、フェイルセーフの助言を併用すると、確認を求めるべき、または
  拒否すべき呼び出し 33 件のうち許可されたものは 0 件、未完了のタスク 33 件のうち閉じられたものも 0 件でした。
- **限界:** 公開されているプロンプトインジェクションの 60% を検出するため、セキュリティ境界ではなくセカンドオピニオンです。

ガイド、デモの出力、数値: [docs/ja/AGENTS_GUIDE.md](docs/ja/AGENTS_GUIDE.md)。

<a id="how-it-works"></a>

## 仕組み

1. **入力。** 文書は 1 回だけ入力され、各質問は、その選択肢と `<decide>` トークンを持つ独立した枝として続きます。
2. **モデル。** LoRA アダプターを備えた Gemma 4 バックボーンと小さなポインターヘッドが各選択肢をスコアリングし、較正済みソフトマックスがスコアを確率に変換します。
3. **2 つの実行方法。** パック方式（ブロック因果マスクの下で 1 つのシーケンスとして処理）または行方式（文書をキャッシュし、質問ごとに短い 1 行を処理）で、回答は同じです。
4. **実行場所。** `d1a.serving.serve` が MLX または PyTorch 上で System One API を提供し、Gemma 4 自身のエンコーダーで写真、ボイスメモ、動画を読み取ります。
5. **結果からの学習。** 意思決定は、それを行ったモデルとともに記録されます。結果が戻ってくると、それを使って確率を再フィットします。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/arch/model-dark.svg">
  <img src="docs/arch/model-light.svg" alt="LoRA を備えた Gemma 4 バックボーン、ポインターヘッド、選択肢に対するソフトマックス" width="100%">
</picture>

解説とその他の図: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。正確な規則: [docs/SPEC.md](docs/SPEC.md)。

<a id="learns-from-outcomes"></a>

## 結果から学習する

D1A はすべての意思決定を記録します。結果が戻ってくると確率を再較正し（本番で使用中）、ホールドアウトのゲートを通すことで再学習もできます（これまでにゲートを通過した再学習はありません）。詳細と数値: [docs/SELF_LEARNING.md](docs/SELF_LEARNING.md)。

<a id="demos"></a>

## デモ

1 回のフォワードパスでそれぞれ答える 13 のユースケースが [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground) にあります。そのうち 12 件（動画チェックを除くすべて）の録画は [docs/ja/DEMOS.md](docs/ja/DEMOS.md) にあります。

<table>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/routing.gif" alt="D1A 上で動作するモデルルーティングのデモ" width="100%"><br><b>モデルルーティング</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/tools.gif" alt="D1A 上で動作するツール呼び出しゲートのデモ" width="100%"><br><b>ツール呼び出しゲート</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/057ed20659fe42d5192345cfa7ec754e09e3394d/docs/gifs/en/photo.gif" alt="D1A 上で動作する写真チェックのデモ" width="100%"><br><b>写真チェック</b></td></tr>
</table>

<a id="train-your-own"></a>

## 自分で学習させる

```bash
uv run python -m d1a.training.recipe run recipes/d1a-e2b.yaml --out runs/d1a-e2b --device cuda
```

L4 が 1 枚あれば十分です。ファインチューニング、較正、スコアリング: [docs/TRAINING.md](docs/TRAINING.md)。レシピ: [recipes/README.md](recipes/README.md)。

<a id="status"></a>

## 現状

2026-10-22 時点:

- D1A-E4B-it v0.6 は 1 つのモデルとして日常的に使われています。意思決定（PR ラベラー、およびシャドーモードでの Hermes エージェントのツールゲート）と、ローカルアプリやエージェント向けの OpenAI 形式のチャットおよび埋め込み（プロンプトキャッシュ付き）です。
- SWE ベリファイアーは、1 つの issue の中では安価な軌跡ヒューリスティックに勝てません（ヌル結果）。結果に基づく再較正は機能します（[docs/SELF_LEARNING.md](docs/SELF_LEARNING.md)）。
- 写真、音声、動画のデモはゼロショットです。チェックポイントはテキストのみで学習されています（[docs/ja/DEMOS.md](docs/ja/DEMOS.md)）。
- Core ML はプレビューで、Mac では MLX より遅いです（[docs/ja/APPLE.md](docs/ja/APPLE.md)）。v0.6 のビルドはテキストのみ（入力は 512 トークンまで）で、D1A-E2B-it QAT のビルドは 5 つの機能すべてに対応し、Mac で検証済みですが iPhone ではまだです。
- 1 つの質問内の選択肢は、依然として互いに影響し合うことがあります。質問をまとめて尋ねても個別に尋ねても確率は同じですが、選択肢の順序は無関係ではありません。
- Kev が公開しているベンチマークの数値は Kev 自身の Qwen チェックポイントのものであり、D1A のものではありません。
- レポートは [docs/reports/](docs/reports/) にあります。

<a id="development"></a>

## 開発

```bash
uv sync --extra serve
uv run python -m pytest tests --unit -q   # what CI runs: every test file but the few tests/conftest.py's OUTSIDE_UNIT_TESTS lists
python scripts/check_license.py          # license and provenance rules (see CONTRIBUTING.md)
uv run python scripts/equivalence/metrics.py   # a rewritten module against its last version before the rewrite (api, metrics, benchmark, suite)
```

これらのスイートにはモデルの重みは不要です（トークナイザー、約 10 MB は Hub からダウンロードされます）。`tests/test_model.py` と `tests/test_mlx.py` には実際の重みが必要です。`tests/test_api.py` はコミット済みの小さなチェックポイントをプロセス内で提供します。代わりに稼働中のサーバーに対して実行するには、`D1A_BASE_URL` を設定してください（`pytest tests/test_api.py -m server`）。

<a id="releases"></a>

## リリース

D1A のコードは [Semantic Versioning](https://semver.org) でバージョン管理され、[GitHub](https://github.com/jonpol01/d1a/releases) でリリースされます。各バージョンの変更内容は [CHANGELOG.md](CHANGELOG.md) にあり、モデルのチェックポイントにはそれぞれ独自の Hugging Face タグが付いています。リリースの切り方: [CONTRIBUTING.md](CONTRIBUTING.md#releasing)。

<a id="license"></a>

## ライセンス

Apache-2.0: [LICENSE](LICENSE) と [NOTICE](NOTICE) を参照してください。

- D1A のコードは [Kev](https://github.com/jaredpalmer/kev) に由来します。Copyright 2026 Jared Palmer、Apache-2.0。Kev から取り込んで変更したファイルには、冒頭に通知が付いています。それらとすべての変更は [docs/UPSTREAM.md](docs/UPSTREAM.md) に一覧があります。
- D1A の変更の著作権は Copyright 2026 John Soliva、Apache-2.0 です。
- Gemma 4 は Google が Apache-2.0 で公開しています。Qwen のベースモデルも Apache-2.0 です。学習データセットにはそれぞれ独自のライセンスがあります。
- 作者: Kev は Jared Palmer（[@jaredpalmer](https://github.com/jaredpalmer)）によるもので、[Devin](https://devin.ai) で構築されました。D1A は John Soliva（[@jonpol01](https://github.com/jonpol01)）によるものです。
- Kev から引き継いだクレジット: Kev は、アーキテクチャの解説について [Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked)、API 設計について [TypeSafe](https://docs.typesafe.ai/api)、ベースモデルについて [Qwen](https://huggingface.co/Qwen/Qwen3.5-9B-Base)、そして貢献について [3x3xX3N0N](https://github.com/jaredpalmer/kev/issues/8) と [Radexito](https://github.com/Radexito) に感謝しています。関連研究: [Hydragen](https://arxiv.org/abs/2402.05099)、[DeFT](https://arxiv.org/abs/2404.00242)、[FIRST](https://arxiv.org/abs/2406.15657)。

<p><a href="../APPLE.md">English</a> · 日本語 · <a href="../ko/APPLE.md">한국어</a></p>

> この文書は英語版 [APPLE.md](../APPLE.md)（コミット 9eb86269 時点）の翻訳です。内容が異なる場合は英語版が正しい情報です。

<a id="apple-silicon"></a>

# Apple Silicon

Apple ハードウェア上の D1A: GPU 上の MLX と、Neural Engine 上の Core ML プレビュー。

<a id="apple-silicon-mlx"></a>

## Apple Silicon（MLX）

Apple Silicon では `d1a.serving.serve` が Gemma 4 チェックポイントを MLX（`d1a/backends/mlx.py`、mlx-lm 0.31.3）で実行します。アダプターはロード時に bf16 のベースにマージされ、すべてのリクエストは、state を 1 回と、質問ごとに 1 行として実行されます。`scripts/export_mlx.py` は、そのマージ済みモデルを、ベースもアダプターも不要でロードできるフォルダーとして書き出します。必要に応じて量子化もできます:

```bash
uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b --out runs/exports/d1a-e2b-mlx-bf16
uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b --q-bits 4 --q-group-size 64 --out runs/exports/d1a-e2b-mlx-4bit
uv run --extra mlx python scripts/export_mlx.py --run <a Gemma 4 12B-it QAT checkpoint> --q4-0 --out runs/exports/d1a-12b-mlx-q4_0   # Q4_0 grid
uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b-it --ct-base google/gemma-4-E2B-it-qat-w4a16-ct@<rev> --unmerged --out runs/exports/d1a-e2b-it-qat-mlx-q4   # a QAT checkpoint's own int4
uv run --extra serve python -m d1a.serving.serve --run runs/exports/d1a-e2b-mlx-4bit --port 8009
```

`--q4-0` は、mlx-lm の min/max スケールの代わりに、llama.cpp の Q4_0 グリッド（scale = d、bias = −8·d）上で、32 個ずつのグループとして 4 ビットを書き出します。`google/gemma-4-12B-it-qat-q4_0-unquantized` のように Q4_0 向けに学習されたベース用です。そのチェックポイントでは、Q4_0 の重み誤差は mlx-lm の 4-bit グループ 32 より約 20%、グループ 64 より 30% 小さくなります（#270）。

`--ct-base` は、run のベースの代わりに、同じモデルの compressed-tensors QAT チェックポイント（たとえば `google/gemma-4-E2B-it-qat-w4a16-ct`）を土台にエクスポートします。その int4 の Linear（対称、32 個ずつのグループ、int32 に 8 個ずつ下位ニブルから順に詰められ、それぞれ q + 8 として保存）は、すでに MLX の 4-bit レイアウトそのものなので、厳密にそのまま保持されます。ワードはそのまま、スケール s とバイアス −8·s で、s·q に逆量子化されます。それ以外のテンソルもそのまま保持され、埋め込みは通常どおり量子化されるか、`--q-embedding-bits` で独自のビット幅（8 にするとチェックポイントの bf16 により近くなります）、`--q-embedding-group-size` でグループサイズを指定できます（QAT が固定するのは Linear の 32 個ずつのグループであって、埋め込みのものではありません。64 にすると D1A-E2B-it で 0.17 GB 節約できます）。`--unmerged` が必要です。アダプターをマージするとグリッドから外れてしまうためです。D1A-E2B-it では、量子化された 276 個の Linear がすべて厳密です（d1a#341）。そのようなチェックポイント上で学習した run（`--base_ct`、docs/TRAINING.md）は、`--ct-base` なしでそのチェックポイント上にエクスポートされ、別のチェックポイントは拒否されます。`--media` は、QAT チェックポイント自身のエンコーダーを書き出します: ベースと同じ bf16 で、ビジョンの射影は int4 のテキストモデル向けに QAT で再学習されたものです。

パリティ（一致度）は、fp32 PyTorch パスのゴールデンベクトルに対して測定されます（`scripts/golden_vectors.py`: 209 レコード、274 質問: decision-v7 の開発用レコード、プレイグラウンドのプリセット、3 件の長い state のレコード）。M1 Max 上の JohnP1/d1a-e2b で、|Δp| は、ある質問内のいずれかの選択肢の確率の最大変化量です:

| | 重み | 最大 \|Δp\| | 平均 \|Δp\| | argmax の反転 | 精度（dev、fp32 は 0.811） | ECE（fp32 は 0.056） |
|---|---|---|---|---|---|---|
| MLX bf16 | 8.7 GB | 0.043 | 0.003 | 5（すべて、リファレンスのマージンが 0.025 未満のもの） | 0.824 | 0.061 |
| MLX 4-bit、グループ 64、埋め込みを含む | 2.5 GB | 0.363 | 0.054 | 23 | 0.811 | 0.051 |
| MLX 8-bit、レイヤーごとの埋め込みは 4-bit（`--q-bits 8 --q-per-layer-bits 4`） | 3.5 GB | 0.084 | 0.009 | 5（マージンが 0.053 未満） | 0.824 | 0.062 |

4-bit のエクスポートは精度を保ちますが、個々の確率を動かしすぎて fp32 モデルの代わりにはならない（回答の 8% が変わる）ため、公開していません。誤差は線形層に由来します: 線形層を bf16 にして両方の埋め込みを 4-bit にすると、最大値は 0.089 です。混合エクスポート（8-bit の線形層とトークン埋め込み、4-bit のレイヤーごとの埋め込み。これは E2B の重みの半分です）は bf16 に近いままです。M1 Max でのプロセスのフットプリント: 4-bit エクスポートのロード後は 3.1 GB、混合のものは 4.2 GB、提供後はさらに約 1 GB 増え、60 質問のリクエストでのピークは 5.3 / 6.4 GB です。6 質問のリクエストは約 0.4 s、60 質問のものは約 4.5 s で、どちらのエクスポートでも同じです。

Gemma 4 のレイヤーごとの埋め込みは、メモリに保持するのではなく、リクエストごとに重みファイルから読み込まれます（各リクエストに必要なのは、そのトークンの行だけです）: 8-bit の E2B エクスポートはロード後 4.22 GB の代わりに 2.96 GB、E4B は 6.88 GB の代わりに 5.37 GB を使い、確率はビット単位で同一で、遅くもなりません（ルックアップは 1 パスあたり約 0.25 ms 加わります）。`D1A_PLE_FLASH=0` を指定すると、メモリに保持します。

MPS 上の PyTorch（`D1A_BACKEND=torch`、Mac での学習、PyTorch メディアサーバー）: torch 2.14 より前では、MPS は 2^31 要素を超える埋め込みテーブルで、2^31 / その幅 を超える行すべてについて、ゼロの行を返します（pytorch/pytorch#188756）。
Gemma 4 のレイヤーごとのテーブルはそれほど大きく、E2B ではトークン ID 239,675 から（タブ、一部の CJK、絵文字、画像トークン）、E4B では
199,729 から（すべての句読点記号と数字のトークン）誤りになります。D1A はそのようなテーブルを MPS 上でインデックス指定して読み込み、これは厳密です（#353）。MLX、
CUDA、CPU は影響を受けたことはありません。この修正前の MPS 上の PyTorch の実行は、それらのトークンのレイヤーごとの入力をゼロとして読み込んでいました。

<a id="apple-neural-engine-core-ml"></a>

## Apple Neural Engine（Core ML）

D1A は Core ML を通じて Apple Neural Engine 上でも動作します。iPhone/iPad 向けのプレビューで、Mac では MLX の方が高速です。以下の v0.6 ビルドは
テキストのみ、512 トークン以下です（#224）。D1A-E2B-it QAT のビルドは 5 つの機能すべてに対応します（[下記](#d1a-e2b-it-qat-on-core-ml)）:
- テキストモデルは 6 または 7 レイヤーずつのチャンクに変換されます（E2B は 5 チャンク、E4B は 7 チャンク）。重みは int8（64 個のブロックごと）です。
- 埋め込みテーブルは、行のルックアップとしてホスト側に残ります。
- 各チャンクは独自のプロセスで動作します。

どちらの Core ML ビルドも、明確な質問ではすべて、完全精度のモデルと同じ回答を返します。Mac では GPU 上の MLX の方が数倍高速
なため、サービング用ホストでは MLX を使い続けます。Core ML は iPhone と iPad 向けで、GPU を空けておけます。
v0.6 のビルドには写真、音声、動画のエンコーダーがありません。各 Core ML リポジトリの `reference/run.py` は、ダウンロードしたファイルからリクエストに回答します。

2026-10-11 に、同じ 28 件の実際のリクエスト（42 質問、すべて 512 トークン以下）で、Apple M1 Max と Apple M4 上で測定しました。

| | E2B · MLX | E2B · Core ML | E4B · MLX | E4B · Core ML |
|---|---|---|---|---|
| バージョン | v0.6 | v0.6 | v0.6 | v0.6 |
| リポジトリ | [JohnP1/d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) | [JohnP1/d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml) | [JohnP1/d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8) | [JohnP1/d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml) |
| 実行環境 | Mac の GPU | Neural Engine（iPhone、iPad、M4 クラスの Mac） | Mac の GPU | Neural Engine |
| テキスト | はい | はい | はい | はい |
| 写真、音声、動画（16 フレーム、音声トラックなし） | はい | 未対応（次の予定） | はい | 未対応 |
| 入力長 | 長い文書 | 512 トークン以下 | 長い文書 | 512 トークン以下 |
| 質問 | yes/no、choice、score。各選択肢に較正済み確率が付く | 同じ | 同じ | 同じ |
| 言語 | 質問は英語、文書は英語または日本語 | 同じ | 同じ | 同じ |
| PR ラベル、最新の 487 件の PR: タイプ / 重大度 | 87.5% / 77.2%。P0/P1 は 30 件中 0 件を検出 | 同じモデル | 90.6% / 79.1%。P0/P1 は 30 件中 13 件 | 同じモデル |
| サイズ | 3.8 GB + メディア 1.0 GB | 3.8 GB（チャンク 2.0 GB + 埋め込み 1.7 GB） | 6.6 GB + メディア 1.0 GB | 6.5 GB（チャンク 4.2 GB + 埋め込み 2.3 GB） |
| リクエスト中央値、M4（p95） | **0.20 s**（0.26） | 0.96 s（0.97） | 0.63 s（0.89） | 1.80 s（1.91） |
| リクエスト中央値、M1 Max（p95） | **0.17 s**（0.23） | 9.7 s（10.0） ² | 0.81 s（1.33） | 2.93 s（2.95） |
| ロード時間、M4 / M1 Max | 2 s / 9 s | 40 s / 247 s | 3 s / – | 52 s / 106 s |
| fp32 PyTorch との回答の違い ¹ | 明確なもの 37 件中 0 件（M4: 42 件中 1 件の僅差） | 明確なもの 37 件中 0 件（M4: 42 件中 1 件の僅差） | 明確なもの 38 件中 0 件 | 明確なもの 38 件中 0 件 |
| 写真 / 音声 / 動画、中央値（M1 Max） | 14 件のデモリクエストで 0.83 s | – | – | – |

¹ 各ビルドを、同じ v0.6 モデルの fp32 PyTorch 再ビルドと比較しています。「明確」とは、リファレンスの最上位の
回答が 0.05 より大きく先行していることを指し、僅差のものはどちらにも反転し得ます。
² M1 Max の Neural Engine では、E2B のチャンクは CPU にフォールバックします: 各チャンクは Neural Engine の有無にかかわらず
同じ時間がかかります。M4 は、同じファイルをその Neural Engine 上で実行します。

<a id="d1a-e2b-it-qat-on-core-ml"></a>

### Core ML 上の D1A-E2B-it QAT

[JohnP1/d1a-e2b-it-qat-coreml](https://huggingface.co/JohnP1/d1a-e2b-it-qat-coreml) は、推奨の E2B-it ビルド
（[d1a-e2b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4)）の Core ML 変換版です: 意思決定、チャット、
写真、音声、動画（#341）。4.6 GB。
- **テキストモデル:** 35 レイヤーを 5 つのパッケージに分け、それぞれが 2 つのプリフィルサイズ（512 と 1,536 トークン）と、2,048 ポジションの
  KV キャッシュ上の 1 トークンデコードを備え、重みのコピーを 1 つ共有します。Google の QAT int4 の重みは厳密に保持されています（276 レイヤー、
  チェックポイント自身の整数とスケール）。D1A アダプターは、入力によってスケールされる別のブランチです: 意思決定では 1、チャットでは 0。
- **それ以外:** チャット用の int8 の出力層、写真と動画のエンコーダー（fp16、ピクセル経路は fp32）、音声
  エンコーダー（fp32）、そしてホスト側の行ルックアップとしての埋め込みテーブル（4-bit）。
- **GPU 上で動作します。** Neural Engine ではなく、iPhone や iPad ではまだ実行されていません。リポジトリの
  Python リファレンスランナー `reference/d1a_coreml_qat.py` 経由で使います。移植用に、`io.json` にホスト側が一覧されています。
- **Apple M1 Max でのチェック:**
  - 意思決定: MLX ビルドに対して反転した回答は 42 件中 0 件。fp32 に対する確率の最大変化は 0.0017（MLX
    ビルド自体は fp32 から 0.035）。
  - 写真、音声、動画（プレイグラウンドの 14 件のリクエスト、28 件の回答）: MLX ビルドに対して反転 0 件。
  - チャット（写真 8 枚、回答 427 トークン、teacher-forced）を CPU 上の fp32 PyTorch と比較: 1 トークンが異なる（僅差）。
- **M1 Max での所要時間:** 写真または音声クリップ付きの意思決定で 1 件あたり約 0.4 s、16 フレームの動画で 8〜11 s、
  チャットでトークンあたり約 85 ms。

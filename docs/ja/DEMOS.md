<p><a href="../DEMOS.md">English</a> · 日本語 · <a href="../ko/DEMOS.md">한국어</a></p>

> この文書は英語版 [DEMOS.md](../DEMOS.md)（コミット 9eb86269 時点）の翻訳です。内容が異なる場合は英語版が正しい情報です。

<a id="demos"></a>

# デモ

D1A が 1 回のフォワードパスで答える 13 のユースケース（うち 12 はプレイグラウンドから録画したもの）と、写真・音声・動画のデモが示すこと、示さないこと。

<a id="see-it-running"></a>

## 動作を見る

D1A がそれぞれ 1 回のフォワードパスで答える 13 のユースケースです。以下の 12 件は、[プレイグラウンド](https://github.com/jonpol01/d1a-playground)からライブで録画しました: 1–9 は M4 の Mac 上の MLX 8-bit、10–11 は M1 Max 上の [`d1a.serving.media`](USAGE.md#photos-voice-and-video) 経由、12 はサービング用ホストの D1A-E4B v0.2 によるものです:

<table>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/routing.gif" alt="D1A 上で動作するモデルルーティングのデモ" width="100%"><br><b>1. モデルルーティング</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/guardrails.gif" alt="D1A 上で動作するガードレールのデモ" width="100%"><br><b>2. ガードレール</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/tools.gif" alt="D1A 上で動作するツール呼び出しゲートのデモ" width="100%"><br><b>3. ツール呼び出しゲート</b></td></tr>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/inbox.gif" alt="D1A 上で動作する受信トレイのトリアージのデモ" width="100%"><br><b>4. 受信トレイのトリアージ</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/rerank.gif" alt="D1A 上で動作するリランキングのデモ" width="100%"><br><b>5. リランキング</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/evals.gif" alt="D1A 上で動作する LLM 評価のデモ" width="100%"><br><b>6. LLM 評価</b></td></tr>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/labeling.gif" alt="D1A 上で動作する一括ラベル付けのデモ" width="100%"><br><b>7. 一括ラベル付け</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/control.gif" alt="D1A 上で動作するリアルタイム制御のデモ" width="100%"><br><b>8. リアルタイム制御</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/gate.gif" alt="D1A 上で動作する確信度ゲートのデモ" width="100%"><br><b>9. 確信度ゲート</b></td></tr>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/057ed20659fe42d5192345cfa7ec754e09e3394d/docs/gifs/en/photo.gif" alt="D1A 上で動作する写真チェックのデモ" width="100%"><br><b>10. 写真チェック</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/057ed20659fe42d5192345cfa7ec754e09e3394d/docs/gifs/en/voice.gif" alt="D1A 上で動作する音声トリアージのデモ" width="100%"><br><b>11. 音声トリアージ</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/eccbe2b3a5e4de053f95ec58af387be2803964a0/docs/gifs/en/prlabel.gif" alt="D1A 上で動作する PR ラベラーのデモ" width="100%"><br><b>12. PR ラベラー</b></td></tr>
</table>

<a id="playground"></a>

## プレイグラウンド

上記のデモは [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground) にあります。これは `d1a.serving.serve` と通信する Next.js アプリです。Kev のリポジトリ内にあった開発者向けプレイグラウンド（パック方式と個別方式の回答、選択肢の順列、チェス）とそのラベルレビューのページは削除されました。[docs/removed-tools.md](../removed-tools.md) に、それらの内容と復元方法が書かれています。それらが使っていたサーバーのエンドポイント `/v1/systemone/separate` と `/v1/systemone/permute` は、今も残っています。

<a id="photo-voice-and-video-results"></a>

## 写真・音声・動画の結果

チェックポイントはテキストのみで学習されているため、これはゼロショットです。最初のテスト（[#78](https://github.com/jonpol01/d1a/issues/78)）では、D1A-E2B は 6 枚の配送写真のうち 6 枚で損傷を、6 枚のうち 5 枚で置き場所を、英語と日本語の 4 件のボイスメモのうち 4 件でリクエストを読み取りました。プレイグラウンドの 10 件のサンプルでは、D1A-E4B v0.3 は損傷が 6 件中 5 件、置き場所が 6 件中 6 件、リクエストが 4 件中 4 件です。これはデモとして扱ってください。測定結果ではありません。動画も同じ方法で読み取ります。サンプル写真から作った単一シーンのクリップでは、D1A-E4B v0.3 は写真の場合と同じ回答を返します。出来事の順序についての質問（「最後にそれはどこにあるか」）は、まだ信頼できません。

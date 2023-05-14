# Hybrid chatbot of GPT and local LLM
## 注意
- 開発中･未完成です

## 仕組み
- LLMで手持ちのファイルのembed vectorを計算
- queryのembed vectorを同様に計算
- 類似度の高いreferenceを抽出し､GPTのqueryに載せて回答

- ![](images/demo.png)

## Embedding vectorの計算
- 色々なアルゴリズムを利用可能です｡settingsフォルダ内のsettings.jsonを変更して変えられます｡
- GPTが最近の主流ですが､ローカルで動くSBERTが速度･精度･費用面で良さそうです(2023/5/15)
  - [SBERT](https://www.sbert.net/docs/pretrained_models.html#model-overview)
  - [Vicuna](https://huggingface.co/AlekseyKorshuk/vicuna-7b)
  - [GPT](https://platform.openai.com/docs/guides/embeddings)

## 日本語で質問して英語の文章を探す
- "Translate Japanese to English"にチェックを入れると可能です
- 日本語の質問を一旦､DeepLで英語に変換し､英語で内部処理を行った後､最後の回答をDeepLで日本語に戻します

## モチベーション
- GPTのAPIでもembedding vectorは計算可能です
  - しかし､数十ページ程度の日本語の文章を読ませるだけで､数十円かかります
  - ラボ内の書類データ､数千件以上を検索したかったので､コスト的に､この方法は採用できませんでした
  - そこで､embedding vectorの計算はローカルLLMに行わせるというアプローチを取ることにしました

## Install
- conda_requirements.txtで環境構築
- settingsフォルダにkey.pyを作成
  - GPTのAPI KEYを設定
    - GPT_API_KEY = "sk-XXXXXXX..."
  - DeepLのAPI KEYを設定(翻訳を使う場合)
    - DEEPL_API_KEY="...."
- UseDataフォルダにoriginalフォルダを作成
  - その中に､回答に含めたいpdf or text dataを入れる

## 動作確認マシン
- Ununtu 18.04.5 LTS
- メモリ32 GB + スワッピング (Vicunaの場合､言語モデルで30GB以上使います)
- Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz

## 起動
- embedサーバー(vicunaの場合､起動に5分くらいかかります)
```
python embed_server.py
```
- フロントエンドサーバー
```
streamlit run bot_server.py --server.address 0.0.0.0
``` 
- index 更新
  - ipynbを参照

## 主なTODO
- 言語を跨いだ検索 (e.g., 日本語で検索→英語の文章を発掘)
    - DeepLで翻訳予定
- embedding vectorの計算､チャンク区切りetcの最適化
- バグ類の修正
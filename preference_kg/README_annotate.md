# アノテーションツール

対話データセットから嗜好を抽出してアノテーションするツールです。

## 使い方

### 基本的な使い方

```bash
python -m preference_kg.annotate_dialogue
```

デフォルトで `data/raw/DailyDialog/dailydialog_trial.json` を読み込みます。

### 入力ファイルを指定する場合

```bash
python -m preference_kg.annotate_dialogue -i data/processed/annotated/output.json
```

### 出力先を指定する場合

```bash
python -m preference_kg.annotate_dialogue -i data/raw/DailyDialog/dailydialog_trial.json -o data/processed/annotated/output.json
```

## プログラムの動作

### 1. データ読み込みと分類
- 入力JSONファイルから対話データを読み込み
- `annotations` フィールドが空の対話のみを処理対象とする
- アノテーション済みの対話は自動的にスキップ

### 2. 対話の表示
- 画面上部にパネル形式で対話を固定表示
- **原文**と**和訳**の両方を表示
- 進捗情報（対話 X/Y）も表示

### 3. 嗜好の抽出
対話ごとに以下の質問を繰り返します：
1. 「この対話に嗜好は含まれますか？」（最初の質問）
2. 「まだ嗜好は含まれますか？」（2回目以降）

### 4. アノテーションフィールド
各嗜好について以下の順番で入力：
1. **entity**: エンティティ（自由入力）
2. **aspect**: アスペクト（自由入力、空欄可）
3. **classification**: 分類（選択）
   - `1. liking`
   - `2. wanting`
   - `3. need`
4. **sub_classification**: サブ分類（選択、classificationに応じて変化）
   - liking → `aesthetic/sensory`, `stimulation`, `identification`
   - wanting → `interest`, `goal`
   - need → `functional`, `social`, `personal`
5. **polarity**: 極性（選択）
   - `1. positive`
   - `2. neutral`
   - `3. negative`
6. **intensity**: 強度（選択）
   - `1. low`
   - `2. mid`
   - `3. high`
7. **context**: コンテキスト（自由入力）
8. **explicitness**: 明示性（選択）
   - `1. explicit`
   - `2. implicit`

**注意**: `timestamp` は自動で現在時刻が設定されます。

### 5. 保存
- アノテーション完了後、自動的に保存
- 出力先を指定しない場合: `data/processed/annotated/dailydialog_annotated.json`

## 特徴

### 画面表示
- **Richライブラリ**を使用した美しい表示
- 対話パネルが画面上部に固定表示
- 対話が変わると自動的に新しい対話を表示
- カラー表示で視認性向上

### スキップ機能
- 既にアノテーション済みの対話は自動的にスキップ
- 途中から再開する場合も安全

### 中断と復帰
- `Ctrl+C` で処理を中断可能
- 中断時、途中までのデータを保存するか選択可能
- 保存した場合、次回実行時に続きから再開

## ディレクトリ構造

```
preference-kg/
├── preference_kg/
│   ├── annotate_dialogue.py    # 対話アノテーションツール
│   └── annotate.py              # 文章アノテーションツール（旧版）
├── data/
│   ├── raw/
│   │   └── DailyDialog/
│   │       └── dailydialog_trial.json  # 入力データ
│   └── processed/
│       └── annotated/           # アノテーション済みデータの保存先
```

## 実行例

```bash
$ python -m preference_kg.annotate_dialogue

============================================================
対話データセット アノテーション
============================================================
各対話から嗜好を抽出してアノテーションします。
- 嗜好は0個以上含まれます
- aspectフィールドのみ空欄を許可
- Ctrl+C で中断できます
============================================================

総対話数: 30
アノテーション済み: 5 件（スキップ）
アノテーション対象: 25 件
============================================================

Enterキーを押して開始 

┌─ 対話 1/25 (全体ID: 6) ──────────────────────────┐
│ 【原文】                                          │
│ A: Say , Jim , how about going for a few beers  │
│ after dinner ?                                   │
│ B: You know that is tempting but is really not   │
│ good for our fitness .                           │
│                                                  │
│ 【和訳】                                         │
│ A: ねえ、ジム、夕食後にビールでも飲みにいかない？│
│ B: 確かに魅力的だけど、本当に健康に良くないよ。 │
└──────────────────────────────────────────────────┘

------------------------------------------------------------
この対話に嗜好は含まれますか？ (y/n): y

--- 嗜好 1 ---

[入力] entity: ビール
[入力] aspect: 
[選択] classification:
  1. liking
  2. wanting
  3. need
選択してください (1-3): 2
  → classification: wanting

[選択] sub_classification:
  1. interest
  2. goal
選択してください (1-2): 1
  → sub_classification: interest

[選択] polarity:
  1. positive
  2. neutral
  3. negative
選択してください (1-3): 1
  → polarity: positive

[選択] intensity:
  1. low
  2. mid
  3. high
選択してください (1-3): 2
  → intensity: mid

[入力] context: 夕食後の提案
[選択] explicitness:
  1. explicit
  2. implicit
選択してください (1-2): 1
  → explicitness: explicit

------------------------------------------------------------
まだ嗜好は含まれますか？ (y/n): n

------------------------------------------------------------
→ 合計 1 件の嗜好を抽出しました
------------------------------------------------------------
```

## 必要なライブラリ

- rich: 美しいターミナル表示のため
- loguru: ロギング
- その他: json, pathlib など（標準ライブラリ）

インストール:
```bash
uv add rich loguru
```

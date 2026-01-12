# PreferenceKG

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

対話テキストからユーザー嗜好を抽出し、知識グラフとして構造化するプロジェクト。

## Features

- **嗜好抽出**: LLM（GPT-4o, Gemma等）を用いた Few-shot + Chain of Thought による嗜好抽出
- **階層的評価**: Kiritchenko et al. (2006) の手法による階層的 F1-score
- **マルチステップ抽出**: Entity → Axis → Preference の3段階パイプライン

## Quick Start

```bash
# 環境構築
uv sync  # or pip install -e .

# 抽出実験
python preference_kg/experiments/run_extraction.py

# 評価
python preference_kg/experiments/run_evaluation.py

# 結果比較グラフ
python preference_kg/visualization/compare_results.py
```

## Project Structure

```
preference_kg/
├── evaluation/           # 評価ロジック
│   ├── normalizers.py    # 正規化関数
│   ├── metrics.py        # 階層的F1, Precision/Recall計算
│   └── reporter.py       # CSV出力, サマリー表示
│
├── experiments/          # 実験スクリプト
│   ├── run_extraction.py # 嗜好抽出実験
│   ├── run_extraction_multistep.py  # マルチステップ抽出
│   └── run_evaluation.py # 評価実験
│
├── extractors/           # 抽出器
│   ├── entity_extractor.py     # Step 1: Entity抽出
│   ├── axis_classifier.py      # Step 2: Axis分類
│   ├── preference_builder.py   # Step 3: 詳細構築
│   └── pipeline.py             # 統合パイプライン
│
├── annotation/           # アノテーション機能
├── visualization/        # グラフ・可視化
├── data/                 # データ読み込み
└── results/              # 実験結果
    ├── gpt-4o/
    ├── gpt-4o-mini/
    └── localLLM/

prompt_templates/         # プロンプトテンプレート
├── few_shot_extract_template_cot.txt   # 4-step CoT
└── schema_template_cot.json            # 出力スキーマ
```

## 嗜好スキーマ

抽出される嗜好の構造:

| フィールド | 説明 | 例 |
|-----------|------|-----|
| entity | 嗜好対象 | "classical music" |
| combined_axis | 嗜好軸 | "liking__aesthetic_sensory" |
| polarity | 極性 | positive / negative / neutral |
| intensity | 強度 | high / medium / low |
| context_tags | 文脈タグ | ["temporal-night", "location-home"] |

### 嗜好軸 (Axis)

- **Liking** (経験ベース): aesthetic_sensory, stimulation, identification, general
- **Wanting** (未来志向): interest, goal
- **Need** (欠乏ベース): functional, personal

## 評価指標

| 指標 | 説明 |
|------|------|
| Entity F1 | エンティティ抽出の精度 |
| Axis Accuracy | 親軸の正解率 |
| Sub-Axis Accuracy | 子軸の正解率 |
| Hierarchical F1 | 親子階層を考慮したF1 |
| Perfect Match | 全項目一致率 |

## License

MIT License

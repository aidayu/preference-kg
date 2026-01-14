# Results Directory Structure

実験結果と評価結果を整理したディレクトリ構造です。

## 📁 ディレクトリ構造

```
results/
├── experiments/          # 実験生データ（抽出結果JSON）
│   ├── gpt-4o/
│   ├── gpt-4o-mini/
│   └── localLLM/
│
├── evaluations/          # 評価結果
│   ├── gpt-4o/
│   │   └── {experiment_timestamp}/
│   │       ├── *.csv         # 評価指標
│   │       └── charts/       # 可視化チャート
│   ├── gpt-4o-mini/
│   └── localLLM/
│
├── reports/              # 最終レポート・モデル比較
│   └── (model_comparison.md など)
│
└── _archive/             # 古いファイル（旧形式）
    ├── charts/
    ├── gpt-4o/
    ├── gpt-4o-mini/
    └── localLLM/
```

## 🔖 命名規則

### 実験ファイル (experiments/)
- `experiment_results_{YYYYMMDD_HHMMSS}.json`
- `experiment_results_{model}_{method}.json` (localLLM用)

### 評価ファイル (evaluations/)
- `evaluation_{YYYYMMDD_HHMMSS}_{method}_{f1type}.csv`
- 例: `evaluation_20260114_132228_SemEMatch_3F1.csv`

### チャート (evaluations/{experiment}/charts/)
- `{evaluation_name}_f1.png` - F1スコア比較
- `{evaluation_name}_prf.png` - Precision/Recall/F1内訳

## 📊 評価方法コード

| コード | 説明 |
|--------|------|
| `PartEMatch` | 部分一致によるエンティティマッチング |
| `SemEMatch` | 意味的類似度によるエンティティマッチング |
| `3F1` | Micro/Macro/Weighted F1の3種類 |

## 🗓️ 実験履歴

| タイムスタンプ | モデル | 説明 |
|---------------|--------|------|
| 20260112_194507 | gpt-4o | Few-shot CoT抽出実験 |
| 20260112_153033 | gpt-4o-mini | CoT 3ステップ |
| 20260112_161922 | gpt-4o-mini | CoT 4ステップ |

"""評価結果の比較グラフを生成するスクリプト"""

import os
import matplotlib.pyplot as plt
import numpy as np

# 評価結果ファイルのパスと表示名
RESULT_FILES = {
    "GPT-4o\n(CoT4)": "/home/y-aida/Programs/preference-kg/preference_kg/results/gpt-4o/evaluation_results_20260112_194607.csv",
    "GPT-4o-mini\n(CoT4)": "/home/y-aida/Programs/preference-kg/preference_kg/results/gpt-4o-mini/evaluation_results_20260112_gpt4omini_CoT4step.csv",
    "GPT-4o-mini\n(CoT3)": "/home/y-aida/Programs/preference-kg/preference_kg/results/gpt-4o-mini/evaluation_results_20260112_153420_gpt4omini_CoT3step.csv",
    "Gemma3-27B\n(CoT4)": "/home/y-aida/Programs/preference-kg/preference_kg/results/localLLM/evaluation_results_gemma3_27B_CoT4step.csv",
}

# 指標カテゴリごとの定義
METRIC_CATEGORIES = {
    "Entity": ["Entity Recall", "Entity Precision", "Entity F1"],
    "Axis": ["Axis Accuracy", "Axis Recall", "Axis Precision", "Axis F1"],
    "Sub-Axis": ["Sub-Axis Accuracy", "Sub-Axis Recall", "Sub-Axis Precision", "Sub-Axis F1"],
    "Polarity": ["Polarity Accuracy", "Polarity Recall", "Polarity Precision", "Polarity F1"],
    "Intensity": ["Intensity Accuracy", "Intensity Recall", "Intensity Precision", "Intensity F1"],
    "Context": ["Context Accuracy", "Context Recall", "Context Precision", "Context F1"],
    "Perfect Match": ["Perfect Match Accuracy", "Perfect Match Recall", "Perfect Match Precision", "Perfect Match F1"],
}

COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]


def parse_csv(filepath: str) -> dict:
    """CSVファイルから指標を抽出"""
    metrics = {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                metric_name = parts[0].strip()
                value = parts[1].strip()
                
                if "%" in value:
                    try:
                        metrics[metric_name] = float(value.replace("%", ""))
                    except ValueError:
                        pass
    
    return metrics


def load_all_results() -> dict:
    """すべての評価結果を読み込み"""
    results = {}
    
    for model_name, filepath in RESULT_FILES.items():
        if os.path.exists(filepath):
            results[model_name] = parse_csv(filepath)
        else:
            print(f"Warning: {filepath} not found")
    
    return results


def plot_category_bar(category_name: str, metrics: list, results: dict, output_dir: str):
    """カテゴリごとの棒グラフを生成"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.18
    
    for i, model in enumerate(models):
        values = [results[model].get(m, 0.0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=COLORS[i % len(COLORS)])
        
        # 値ラベル
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    # X軸ラベルを簡略化
    short_labels = [m.replace(f"{category_name} ", "").replace("Sub-Axis ", "") for m in metrics]
    
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title(f"{category_name} Metrics Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    
    filename = f"comparison_{category_name.lower().replace(' ', '_').replace('-', '')}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return filepath


def main():
    """メイン関数"""
    print("=== 評価結果比較グラフ生成 ===\n")
    
    # データ読み込み
    results = load_all_results()
    print(f"読み込んだモデル: {len(results)}件")
    for model in results.keys():
        print(f"  - {model.replace(chr(10), ' ')}")
    print()
    
    # 出力ディレクトリ
    output_dir = "/home/y-aida/Programs/preference-kg/preference_kg/results/charts"
    os.makedirs(output_dir, exist_ok=True)
    
    # カテゴリごとにグラフ生成
    for category, metrics in METRIC_CATEGORIES.items():
        filepath = plot_category_bar(category, metrics, results, output_dir)
        print(f"✓ {category}: {filepath}")
    
    print(f"\n✓ 完了! ({len(METRIC_CATEGORIES)}件のグラフを生成)")


if __name__ == "__main__":
    main()


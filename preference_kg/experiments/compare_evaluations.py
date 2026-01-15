"""モデル間比較スクリプト

複数の評価結果CSVを読み込み、Macro-F1スコアを比較するグラフを生成する。
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================================
# === ユーザー設定 ===
# 比較したいモデル名とCSVパスのリスト
# ("表示名", "CSVパス")
# =====================================================================
MODELS = [
    ("gpt-4o-mini", "/home/y-aida/Programs/preference-kg/preference_kg/results/evaluations/gpt-4o-mini/20260112_161922/evaluation_20260115_131736_SemEMatch_3F1_4cot.csv"),
    ("gpt-4o", "/home/y-aida/Programs/preference-kg/preference_kg/results/evaluations/gpt-4o/20260112_194507/evaluation_20260114_132228_SemEMatch_3F1.csv"),
    # 追加のモデルがあればここに追記
    # ("Llama-3-8B", "path/to/csv"),
]

RESULTS_ROOT = Path(__file__).parent.parent / "results"
OUTPUT_DIR = RESULTS_ROOT / "comparisons"
# =====================================================================


def parse_evaluation_csv(filepath: str) -> pd.DataFrame | None:
    """CSVからサマリーテーブル（Macro-F1等）を抽出する"""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # サマリーテーブルを探す
    for i, line in enumerate(lines):
        if line.startswith("Metric,Micro-F1,"):
            summary_lines = []
            for j in range(i, len(lines)):
                if lines[j].strip() == "" or lines[j].startswith(",,"):
                    break
                summary_lines.append(lines[j].strip())
            
            if len(summary_lines) > 1:
                header = summary_lines[0].split(",")
                rows = [line.split(",") for line in summary_lines[1:]]
                return pd.DataFrame(rows, columns=header)
    
    return None


def create_comparison_chart(model_data: list[tuple[str, pd.DataFrame]], output_path: str):
    """モデル間のMacro-F1比較グラフを作成（学術論文向けスタイル）"""
    if not model_data:
        print("No valid data to plot.")
        return

    # メトリクス順序定義
    metrics_order = [
        "Entity", "Axis", "Sub-Axis", "Hierarchical Axis", 
        "Polarity", "Intensity", "Context", "Perfect Match"
    ]
    display_names = {
        "Entity": "Entity", "Axis": "Axis", "Sub-Axis": "Sub-Axis",
        "Hierarchical Axis": "H-Axis", "Polarity": "Polarity",
        "Intensity": "Intensity", "Context": "Context", "Perfect Match": "Perfect",
    }
    
    # データ構築
    plot_data = {model_name: [] for model_name, _ in model_data}
    labels = [display_names.get(m, m) for m in metrics_order]
    
    for model_name, df in model_data:
        # dfからデータを辞書化
        metric_map = dict(zip(df["Metric"], df["Macro-F1"]))
        
        for metric in metrics_order:
            val = float(metric_map.get(metric, 0))
            plot_data[model_name].append(val * 100) # %表記にする

    # プロット
    x = np.arange(len(labels))
    width = 0.8 / len(model_data)  # バーの幅を調整
    
    # 学術論文向けのスタイル設定
    # 色とハッチングの循環リスト
    colors_cycle = ['#ffffff', '#808080', '#404040', '#e0e0e0']
    hashes_cycle = ['///', '', '...', 'xx']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (model_name, _) in enumerate(model_data):
        offset = (i - (len(model_data) - 1) / 2) * width
        
        # 色とハッチングを選択
        color = colors_cycle[i % len(colors_cycle)]
        hatch = hashes_cycle[i % len(hashes_cycle)]
        
        rects = ax.bar(x + offset, plot_data[model_name], width, label=model_name, 
                       color=color, edgecolor='black', hatch=hatch)
        
        # 値を表示
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontfamily='serif', rotation=90)

    ax.set_xlabel('Evaluation Metric', fontsize=12, fontfamily='serif')
    ax.set_ylabel('Macro F1 Score (%)', fontsize=12, fontfamily='serif')
    ax.set_title('Model Performance Comparison (Macro F1)', fontsize=14, fontfamily='serif')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontfamily='serif', rotation=45)
    ax.legend(loc='lower right', fontsize=10, frameon=True, edgecolor='black', fancybox=False)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to: {output_path}")
    plt.close()


def main():
    print("=== モデル比較開始 ===")
    
    valid_data = []
    
    for model_name, csv_path in MODELS:
        print(f"Loading: {model_name}...")
        df = parse_evaluation_csv(csv_path)
        if df is not None:
            valid_data.append((model_name, df))
        else:
            print(f"Failed to load data for {model_name}")
            
    if not valid_data:
        print("No valid data found. Exiting.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = OUTPUT_DIR / "model_comparison_macro_f1.png"
    
    create_comparison_chart(valid_data, str(output_path))
    print("=== 完了 ===")


if __name__ == "__main__":
    main()

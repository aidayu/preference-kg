"""評価結果のグラフ化スクリプト

評価結果CSVファイルからMicro/Macro/Weighted F1の棒グラフを生成する
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================================================================
# === ユーザー設定 ===
# グラフ化したい評価結果CSVのパスをここに貼り付けてください
# =====================================================================
EVALUATION_CSV_PATH = "/home/y-aida/Programs/preference-kg/preference_kg/results/experiments/llama3_8B/experiment_results_llama3_8B_CoT4step.json"
# =====================================================================


def parse_evaluation_csv(filepath: str) -> dict:
    """
    新形式の評価結果CSVファイルをパースする
    
    Returns:
        {
            "info": {"model": str, "timestamp": str, ...},
            "summary": DataFrame (Metric, Micro-F1, Macro-F1, Weighted-F1),
            "detailed": DataFrame (Metric, Type, Precision, Recall, F1)
        }
    """
    data = {"info": {}, "summary": None, "detailed": None}
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 実験情報をパース
    for i, line in enumerate(lines):
        if line.startswith("Model,"):
            data["info"]["model"] = line.split(",")[1].strip()
        elif line.startswith("Timestamp,"):
            data["info"]["timestamp"] = line.split(",")[1].strip()
        elif line.startswith("Total Test Dialogues,"):
            data["info"]["n_dialogues"] = int(line.split(",")[1].strip())
    
    # サマリーテーブルを探す
    for i, line in enumerate(lines):
        if line.startswith("Metric,Micro-F1,"):
            # ヘッダー行から読み込む
            summary_lines = []
            for j in range(i, len(lines)):
                if lines[j].strip() == "" or lines[j].startswith(",,"):
                    break
                summary_lines.append(lines[j].strip())
            
            if len(summary_lines) > 1:
                header = summary_lines[0].split(",")
                rows = [line.split(",") for line in summary_lines[1:]]
                data["summary"] = pd.DataFrame(rows, columns=header)
            break
    
    # 詳細テーブルを探す
    for i, line in enumerate(lines):
        if line.startswith("Metric,Type,Precision"):
            detailed_lines = []
            for j in range(i, len(lines)):
                if lines[j].strip() == "" or lines[j].startswith(",,"):
                    break
                detailed_lines.append(lines[j].strip())
            
            if len(detailed_lines) > 1:
                header = detailed_lines[0].split(",")
                rows = [line.split(",") for line in detailed_lines[1:]]
                data["detailed"] = pd.DataFrame(rows, columns=header)
            break
    
    return data


def create_f1_comparison_chart(data: dict, output_path: str, title: str = None):
    """
    Micro/Macro/Weighted F1の比較棒グラフを作成
    """
    if data["summary"] is None:
        print("Error: Summary data not found in CSV")
        return
    
    df = data["summary"]
    
    # 数値に変換
    metrics = df["Metric"].tolist()
    micro_f1 = df["Micro-F1"].astype(float).tolist()
    macro_f1 = df["Macro-F1"].astype(float).tolist()
    weighted_f1 = df["Weighted-F1"].astype(float).tolist()
    
    # 表示用の短い名前
    display_names = {
        "Entity": "Entity",
        "Axis": "Axis",
        "Sub-Axis": "Sub-Axis",
        "Hierarchical Axis": "H-Axis",
        "Polarity": "Polarity",
        "Intensity": "Intensity",
        "Context": "Context",
        "Perfect Match": "Perfect",
    }
    labels = [display_names.get(m, m) for m in metrics]
    
    # グラフ作成
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 学術論文向けのグレースケール/ハイコントラスト設定
    colors = ['#e0e0e0', '#808080', '#202020']
    
    bars1 = ax.bar(x - width, [v * 100 for v in micro_f1], width, label='Micro-F1', color=colors[0], edgecolor='black')
    bars2 = ax.bar(x, [v * 100 for v in macro_f1], width, label='Macro-F1', color=colors[1], edgecolor='black')
    bars3 = ax.bar(x + width, [v * 100 for v in weighted_f1], width, label='Weighted-F1', color=colors[2], edgecolor='black')
    
    ax.set_xlabel('Evaluation Metric', fontsize=14, fontfamily='serif')
    ax.set_ylabel('F1 Score (%)', fontsize=14, fontfamily='serif')
    
    model = data["info"].get("model", "Unknown")
    if title is None:
        title = f'{model} Preference Extraction Evaluation'
    ax.set_title(title, fontsize=16, fontfamily='serif')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontfamily='serif', rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=12, frameon=True, edgecolor='black', fancybox=False)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # 値を表示
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            ha='center', va='bottom', fontsize=8, fontfamily='serif')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to: {output_path}")
    plt.close()


def create_precision_recall_chart(data: dict, output_path: str):
    """
    Precision/Recall/F1のグループ化棒グラフを作成（Micro平均のみ）
    """
    if data["detailed"] is None:
        print("Error: Detailed data not found in CSV")
        return
    
    df = data["detailed"]
    micro_df = df[df["Type"] == "Micro"].copy()
    
    metrics = micro_df["Metric"].tolist()
    precision = micro_df["Precision"].astype(float).tolist()
    recall = micro_df["Recall"].astype(float).tolist()
    f1 = micro_df["F1"].astype(float).tolist()
    
    display_names = {
        "Entity": "Entity", "Axis": "Axis", "Sub-Axis": "Sub-Axis",
        "Hierarchical Axis": "H-Axis", "Polarity": "Polarity",
        "Intensity": "Intensity", "Context": "Context", "Perfect Match": "Perfect",
    }
    labels = [display_names.get(m, m) for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 学術論文向けのグレースケール/ハイコントラスト設定
    colors = ['#ffffff', '#808080', '#000000']
    hatches = ['///', '', '']
    
    bars1 = ax.bar(x - width, [v * 100 for v in precision], width, label='Precision', color=colors[0], edgecolor='black', hatch=hatches[0])
    bars2 = ax.bar(x, [v * 100 for v in recall], width, label='Recall', color=colors[1], edgecolor='black', hatch=hatches[1])
    bars3 = ax.bar(x + width, [v * 100 for v in f1], width, label='F1', color=colors[2], edgecolor='black', hatch=hatches[2])
    
    ax.set_xlabel('Evaluation Metric', fontsize=14, fontfamily='serif')
    ax.set_ylabel('Score (%)', fontsize=14, fontfamily='serif')
    ax.set_title('Micro-Average: Precision / Recall / F1', fontsize=16, fontfamily='serif')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontfamily='serif', rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=12, frameon=True, edgecolor='black', fancybox=False)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    ax.set_axisbelow(True)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            ha='center', va='bottom', fontsize=8, fontfamily='serif')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to: {output_path}")
    plt.close()


def main():    
    parser = argparse.ArgumentParser(description="評価結果CSVからグラフを生成")
    parser.add_argument("csv_path", nargs="?", default=EVALUATION_CSV_PATH,
                        help="評価結果CSVファイルのパス（省略時は上記設定を使用）")
    parser.add_argument("-o", "--output", help="出力画像パス（省略時はCSVと同じ場所）")
    parser.add_argument("-t", "--title", help="グラフタイトル")
    parser.add_argument("--type", choices=["f1", "prf", "both"], default="both",
                        help="グラフタイプ: f1=Micro/Macro/Weighted, prf=Precision/Recall/F1, both=両方")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    print(f"評価結果ファイル: {csv_path}")
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return
    
    data = parse_evaluation_csv(str(csv_path))
    
    if args.type in ["f1", "both"]:
        output = args.output or str(csv_path.with_suffix("")) + "_f1.png"
        create_f1_comparison_chart(data, output, args.title)
    
    if args.type in ["prf", "both"]:
        output = str(csv_path.with_suffix("")) + "_prf.png"
        create_precision_recall_chart(data, output)


if __name__ == "__main__":
    main()


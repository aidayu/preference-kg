"""評価結果のグラフ化スクリプト

評価結果CSVファイルから棒グラフを生成する
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_evaluation_csv(filepath: str) -> dict:
    """
    評価結果CSVファイルをパースする
    
    Args:
        filepath: CSVファイルパス
    
    Returns:
        パースされた評価データの辞書
    """
    data = {
        "info": {},
        "metrics": {},
    }
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    def extract_metric_name(raw_name: str, suffix: str) -> str:
        """メトリクス名からサフィックスを除去"""
        name = raw_name.replace(suffix, "").strip()
        # "Hierarchical Axis" の形式を維持
        return name
    
    for line in lines:
        line = line.strip()
        if not line or line == ",,":
            continue
        
        parts = [p.strip() for p in line.split(",")]
        
        # 実験情報
        if parts[0] == "Model":
            data["info"]["model"] = parts[1]
        elif parts[0] == "Timestamp":
            data["info"]["timestamp"] = parts[1]
        
        # パーセント値を抽出する関数
        def extract_value(value_str: str) -> float | None:
            try:
                return float(value_str.replace("%", "").strip())
            except ValueError:
                return None
        
        # F1スコア
        if " F1" in parts[0] and len(parts) >= 2:
            metric_name = extract_metric_name(parts[0], " F1")
            value = extract_value(parts[1])
            if value is not None:
                if metric_name not in data["metrics"]:
                    data["metrics"][metric_name] = {}
                data["metrics"][metric_name]["f1"] = value
        
        # Precision
        elif " Precision" in parts[0] and len(parts) >= 2:
            metric_name = extract_metric_name(parts[0], " Precision")
            value = extract_value(parts[1])
            if value is not None:
                if metric_name not in data["metrics"]:
                    data["metrics"][metric_name] = {}
                data["metrics"][metric_name]["precision"] = value
        
        # Recall
        elif " Recall" in parts[0] and len(parts) >= 2:
            metric_name = extract_metric_name(parts[0], " Recall")
            value = extract_value(parts[1])
            if value is not None:
                if metric_name not in data["metrics"]:
                    data["metrics"][metric_name] = {}
                data["metrics"][metric_name]["recall"] = value
    
    return data


def create_evaluation_chart(data: dict, output_path: str, title: str = None):
    """
    評価結果の棒グラフを作成する
    
    Args:
        data: parse_evaluation_csvの戻り値
        output_path: 出力画像パス
        title: グラフタイトル（省略時は自動生成）
    """
    # メトリクス順序を定義
    metric_order = [
        "Entity", "Axis", "Sub-Axis", "Hierarchical Axis",
        "Polarity", "Intensity", "Context", "Perfect Match"
    ]
    
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
    
    # データ抽出
    metrics = []
    precision = []
    recall = []
    f1 = []
    
    for metric in metric_order:
        if metric in data["metrics"]:
            metrics.append(display_names.get(metric, metric))
            m = data["metrics"][metric]
            precision.append(m.get("precision", 0))
            recall.append(m.get("recall", 0))
            f1.append(m.get("f1", 0))
    
    # グラフ作成
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1', color='#e74c3c')
    
    ax.set_xlabel('Evaluation Metric', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    
    if title is None:
        model = data["info"].get("model", "Unknown")
        title = f'{model} Preference Extraction Evaluation'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # 値をバーの上に表示
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="評価結果CSVからグラフを生成")
    parser.add_argument("csv_path", help="評価結果CSVファイルのパス")
    parser.add_argument("-o", "--output", help="出力画像パス（省略時はCSVと同じ場所）")
    parser.add_argument("-t", "--title", help="グラフタイトル")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return
    
    # 出力パス
    if args.output:
        output_path = args.output
    else:
        output_path = str(csv_path.with_suffix(".png"))
    
    # データ読み込みとグラフ生成
    data = parse_evaluation_csv(str(csv_path))
    create_evaluation_chart(data, output_path, args.title)


if __name__ == "__main__":
    main()

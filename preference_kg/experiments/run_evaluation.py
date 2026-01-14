"""嗜好抽出結果の評価スクリプト

evaluation/ モジュールを使用して抽出結果を評価する
- 対話ごとにOptimal Matchingで評価
- Micro/Macro/Weighted F1で集計
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 親ディレクトリをパスに追加（experiments/ から preference_kg/ へ）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import (
    split_combined_axis,
    evaluate_dialogue,
    aggregate_all_metrics,
    DialogueResult,
    AggregatedMetrics,
)

# =====================================================================
# === ユーザー設定 ===
# 評価したい実験結果のパスをここに貼り付けてください
# =====================================================================
EXPERIMENT_RESULTS_PATH = "/home/y-aida/Programs/preference-kg/preference_kg/results/experiments/localLLM/experiment_results_gemma3_27B_CoT4step.json"
# =====================================================================

# --- 以下は自動生成（編集不要） ---
import re
RESULTS_ROOT = Path(__file__).parent.parent / "results"

# パスからモデル名とタイムスタンプを抽出
_exp_path = Path(EXPERIMENT_RESULTS_PATH)
_filename = _exp_path.stem  # "experiment_results_20260112_194507"
_timestamp_match = re.search(r"(\d{8}_\d{6})", _filename)
EXPERIMENT_TIMESTAMP = _timestamp_match.group(1) if _timestamp_match else datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_NAME = _exp_path.parent.name  # "gpt-4o"

# 評価結果の出力先
EVALUATION_OUTPUT_DIR = RESULTS_ROOT / "evaluations" / MODEL_NAME / EXPERIMENT_TIMESTAMP


def load_experiment_results(filepath: str) -> dict:
    """実験結果ファイルを読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_predictions(extracted_prefs: dict) -> list[dict]:
    """抽出結果を評価用の形式に変換する"""
    if "error" in extracted_prefs:
        return []
    
    predictions_raw = extracted_prefs.get("preferences", [])
    predictions = []
    
    for pred in predictions_raw:
        axis, sub_axis = split_combined_axis(pred.get("combined_axis", ""))
        pred_converted = pred.copy()
        pred_converted["axis"] = axis
        pred_converted["sub_axis"] = sub_axis
        predictions.append(pred_converted)
    
    return predictions


def evaluate_experiment(experiment_data: dict) -> tuple[list[DialogueResult], dict[str, AggregatedMetrics]] | None:
    """
    実験結果を評価する
    
    Args:
        experiment_data: 実験データ全体
    
    Returns:
        (dialogue_results, aggregated_metrics): 対話ごとの結果と集計結果
    """
    results = experiment_data.get("results", [])
    
    if len(results) == 0:
        print("評価対象データがありません。")
        return None
    
    print(f"\n--- 評価開始: {len(results)}件の対話 ---")
    
    dialogue_results = []
    
    for result_entry in results:
        dialogue_id = result_entry["dialogue_id"]
        ground_truths = result_entry.get("ground_truth_annotations", [])
        extracted_prefs = result_entry.get("extracted_preferences", {})
        predictions = convert_predictions(extracted_prefs)
        
        # 対話ごとの評価
        result = evaluate_dialogue(
            dialogue_id=str(dialogue_id),
            ground_truths=ground_truths,
            predictions=predictions,
        )
        dialogue_results.append(result)
        
        # 対話ごとのサマリー表示
        status = "SKIP" if result.n_gt == 0 else (
            "PERFECT" if result.perfect_tp == result.n_gt else "PARTIAL"
        )
        print(f"[ID:{dialogue_id}] GT={result.n_gt}, Pred={result.n_pred}, "
              f"Entity-F1={result.entity_f1:.2%}, Axis-F1={result.axis_f1:.2%} -> {status}")
    
    # 集計
    aggregated = aggregate_all_metrics(dialogue_results)
    
    return dialogue_results, aggregated


def print_aggregated_summary(aggregated: dict[str, AggregatedMetrics]):
    """集計結果を表示する"""
    print("\n" + "=" * 80)
    print("評価結果サマリー (Micro / Macro / Weighted F1)")
    print("=" * 80)
    
    header = f"{'Metric':<20} {'Micro-F1':>10} {'Macro-F1':>10} {'Weighted-F1':>12}"
    print(header)
    print("-" * 60)
    
    for name, metrics in aggregated.items():
        row = f"{name:<20} {metrics.micro_f1:>10.2%} {metrics.macro_f1:>10.2%} {metrics.weighted_f1:>12.2%}"
        print(row)
    
    print("=" * 80)


def save_aggregated_results(
    aggregated: dict[str, AggregatedMetrics],
    dialogue_results: list[DialogueResult],
    experiment_info: dict,
    output_path: str,
):
    """集計結果をCSVに保存する"""
    with open(output_path, "w", encoding="utf-8") as f:
        # 実験情報
        f.write("Experiment Information,,,,\n")
        f.write(f"Timestamp,{experiment_info.get('timestamp', '')},,,\n")
        f.write(f"Model,{experiment_info.get('model', '')},,,\n")
        f.write(f"Few-shot IDs,\"{experiment_info.get('few_shot_ids', '')}\",,,\n")
        f.write(f"Total Test Dialogues,{experiment_info.get('total_test_dialogues', len(dialogue_results))},,,\n")
        f.write(",,,,\n")
        
        # 集計結果
        f.write("Metric,Micro-F1,Macro-F1,Weighted-F1,Total TP\n")
        for name, metrics in aggregated.items():
            f.write(f"{name},{metrics.micro_f1:.4f},{metrics.macro_f1:.4f},{metrics.weighted_f1:.4f},{metrics.total_tp}\n")
        
        f.write(",,,,\n")
        
        # 詳細（Precision/Recallも含む）
        f.write("Detailed Metrics,,,,\n")
        f.write("Metric,Type,Precision,Recall,F1\n")
        for name, metrics in aggregated.items():
            f.write(f"{name},Micro,{metrics.micro_precision:.4f},{metrics.micro_recall:.4f},{metrics.micro_f1:.4f}\n")
            f.write(f"{name},Macro,{metrics.macro_precision:.4f},{metrics.macro_recall:.4f},{metrics.macro_f1:.4f}\n")
            f.write(f"{name},Weighted,{metrics.weighted_precision:.4f},{metrics.weighted_recall:.4f},{metrics.weighted_f1:.4f}\n")
    
    print(f"結果を保存しました: {output_path}")


def main(experiment_results_path=EXPERIMENT_RESULTS_PATH, evaluation_output_dir=EVALUATION_OUTPUT_DIR):
    """メイン評価関数"""
    # Path を文字列に変換
    experiment_results_path = str(experiment_results_path)
    evaluation_output_dir = str(evaluation_output_dir)
    
    print("=== 実験結果評価開始 ===")
    print(f"実験結果ファイル: {experiment_results_path}")
    print(f"評価結果出力先: {evaluation_output_dir}")
    
    print("\n[1/4] 実験結果読み込み中...")
    experiment_data = load_experiment_results(experiment_results_path)
    experiment_info = experiment_data.get("experiment_info", {})
    
    print(f"実験タイムスタンプ: {experiment_info.get('timestamp')}")
    print(f"モデル: {experiment_info.get('model')}")
    print(f"Few-shot IDs: {experiment_info.get('few_shot_ids')}")
    print(f"テスト対話数: {experiment_info.get('total_test_dialogues')}")
    
    print("\n[2/4] 評価実行中...")
    eval_result = evaluate_experiment(experiment_data)
    
    if eval_result is None:
        print("評価に失敗しました。")
        return
    
    dialogue_results, aggregated = eval_result
    
    print("\n[3/4] 結果表示中...")
    print_aggregated_summary(aggregated)
    
    print("\n[4/4] 結果保存中...")
    os.makedirs(evaluation_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(evaluation_output_dir, f"evaluation_{timestamp}_SemEMatch_3F1.csv")
    save_aggregated_results(aggregated, dialogue_results, experiment_info, output_path)
    
    print("\n✓ 評価完了！")


if __name__ == "__main__":
    main()


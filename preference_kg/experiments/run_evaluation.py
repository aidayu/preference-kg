"""嗜好抽出結果の評価スクリプト

evaluation/ モジュールを使用して抽出結果を評価する
"""

import json
import os
import sys
from datetime import datetime

# 親ディレクトリをパスに追加（experiments/ から preference_kg/ へ）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import (
    split_combined_axis,
    normalize_sub_axis,
    normalize_context,
    normalize_intensity,
    compute_hierarchical_metrics,
    compute_f1,
    save_evaluation_results,
    print_evaluation_summary,
    find_optimal_matching,
)

# --- 設定 ---
EXPERIMENT_RESULTS_PATH = "/home/y-aida/Programs/preference-kg/preference_kg/results/gpt-4o/experiment_results_20260112_194507.json"
RESULT_DIR = "/home/y-aida/Programs/preference-kg/preference_kg/results/gpt-4o"


def load_experiment_results(filepath: str) -> dict:
    """実験結果ファイルを読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_experiment(experiment_data: dict) -> dict | None:
    """
    実験結果を評価する
    
    Args:
        experiment_data: 実験データ全体
    
    Returns:
        metrics: 評価指標の辞書
    """
    results = experiment_data.get("results", [])
    
    total_gt_items = 0
    total_pred_items = 0
    matched_items = 0
    
    correct_axis = 0
    correct_sub_axis = 0
    correct_polarity = 0
    correct_intensity = 0
    correct_context = 0
    perfect_matches = 0
    
    tp_axis = 0
    tp_sub_axis = 0
    tp_polarity = 0
    tp_intensity = 0
    tp_context = 0
    
    matched_pairs = []
    
    print(f"\n--- 評価開始: {len(results)}件の対話 ---")
    
    for result_entry in results:
        dialogue_id = result_entry["dialogue_id"]
        ground_truths = result_entry.get("ground_truth_annotations", [])
        
        extracted_prefs = result_entry.get("extracted_preferences", {})
        if "error" in extracted_prefs:
            predictions = []
        else:
            predictions_raw = extracted_prefs.get("preferences", [])
            
            predictions = []
            for pred in predictions_raw:
                axis, sub_axis = split_combined_axis(pred.get("combined_axis", ""))
                pred_converted = pred.copy()
                pred_converted["axis"] = axis
                pred_converted["sub_axis"] = sub_axis
                predictions.append(pred_converted)
        
        total_gt_items += len(ground_truths)
        total_pred_items += len(predictions)
        
        # 最適マッチングを使用（Hungarian algorithm）
        matching_results = find_optimal_matching(ground_truths, predictions)
        
        for gt_idx, pred_idx, match, score in matching_results:
            gt = ground_truths[gt_idx]
            gt_entity = gt["entity"].lower().strip()
            
            status = "MISSING"
            if match:
                matched_items += 1
                
                is_axis_ok = (match["axis"] == gt["axis"])
                
                gt_sub_axis = normalize_sub_axis(gt.get("sub_axis"))
                pred_sub_axis = normalize_sub_axis(match.get("sub_axis"))
                is_sub_axis_ok = (pred_sub_axis == gt_sub_axis)
                is_polarity_ok = (match["polarity"] == gt["polarity"])
                
                gt_intensity = normalize_intensity(gt.get("intensity"))
                pred_intensity = normalize_intensity(match.get("intensity"))
                is_intensity_ok = (pred_intensity == gt_intensity) if (gt_intensity and pred_intensity) else False
                
                gt_context = normalize_context(gt.get("context", []))
                pred_context = normalize_context(match.get("context_tags", []))
                
                if len(gt_context) == 0 and len(pred_context) == 0:
                    is_context_ok = True
                elif len(gt_context) > 0 and len(pred_context) > 0:
                    is_context_ok = len(gt_context & pred_context) > 0
                else:
                    is_context_ok = False
                
                if is_axis_ok:
                    correct_axis += 1
                    tp_axis += 1
                if is_sub_axis_ok:
                    correct_sub_axis += 1
                    tp_sub_axis += 1
                if is_polarity_ok:
                    correct_polarity += 1
                    tp_polarity += 1
                if is_intensity_ok:
                    correct_intensity += 1
                    tp_intensity += 1
                if is_context_ok:
                    correct_context += 1
                    tp_context += 1
                
                if is_axis_ok and is_sub_axis_ok and is_polarity_ok and is_intensity_ok and is_context_ok:
                    perfect_matches += 1
                    status = "PERFECT"
                else:
                    status = "MISMATCH"
                
                matched_pairs.append({
                    "dialogue_id": dialogue_id,
                    "gt": gt,
                    "pred": match,
                })
            
            print(f"[ID:{dialogue_id}] GT: {gt_entity} ({gt['axis']}/{gt.get('sub_axis')}) -> {status}")
            if status == "MISMATCH" and match:
                print(f"   Expected: {gt['axis']}, {gt.get('sub_axis')}, {gt['polarity']}, {gt.get('intensity')}, {gt.get('context')}")
                print(f"   Got:      {match['axis']}, {match.get('sub_axis')}, {match['polarity']}, {match.get('intensity')}, {match.get('context_tags')}")
    
    if total_gt_items == 0:
        print("評価対象データがありません。")
        return None
    
    # Entity-level metrics
    entity_recall = matched_items / total_gt_items
    entity_precision = matched_items / total_pred_items if total_pred_items > 0 else 0
    entity_f1 = compute_f1(entity_precision, entity_recall)
    
    # Axis metrics
    axis_recall = correct_axis / total_gt_items
    axis_precision = tp_axis / total_pred_items if total_pred_items > 0 else 0
    axis_f1 = compute_f1(axis_precision, axis_recall)
    
    # Sub-Axis metrics
    sub_axis_recall = correct_sub_axis / total_gt_items
    sub_axis_precision = tp_sub_axis / total_pred_items if total_pred_items > 0 else 0
    sub_axis_f1 = compute_f1(sub_axis_precision, sub_axis_recall)
    
    # Hierarchical Axis metrics
    gt_axis_pairs = []
    pred_axis_pairs = []
    for pair in matched_pairs:
        gt = pair["gt"]
        pred = pair["pred"]
        gt_axis_pairs.append((gt.get("axis", ""), normalize_sub_axis(gt.get("sub_axis"))))
        pred_axis_pairs.append((pred.get("axis", ""), normalize_sub_axis(pred.get("sub_axis"))))
    
    hierarchical_metrics = compute_hierarchical_metrics(gt_axis_pairs, pred_axis_pairs)
    
    # Polarity metrics
    polarity_recall = correct_polarity / total_gt_items
    polarity_precision = tp_polarity / total_pred_items if total_pred_items > 0 else 0
    polarity_f1 = compute_f1(polarity_precision, polarity_recall)
    
    # Intensity metrics
    intensity_recall = correct_intensity / total_gt_items
    intensity_precision = tp_intensity / total_pred_items if total_pred_items > 0 else 0
    intensity_f1 = compute_f1(intensity_precision, intensity_recall)
    
    # Context metrics
    context_recall = correct_context / total_gt_items
    context_precision = tp_context / total_pred_items if total_pred_items > 0 else 0
    context_f1 = compute_f1(context_precision, context_recall)
    
    # Perfect match metrics
    perfect_recall = perfect_matches / total_gt_items
    perfect_precision = perfect_matches / total_pred_items if total_pred_items > 0 else 0
    perfect_f1 = compute_f1(perfect_precision, perfect_recall)
    
    return {
        "total_gt_items": total_gt_items,
        "total_pred_items": total_pred_items,
        "matched_items": matched_items,
        "correct_axis": correct_axis,
        "correct_sub_axis": correct_sub_axis,
        "correct_polarity": correct_polarity,
        "correct_intensity": correct_intensity,
        "correct_context": correct_context,
        "perfect_matches": perfect_matches,
        
        "entity_recall": entity_recall,
        "entity_precision": entity_precision,
        "entity_f1": entity_f1,
        
        "axis_accuracy": axis_recall,
        "axis_recall": axis_recall,
        "axis_precision": axis_precision,
        "axis_f1": axis_f1,
        
        "sub_axis_accuracy": sub_axis_recall,
        "sub_axis_recall": sub_axis_recall,
        "sub_axis_precision": sub_axis_precision,
        "sub_axis_f1": sub_axis_f1,
        
        "h_axis_recall": hierarchical_metrics["h_recall"],
        "h_axis_precision": hierarchical_metrics["h_precision"],
        "h_axis_f1": hierarchical_metrics["h_f1"],
        "h_gt_augmented_size": hierarchical_metrics["gt_augmented_size"],
        "h_pred_augmented_size": hierarchical_metrics["pred_augmented_size"],
        "h_intersection_size": hierarchical_metrics["intersection_size"],
        
        "polarity_accuracy": polarity_recall,
        "polarity_recall": polarity_recall,
        "polarity_precision": polarity_precision,
        "polarity_f1": polarity_f1,
        
        "intensity_accuracy": intensity_recall,
        "intensity_recall": intensity_recall,
        "intensity_precision": intensity_precision,
        "intensity_f1": intensity_f1,
        
        "context_accuracy": context_recall,
        "context_recall": context_recall,
        "context_precision": context_precision,
        "context_f1": context_f1,
        
        "perfect_match_accuracy": perfect_recall,
        "perfect_recall": perfect_recall,
        "perfect_precision": perfect_precision,
        "perfect_f1": perfect_f1,
    }


def main(experiment_results_path=EXPERIMENT_RESULTS_PATH, result_dir=RESULT_DIR):
    """メイン評価関数"""
    print("=== 実験結果評価開始 ===")
    print(f"実験結果ファイル: {experiment_results_path}")
    
    print("\n[1/4] 実験結果読み込み中...")
    experiment_data = load_experiment_results(experiment_results_path)
    experiment_info = experiment_data.get("experiment_info", {})
    
    print(f"実験タイムスタンプ: {experiment_info.get('timestamp')}")
    print(f"モデル: {experiment_info.get('model')}")
    print(f"Few-shot IDs: {experiment_info.get('few_shot_ids')}")
    print(f"テスト対話数: {experiment_info.get('total_test_dialogues')}")
    
    print("\n[2/4] 評価実行中...")
    metrics = evaluate_experiment(experiment_data)
    
    if not metrics:
        print("評価に失敗しました。")
        return
    
    print("\n[3/4] 結果表示中...")
    print_evaluation_summary(metrics)
    
    print("\n[4/4] 結果保存中...")
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(result_dir, f"evaluation_results_{timestamp}.csv")
    save_evaluation_results(metrics, experiment_info, output_path)
    
    print("\n✓ 評価完了！")


if __name__ == "__main__":
    main()

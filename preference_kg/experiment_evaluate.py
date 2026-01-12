import json
import csv
import os
from datetime import datetime

# --- 設定 ---
EXPERIMENT_RESULTS_PATH = "/home/y-aida/Programs/preference-kg/preference_kg/results/experiment_results_gemma3_27B_CoT4step.json"
RESULT_DIR = "/home/y-aida/Programs/preference-kg/preference_kg/results"


def load_experiment_results(filepath):
    """
    実験結果ファイルを読み込む
    
    Args:
        filepath: 実験結果JSONファイルのパス
    
    Returns:
        experiment_data: 実験データ全体
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def split_combined_axis(combined_axis):
    """
    combined_axisを分割してaxisとsub_axisに分ける
    
    Args:
        combined_axis: "axis__sub_axis" 形式の文字列
    
    Returns:
        (axis, sub_axis): タプル
    """
    if "__" in combined_axis:
        parts = combined_axis.split("__", 1)
        return parts[0], parts[1]
    else:
        return combined_axis, None


def normalize_sub_axis(sub_axis_value):
    """
    sub_axis値を正規化する
    GT: "aesthetic/sensory" <-> EP: "aesthetic_sensory"
    
    Args:
        sub_axis_value: sub_axis文字列
    
    Returns:
        正規化されたsub_axis値（スラッシュをアンダースコアに統一）
    """
    if sub_axis_value is None or sub_axis_value == "":
        return None
    
    # スラッシュをアンダースコアに変換して正規化
    normalized = sub_axis_value.lower().strip().replace("/", "_")
    return normalized


def normalize_context(context_value):
    """
    コンテキスト値を正規化する
    GT: "solo", "group" <-> EP: "social-solo", "social-group"
    GT: "Morning" <-> EP: "temporal-morning"
    GT: "Working/studying" <-> EP: "activity-working_studying"
    
    Args:
        context_value: リストまたは文字列
    
    Returns:
        正規化されたコンテキストのセット（小文字、プレフィックス除去）
    """
    if context_value is None:
        return set()
    
    def normalize_single_context(ctx):
        """単一のコンテキスト値を正規化"""
        ctx_lower = ctx.lower().strip()
        
        if ctx_lower == "none":
            return None
        
        # プレフィックスを除去（例: "social-solo" -> "solo"）
        if "-" in ctx_lower:
            # "social-solo" -> "solo"
            # "temporal-morning" -> "morning"
            # "activity-working_studying" -> "working_studying"
            parts = ctx_lower.split("-", 1)
            if len(parts) == 2:
                return parts[1].replace("_", "/")  # "working_studying" -> "working/studying"
        
        # スラッシュをアンダースコアに統一してから戻す（GT側の処理）
        return ctx_lower.replace("/", "_")
    
    if isinstance(context_value, list):
        # ["None"] や空リストを除外
        contexts = [normalize_single_context(c) for c in context_value if c]
        contexts = [c for c in contexts if c is not None]
        return set(contexts)
    elif isinstance(context_value, str):
        normalized = normalize_single_context(context_value)
        return set([normalized]) if normalized else set()
    
    return set()


def normalize_intensity(intensity_value):
    """
    intensity値を正規化する
    
    Args:
        intensity_value: "high", "mid", "medium", "low" など
    
    Returns:
        正規化されたintensity値
    """
    if intensity_value is None:
        return None
    
    intensity_lower = intensity_value.lower().strip()
    
    # "mid" と "medium" を統一
    if intensity_lower in ["mid", "medium"]:
        return "medium"
    
    return intensity_lower


def evaluate_experiment(experiment_data):
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
    
    # TP (True Positives) をカウント
    tp_axis = 0
    tp_sub_axis = 0
    tp_polarity = 0
    tp_intensity = 0
    tp_context = 0
    
    # マッチングペアの詳細を保存（intensity/context評価用）
    matched_pairs = []
    
    print(f"\n--- 評価開始: {len(results)}件の対話 ---")
    
    for result_entry in results:
        dialogue_id = result_entry["dialogue_id"]
        ground_truths = result_entry.get("ground_truth_annotations", [])
        
        # 抽出結果を取得
        extracted_prefs = result_entry.get("extracted_preferences", {})
        if "error" in extracted_prefs:
            predictions = []
        else:
            predictions_raw = extracted_prefs.get("preferences", [])
            
            # combined_axisを分割してaxisとsub_axisに変換
            predictions = []
            for pred in predictions_raw:
                axis, sub_axis = split_combined_axis(pred.get("combined_axis", ""))
                pred_converted = pred.copy()
                pred_converted["axis"] = axis
                pred_converted["sub_axis"] = sub_axis
                predictions.append(pred_converted)
        
        total_gt_items += len(ground_truths)
        total_pred_items += len(predictions)
        
        # マッチング済みの予測を追跡
        used_predictions = set()
        
        # 各ground truthに対して評価
        for gt in ground_truths:
            gt_entity = gt["entity"].lower().strip()
            
            # 対応する抽出結果を探す（Entity名が部分一致するもの）
            match = None
            match_idx = None
            for idx, pred in enumerate(predictions):
                if idx in used_predictions:
                    continue
                pred_entity = pred["entity"].lower().strip()
                if gt_entity in pred_entity or pred_entity in gt_entity:
                    match = pred
                    match_idx = idx
                    break
            
            # 結果判定
            status = "MISSING"
            if match:
                matched_items += 1
                used_predictions.add(match_idx)
                
                # 項目ごとの正誤判定
                is_axis_ok = (match["axis"] == gt["axis"])
                
                # Sub-Axis評価（正規化して比較）
                gt_sub_axis = normalize_sub_axis(gt.get("sub_axis"))
                pred_sub_axis = normalize_sub_axis(match.get("sub_axis"))
                is_sub_axis_ok = (pred_sub_axis == gt_sub_axis)
                is_polarity_ok = (match["polarity"] == gt["polarity"])
                
                # Intensity評価（正規化して比較）
                gt_intensity = normalize_intensity(gt.get("intensity"))
                pred_intensity = normalize_intensity(match.get("intensity"))
                is_intensity_ok = (pred_intensity == gt_intensity) if (gt_intensity and pred_intensity) else False
                
                # Context評価（部分一致: 少なくとも1つのタグが一致）
                gt_context = normalize_context(gt.get("context", []))
                pred_context = normalize_context(match.get("context_tags", []))
                
                # コンテキストマッチング: 両方が空、または少なくとも1つ一致
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
                
                # Perfect match: axis, sub_axis, polarity, intensity, context全て一致
                if is_axis_ok and is_sub_axis_ok and is_polarity_ok and is_intensity_ok and is_context_ok:
                    perfect_matches += 1
                    status = "PERFECT"
                else:
                    status = "MISMATCH"
                
                # マッチングペアを保存
                matched_pairs.append({
                    "dialogue_id": dialogue_id,
                    "gt": gt,
                    "pred": match,
                    "is_axis_ok": is_axis_ok,
                    "is_sub_axis_ok": is_sub_axis_ok,
                    "is_polarity_ok": is_polarity_ok,
                    "is_intensity_ok": is_intensity_ok,
                    "is_context_ok": is_context_ok
                })
            
            print(f"[ID:{dialogue_id}] GT: {gt_entity} ({gt['axis']}/{gt.get('sub_axis')}) -> {status}")
            if status == "MISMATCH" and match:
                print(f"   Expected: {gt['axis']}, {gt.get('sub_axis')}, {gt['polarity']}, {gt.get('intensity')}, {gt.get('context')}")
                print(f"   Got:      {match['axis']}, {match.get('sub_axis')}, {match['polarity']}, {match.get('intensity')}, {match.get('context_tags')}")
    
    # 評価指標を計算
    if total_gt_items == 0:
        print("評価対象データがありません。")
        return None
    
    # Entity-level metrics
    entity_recall = matched_items / total_gt_items if total_gt_items > 0 else 0
    entity_precision = matched_items / total_pred_items if total_pred_items > 0 else 0
    entity_f1 = (2 * entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0
    
    # Axis metrics
    axis_recall = correct_axis / total_gt_items
    axis_precision = tp_axis / total_pred_items if total_pred_items > 0 else 0
    axis_f1 = (2 * axis_precision * axis_recall) / (axis_precision + axis_recall) if (axis_precision + axis_recall) > 0 else 0
    
    # Sub-Axis metrics
    sub_axis_recall = correct_sub_axis / total_gt_items
    sub_axis_precision = tp_sub_axis / total_pred_items if total_pred_items > 0 else 0
    sub_axis_f1 = (2 * sub_axis_precision * sub_axis_recall) / (sub_axis_precision + sub_axis_recall) if (sub_axis_precision + sub_axis_recall) > 0 else 0
    
    # Polarity metrics
    polarity_recall = correct_polarity / total_gt_items
    polarity_precision = tp_polarity / total_pred_items if total_pred_items > 0 else 0
    polarity_f1 = (2 * polarity_precision * polarity_recall) / (polarity_precision + polarity_recall) if (polarity_precision + polarity_recall) > 0 else 0
    
    # Intensity metrics
    intensity_recall = correct_intensity / total_gt_items
    intensity_precision = tp_intensity / total_pred_items if total_pred_items > 0 else 0
    intensity_f1 = (2 * intensity_precision * intensity_recall) / (intensity_precision + intensity_recall) if (intensity_precision + intensity_recall) > 0 else 0
    
    # Context metrics
    context_recall = correct_context / total_gt_items
    context_precision = tp_context / total_pred_items if total_pred_items > 0 else 0
    context_f1 = (2 * context_precision * context_recall) / (context_precision + context_recall) if (context_precision + context_recall) > 0 else 0
    
    # Perfect match metrics
    perfect_recall = perfect_matches / total_gt_items
    perfect_precision = perfect_matches / total_pred_items if total_pred_items > 0 else 0
    perfect_f1 = (2 * perfect_precision * perfect_recall) / (perfect_precision + perfect_recall) if (perfect_precision + perfect_recall) > 0 else 0
    
    metrics = {
        "total_gt_items": total_gt_items,
        "total_pred_items": total_pred_items,
        "matched_items": matched_items,
        "correct_axis": correct_axis,
        "correct_sub_axis": correct_sub_axis,
        "correct_polarity": correct_polarity,
        "correct_intensity": correct_intensity,
        "correct_context": correct_context,
        "perfect_matches": perfect_matches,
        
        # Entity-level metrics
        "entity_recall": entity_recall,
        "entity_precision": entity_precision,
        "entity_f1": entity_f1,
        
        # Axis metrics
        "axis_accuracy": axis_recall,
        "axis_recall": axis_recall,
        "axis_precision": axis_precision,
        "axis_f1": axis_f1,
        
        # Sub-Axis metrics
        "sub_axis_accuracy": sub_axis_recall,
        "sub_axis_recall": sub_axis_recall,
        "sub_axis_precision": sub_axis_precision,
        "sub_axis_f1": sub_axis_f1,
        
        # Polarity metrics
        "polarity_accuracy": polarity_recall,
        "polarity_recall": polarity_recall,
        "polarity_precision": polarity_precision,
        "polarity_f1": polarity_f1,
        
        # Intensity metrics
        "intensity_accuracy": intensity_recall,
        "intensity_recall": intensity_recall,
        "intensity_precision": intensity_precision,
        "intensity_f1": intensity_f1,
        
        # Context metrics
        "context_accuracy": context_recall,
        "context_recall": context_recall,
        "context_precision": context_precision,
        "context_f1": context_f1,
        
        # Perfect match metrics
        "perfect_match_accuracy": perfect_recall,
        "perfect_recall": perfect_recall,
        "perfect_precision": perfect_precision,
        "perfect_f1": perfect_f1,
    }
    
    return metrics


def save_evaluation_results(metrics, experiment_info, output_path):
    """
    評価結果をCSVファイルに保存する
    
    Args:
        metrics: 評価指標
        experiment_info: 実験メタデータ
        output_path: 出力ファイルパス
    """
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        
        # 実験情報
        writer.writerow(["Experiment Information", "", ""])
        writer.writerow(["Timestamp", experiment_info.get("timestamp", ""), ""])
        writer.writerow(["Model", experiment_info.get("model", ""), ""])
        writer.writerow(["Few-shot IDs", str(experiment_info.get("few_shot_ids", [])), ""])
        writer.writerow(["Total Test Dialogues", experiment_info.get("total_test_dialogues", ""), ""])
        writer.writerow(["", "", ""])
        
        # 評価結果
        writer.writerow(["Metric", "Value", "Count"])
        writer.writerow(["Total Ground Truth Annotations", metrics["total_gt_items"], ""])
        writer.writerow(["Total LLM Predictions", metrics["total_pred_items"], ""])
        writer.writerow(["Matched Entities", metrics["matched_items"], ""])
        writer.writerow(["", "", ""])
        
        # Entity-level metrics
        writer.writerow(["Entity Recall", f"{metrics['entity_recall']:.2%}", f"{metrics['matched_items']}/{metrics['total_gt_items']}"])
        writer.writerow(["Entity Precision", f"{metrics['entity_precision']:.2%}", f"{metrics['matched_items']}/{metrics['total_pred_items']}"])
        writer.writerow(["Entity F1", f"{metrics['entity_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Axis metrics
        writer.writerow(["Axis Accuracy", f"{metrics['axis_accuracy']:.2%}", f"{metrics['correct_axis']}/{metrics['total_gt_items']}"])
        writer.writerow(["Axis Recall", f"{metrics['axis_recall']:.2%}", f"{metrics['correct_axis']}/{metrics['total_gt_items']}"])
        writer.writerow(["Axis Precision", f"{metrics['axis_precision']:.2%}", ""])
        writer.writerow(["Axis F1", f"{metrics['axis_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Sub-Axis metrics
        writer.writerow(["Sub-Axis Accuracy", f"{metrics['sub_axis_accuracy']:.2%}", f"{metrics['correct_sub_axis']}/{metrics['total_gt_items']}"])
        writer.writerow(["Sub-Axis Recall", f"{metrics['sub_axis_recall']:.2%}", f"{metrics['correct_sub_axis']}/{metrics['total_gt_items']}"])
        writer.writerow(["Sub-Axis Precision", f"{metrics['sub_axis_precision']:.2%}", ""])
        writer.writerow(["Sub-Axis F1", f"{metrics['sub_axis_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Polarity metrics
        writer.writerow(["Polarity Accuracy", f"{metrics['polarity_accuracy']:.2%}", f"{metrics['correct_polarity']}/{metrics['total_gt_items']}"])
        writer.writerow(["Polarity Recall", f"{metrics['polarity_recall']:.2%}", f"{metrics['correct_polarity']}/{metrics['total_gt_items']}"])
        writer.writerow(["Polarity Precision", f"{metrics['polarity_precision']:.2%}", ""])
        writer.writerow(["Polarity F1", f"{metrics['polarity_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Intensity metrics
        writer.writerow(["Intensity Accuracy", f"{metrics['intensity_accuracy']:.2%}", f"{metrics['correct_intensity']}/{metrics['total_gt_items']}"])
        writer.writerow(["Intensity Recall", f"{metrics['intensity_recall']:.2%}", f"{metrics['correct_intensity']}/{metrics['total_gt_items']}"])
        writer.writerow(["Intensity Precision", f"{metrics['intensity_precision']:.2%}", ""])
        writer.writerow(["Intensity F1", f"{metrics['intensity_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Context metrics
        writer.writerow(["Context Accuracy", f"{metrics['context_accuracy']:.2%}", f"{metrics['correct_context']}/{metrics['total_gt_items']}"])
        writer.writerow(["Context Recall", f"{metrics['context_recall']:.2%}", f"{metrics['correct_context']}/{metrics['total_gt_items']}"])
        writer.writerow(["Context Precision", f"{metrics['context_precision']:.2%}", ""])
        writer.writerow(["Context F1", f"{metrics['context_f1']:.2%}", ""])
        writer.writerow(["", "", ""])
        
        # Perfect match metrics
        writer.writerow(["Perfect Match Accuracy", f"{metrics['perfect_match_accuracy']:.2%}", f"{metrics['perfect_matches']}/{metrics['total_gt_items']}"])
        writer.writerow(["Perfect Match Recall", f"{metrics['perfect_recall']:.2%}", f"{metrics['perfect_matches']}/{metrics['total_gt_items']}"])
        writer.writerow(["Perfect Match Precision", f"{metrics['perfect_precision']:.2%}", ""])
        writer.writerow(["Perfect Match F1", f"{metrics['perfect_f1']:.2%}", ""])
    
    print(f"\n評価結果を保存: {output_path}")


def print_evaluation_summary(metrics):
    """評価指標のサマリーを表示する"""
    if not metrics:
        return
    
    print("\n" + "="*80)
    print("=== 評価結果サマリー ===")
    print("="*80)
    print(f"Total Ground Truth: {metrics['total_gt_items']}")
    print(f"Total Predictions:  {metrics['total_pred_items']}")
    print(f"Matched Entities:   {metrics['matched_items']}")
    print()
    
    print("--- Entity-level Metrics ---")
    print(f"  Recall:    {metrics['entity_recall']:.2%} ({metrics['matched_items']}/{metrics['total_gt_items']})")
    print(f"  Precision: {metrics['entity_precision']:.2%} ({metrics['matched_items']}/{metrics['total_pred_items']})")
    print(f"  F1:        {metrics['entity_f1']:.2%}")
    print()
    
    print("--- Axis Metrics ---")
    print(f"  Accuracy:  {metrics['axis_accuracy']:.2%} ({metrics['correct_axis']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['axis_recall']:.2%}")
    print(f"  Precision: {metrics['axis_precision']:.2%}")
    print(f"  F1:        {metrics['axis_f1']:.2%}")
    print()
    
    print("--- Sub-Axis Metrics ---")
    print(f"  Accuracy:  {metrics['sub_axis_accuracy']:.2%} ({metrics['correct_sub_axis']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['sub_axis_recall']:.2%}")
    print(f"  Precision: {metrics['sub_axis_precision']:.2%}")
    print(f"  F1:        {metrics['sub_axis_f1']:.2%}")
    print()
    
    print("--- Polarity Metrics ---")
    print(f"  Accuracy:  {metrics['polarity_accuracy']:.2%} ({metrics['correct_polarity']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['polarity_recall']:.2%}")
    print(f"  Precision: {metrics['polarity_precision']:.2%}")
    print(f"  F1:        {metrics['polarity_f1']:.2%}")
    print()
    
    print("--- Intensity Metrics ---")
    print(f"  Accuracy:  {metrics['intensity_accuracy']:.2%} ({metrics['correct_intensity']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['intensity_recall']:.2%}")
    print(f"  Precision: {metrics['intensity_precision']:.2%}")
    print(f"  F1:        {metrics['intensity_f1']:.2%}")
    print()
    
    print("--- Context Metrics ---")
    print(f"  Accuracy:  {metrics['context_accuracy']:.2%} ({metrics['correct_context']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['context_recall']:.2%}")
    print(f"  Precision: {metrics['context_precision']:.2%}")
    print(f"  F1:        {metrics['context_f1']:.2%}")
    print()
    
    print("--- Perfect Match Metrics (All Fields) ---")
    print(f"  Accuracy:  {metrics['perfect_match_accuracy']:.2%} ({metrics['perfect_matches']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['perfect_recall']:.2%}")
    print(f"  Precision: {metrics['perfect_precision']:.2%}")
    print(f"  F1:        {metrics['perfect_f1']:.2%}")
    print("="*80)


def main(experiment_results_path=EXPERIMENT_RESULTS_PATH, result_dir=RESULT_DIR):
    """
    メイン評価関数
    
    Args:
        experiment_results_path: 実験結果ファイルのパス
        result_dir: 結果保存ディレクトリ
    """
    print("=== 実験結果評価開始 ===")
    print(f"実験結果ファイル: {experiment_results_path}")
    
    # 実験結果を読み込み
    print("\n[1/4] 実験結果読み込み中...")
    experiment_data = load_experiment_results(experiment_results_path)
    experiment_info = experiment_data.get("experiment_info", {})
    
    print(f"実験タイムスタンプ: {experiment_info.get('timestamp')}")
    print(f"モデル: {experiment_info.get('model')}")
    print(f"Few-shot IDs: {experiment_info.get('few_shot_ids')}")
    print(f"テスト対話数: {experiment_info.get('total_test_dialogues')}")
    
    # 評価実行
    print("\n[2/4] 評価実行中...")
    metrics = evaluate_experiment(experiment_data)
    
    if not metrics:
        print("評価に失敗しました。")
        return
    
    # 結果表示
    print("\n[3/4] 結果表示中...")
    print_evaluation_summary(metrics)
    
    # 結果保存
    print("\n[4/4] 結果保存中...")
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(result_dir, f"evaluation_results_{timestamp}.csv")
    save_evaluation_results(metrics, experiment_info, output_path)
    
    print("\n✓ 評価完了！")


if __name__ == "__main__":
    main()

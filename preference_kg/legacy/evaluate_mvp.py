import json
import csv
import os
from datetime import datetime

# 相対インポートと絶対インポートの両方に対応
try:
    from .extract_mvp import zero_shot_extract_preferences, few_shot_extract_preferences
except ImportError:
    from extract_mvp import zero_shot_extract_preferences, few_shot_extract_preferences

# --- 設定 ---
DATA_FILE = "/home/y-aida/Programs/preference-kg/data/interim/dailydialog_annotated_SU_20251223_022000_aida.json"
RESULT_DIR = "/home/y-aida/Programs/preference-kg/notebooks/results"


def run_extraction(dataset, max_dialogues=30, extraction_type="zero_shot", save_output=True, result_dir=RESULT_DIR):
    """
    データセットに対してLLM抽出を実行し、結果を返す
    
    Args:
        dataset: 対話データのリスト
        max_dialogues: 処理する対話の最大数
        extraction_type: "zero_shot" または "few_shot"
        save_output: 抽出結果をJSONファイルに保存するか
    
    Returns:
        all_gpt_outputs: LLMの出力結果のリスト
    """
    # 抽出関数の選択
    if extraction_type == "zero_shot":
        extract_func = zero_shot_extract_preferences
    elif extraction_type == "few_shot":
        extract_func = few_shot_extract_preferences
    else:
        raise ValueError(f"Unknown extraction_type: {extraction_type}")
    
    all_gpt_outputs = []
    target_dataset = dataset[:max_dialogues]
    
    print(f"--- {extraction_type.upper()} 抽出開始: {len(target_dataset)}件の対話 ---")
    
    for i, entry in enumerate(target_dataset, 1):
        dialogue_text = entry["original_dialogue"]
        
        try:
            prediction_result = extract_func(dialogue_text)
            all_gpt_outputs.append({
                "dialogue_id": entry["dialogue_id"],
                "original_dialogue": dialogue_text,
                "predictions": prediction_result
            })
            print(f"[{i}/{len(target_dataset)}] ID:{entry['dialogue_id']} 完了")
        except Exception as e:
            print(f"[{i}/{len(target_dataset)}] Error processing ID {entry['dialogue_id']}: {e}")
            all_gpt_outputs.append({
                "dialogue_id": entry["dialogue_id"],
                "original_dialogue": dialogue_text,
                "error": str(e)
            })
    
    # LLM出力をJSONファイルに保存
    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(result_dir, exist_ok=True)
        gpt_output_file = os.path.join(result_dir, f"output_from_gpt4o-mini_{extraction_type}_{timestamp}.json")
        with open(gpt_output_file, "w", encoding="utf-8") as f:
            json.dump(all_gpt_outputs, f, indent=2, ensure_ascii=False)
        print(f"\nLLM出力を保存: {gpt_output_file}")
    
    return all_gpt_outputs


def evaluate_predictions(dataset, gpt_outputs):
    """
    LLMの予測結果とground truthを比較して評価する
    
    Args:
        dataset: ground truthを含む対話データのリスト
        gpt_outputs: LLMの出力結果のリスト
    
    Returns:
        metrics: 評価指標の辞書（Precision, Recall, F1を含む）
    """
    total_gt_items = 0
    total_pred_items = 0
    matched_items = 0
    
    correct_axis = 0
    correct_sub_axis = 0
    correct_polarity = 0
    perfect_matches = 0
    
    # TP (True Positives) をカウント
    tp_axis = 0
    tp_sub_axis = 0
    tp_polarity = 0
    
    # dialogue_id でマッピング
    predictions_map = {output["dialogue_id"]: output for output in gpt_outputs}
    
    # gpt_outputsに含まれるdialogue_idのみをフィルタリング
    extracted_dialogue_ids = set(predictions_map.keys())
    filtered_dataset = [entry for entry in dataset if entry["dialogue_id"] in extracted_dialogue_ids]
    
    print(f"\n--- 評価開始: {len(filtered_dataset)}件の対話 ---")
    
    # マッチング済みの予測を追跡
    used_predictions = set()
    
    for entry in filtered_dataset:
        dialogue_id = entry["dialogue_id"]
        ground_truths = entry["annotations"]
        
        # 対応する予測結果を取得
        prediction_output = predictions_map.get(dialogue_id)
        if not prediction_output or "error" in prediction_output:
            predictions = []
        else:
            predictions = prediction_output.get("predictions", {}).get("preferences", [])
        
        total_gt_items += len(ground_truths)
        total_pred_items += len(predictions)
        
        # 各ground truthに対して評価
        for gt in ground_truths:
            gt_entity = gt["entity"].lower().strip()
            
            # 対応する抽出結果を探す（Entity名が部分一致するもの）
            match = None
            match_key = None
            for idx, pred in enumerate(predictions):
                pred_key = f"{dialogue_id}_{idx}"
                if pred_key in used_predictions:
                    continue
                pred_label = pred["target_label"].lower().strip()
                if gt_entity in pred_label or pred_label in gt_entity:
                    match = pred
                    match_key = pred_key
                    break
            
            # 結果判定
            status = "MISSING"
            if match:
                matched_items += 1
                used_predictions.add(match_key)
                
                # 項目ごとの正誤判定
                is_axis_ok = (match["axis"] == gt["axis"])
                is_sub_axis_ok = (match.get("sub_axis") == gt.get("sub_axis"))
                is_polarity_ok = (match["polarity"] == gt["polarity"])

                if is_axis_ok:
                    correct_axis += 1
                    tp_axis += 1
                if is_sub_axis_ok:
                    correct_sub_axis += 1
                    tp_sub_axis += 1
                if is_polarity_ok:
                    correct_polarity += 1
                    tp_polarity += 1

                if is_axis_ok and is_sub_axis_ok and is_polarity_ok:
                    perfect_matches += 1
                    status = "PERFECT"
                else:
                    status = "MISMATCH"
            
            print(f"[ID:{dialogue_id}] GT: {gt_entity} ({gt['axis']}) -> {status}")
            if status == "MISMATCH":
                 print(f"   Expected: {gt['axis']}, {gt['sub_axis']}, {gt['polarity']}")
                 print(f"   Got:      {match['axis']}, {match['sub_axis']}, {match['polarity']}")
                
    
    # 評価指標を計算
    if total_gt_items == 0:
        print("評価対象データがありません。")
        return None
    
    # Entity-level metrics (マッチング自体の評価)
    entity_recall = matched_items / total_gt_items if total_gt_items > 0 else 0
    entity_precision = matched_items / total_pred_items if total_pred_items > 0 else 0
    entity_f1 = (2 * entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0
    
    # Axis metrics (Recallベース: GT中の正解率)
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
    
    # Perfect match metrics
    perfect_recall = perfect_matches / total_gt_items
    perfect_precision = perfect_matches / total_pred_items if total_pred_items > 0 else 0
    perfect_f1 = (2 * perfect_precision * perfect_recall) / (perfect_precision + perfect_recall) if (perfect_precision + perfect_recall) > 0 else 0
    
    # Accuracy (GT基準での正解率 = Recall)
    axis_accuracy = axis_recall
    sub_axis_accuracy = sub_axis_recall
    polarity_accuracy = polarity_recall
    perfect_match_accuracy = perfect_recall
    
    metrics = {
        "total_gt_items": total_gt_items,
        "total_pred_items": total_pred_items,
        "matched_items": matched_items,
        "correct_axis": correct_axis,
        "correct_sub_axis": correct_sub_axis,
        "correct_polarity": correct_polarity,
        "perfect_matches": perfect_matches,
        
        # Entity-level metrics
        "entity_recall": entity_recall,
        "entity_precision": entity_precision,
        "entity_f1": entity_f1,
        
        # Axis metrics
        "axis_accuracy": axis_accuracy,
        "axis_recall": axis_recall,
        "axis_precision": axis_precision,
        "axis_f1": axis_f1,
        
        # Sub-Axis metrics
        "sub_axis_accuracy": sub_axis_accuracy,
        "sub_axis_recall": sub_axis_recall,
        "sub_axis_precision": sub_axis_precision,
        "sub_axis_f1": sub_axis_f1,
        
        # Polarity metrics
        "polarity_accuracy": polarity_accuracy,
        "polarity_recall": polarity_recall,
        "polarity_precision": polarity_precision,
        "polarity_f1": polarity_f1,
        
        # Perfect match metrics
        "perfect_match_accuracy": perfect_match_accuracy,
        "perfect_recall": perfect_recall,
        "perfect_precision": perfect_precision,
        "perfect_f1": perfect_f1,
    }
    
    return metrics


def save_metrics(metrics, extraction_type="zero_shot", timestamp=None, result_dir=RESULT_DIR):
    """
    評価結果をCSVファイルに保存する
    （注: LLM出力は run_extraction 関数内で既に保存済み）
    
    Args:
        metrics: 評価指標
        extraction_type: "zero_shot" または "few_shot"
        timestamp: タイムスタンプ（Noneの場合は自動生成）
        result_dir: 結果ディレクトリ
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(result_dir, exist_ok=True)
    
    # 評価結果を保存
    if metrics:
        csv_output_file = os.path.join(result_dir, f"result_evaluate_mvp_{extraction_type}_{timestamp}.csv")
        with open(csv_output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
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
            
            # Perfect match metrics
            writer.writerow(["Perfect Match Accuracy", f"{metrics['perfect_match_accuracy']:.2%}", f"{metrics['perfect_matches']}/{metrics['total_gt_items']}"])
            writer.writerow(["Perfect Match Recall", f"{metrics['perfect_recall']:.2%}", f"{metrics['perfect_matches']}/{metrics['total_gt_items']}"])
            writer.writerow(["Perfect Match Precision", f"{metrics['perfect_precision']:.2%}", ""])
            writer.writerow(["Perfect Match F1", f"{metrics['perfect_f1']:.2%}", ""])
            
        print(f"評価結果を保存: {csv_output_file}")


def print_metrics(metrics):
    """評価指標を表示する"""
    if not metrics:
        return
    
    print("\n=== 評価結果 ===")
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
    print(f"  Recall:    {metrics['axis_recall']:.2%} ({metrics['correct_axis']}/{metrics['total_gt_items']})")
    print(f"  Precision: {metrics['axis_precision']:.2%}")
    print(f"  F1:        {metrics['axis_f1']:.2%}")
    print()
    
    print("--- Sub-Axis Metrics ---")
    print(f"  Accuracy:  {metrics['sub_axis_accuracy']:.2%} ({metrics['correct_sub_axis']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['sub_axis_recall']:.2%} ({metrics['correct_sub_axis']}/{metrics['total_gt_items']})")
    print(f"  Precision: {metrics['sub_axis_precision']:.2%}")
    print(f"  F1:        {metrics['sub_axis_f1']:.2%}")
    print()
    
    print("--- Polarity Metrics ---")
    print(f"  Accuracy:  {metrics['polarity_accuracy']:.2%} ({metrics['correct_polarity']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['polarity_recall']:.2%} ({metrics['correct_polarity']}/{metrics['total_gt_items']})")
    print(f"  Precision: {metrics['polarity_precision']:.2%}")
    print(f"  F1:        {metrics['polarity_f1']:.2%}")
    print()
    
    print("--- Perfect Match Metrics ---")
    print(f"  Accuracy:  {metrics['perfect_match_accuracy']:.2%} ({metrics['perfect_matches']}/{metrics['total_gt_items']})")
    print(f"  Recall:    {metrics['perfect_recall']:.2%} ({metrics['perfect_matches']}/{metrics['total_gt_items']})")
    print(f"  Precision: {metrics['perfect_precision']:.2%}")
    print(f"  F1:        {metrics['perfect_f1']:.2%}")


def compare_annotations_detailed(dataset, gpt_outputs, dialogue_ids=None):
    """
    Ground TruthとLLM Outputのアノテーションを一つずつ詳細に比較する
    
    Args:
        dataset: ground truthを含む対話データのリスト
        gpt_outputs: LLMの出力結果のリスト
        dialogue_ids: 比較する対話IDのリスト（Noneの場合は全対話）
    
    Returns:
        comparison_results: 各対話の比較結果のリスト
    """
    # dialogue_id でマッピング
    predictions_map = {output["dialogue_id"]: output for output in gpt_outputs}
    dataset_map = {entry["dialogue_id"]: entry for entry in dataset}
    
    # 比較対象の対話IDを決定
    if dialogue_ids is None:
        dialogue_ids = sorted(predictions_map.keys())
    
    comparison_results = []
    
    for dialogue_id in dialogue_ids:
        # データを取得
        entry = dataset_map.get(dialogue_id)
        prediction_output = predictions_map.get(dialogue_id)
        
        if not entry or not prediction_output:
            print(f"\n[Dialogue ID: {dialogue_id}] - データが見つかりません")
            continue
        
        ground_truths = entry.get("annotations", [])
        
        if "error" in prediction_output:
            predictions = []
            print(f"\n[Dialogue ID: {dialogue_id}] - LLM処理エラー: {prediction_output['error']}")
        else:
            predictions = prediction_output.get("predictions", {}).get("preferences", [])
        
        print(f"\n{'='*100}")
        print(f"対話ID: {dialogue_id}")
        print(f"{'='*100}")
        
        # 対話テキストを表示
        print(f"\n【対話内容】")
        dialogue_text = entry.get("original_dialogue", "")
        print(dialogue_text)
        
        # Ground Truthアノテーションを表示
        print(f"\n{'─'*100}")
        print(f"【Ground Truth】 ({len(ground_truths)}件)")
        print(f"{'─'*100}")
        for i, gt in enumerate(ground_truths, 1):
            print(f"\n  GT-{i}:")
            print(f"    Entity:   {gt['entity']}")
            print(f"    Axis:     {gt['axis']}")
            print(f"    Sub-Axis: {gt.get('sub_axis', 'N/A')}")
            print(f"    Polarity: {gt['polarity']}")
            if 'turn_index' in gt:
                print(f"    Turn:     {gt.get('turn_index', 'N/A')}")
        
        # LLM Outputアノテーションを表示
        print(f"\n{'─'*100}")
        print(f"【LLM Output】 ({len(predictions)}件)")
        print(f"{'─'*100}")
        for i, pred in enumerate(predictions, 1):
            print(f"\n  PRED-{i}:")
            print(f"    Target:       {pred['target_label']}")
            print(f"    Original:     {pred.get('original_mention', 'N/A')}")
            print(f"    Axis:         {pred['axis']}")
            print(f"    Sub-Axis:     {pred.get('sub_axis', 'N/A')}")
            print(f"    Polarity:     {pred['polarity']}")
            print(f"    Intensity:    {pred.get('intensity', 'N/A')}")
            print(f"    Turn:         {pred.get('turn_index', 'N/A')}")
            if 'reasoning' in pred:
                print(f"    Reasoning:    {pred['reasoning'][:100]}..." if len(pred['reasoning']) > 100 else f"    Reasoning:    {pred['reasoning']}")
        
        # マッチング分析
        print(f"\n{'─'*100}")
        print(f"【マッチング分析】")
        print(f"{'─'*100}")
        
        matched_pairs = []
        unmatched_gt_indices = set(range(len(ground_truths)))
        unmatched_pred_indices = set(range(len(predictions)))
        
        # マッチングを試行
        for gt_idx, gt in enumerate(ground_truths):
            gt_entity = gt["entity"].lower().strip()
            
            for pred_idx, pred in enumerate(predictions):
                if pred_idx not in unmatched_pred_indices:
                    continue
                    
                pred_label = pred["target_label"].lower().strip()
                
                if gt_entity in pred_label or pred_label in gt_entity:
                    # マッチング成功
                    is_axis_match = gt["axis"] == pred["axis"]
                    is_sub_axis_match = gt.get("sub_axis") == pred.get("sub_axis")
                    is_polarity_match = gt["polarity"] == pred["polarity"]
                    
                    matched_pairs.append({
                        "gt_idx": gt_idx,
                        "pred_idx": pred_idx,
                        "gt": gt,
                        "pred": pred,
                        "is_axis_match": is_axis_match,
                        "is_sub_axis_match": is_sub_axis_match,
                        "is_polarity_match": is_polarity_match,
                        "is_perfect": is_axis_match and is_sub_axis_match and is_polarity_match
                    })
                    
                    unmatched_gt_indices.discard(gt_idx)
                    unmatched_pred_indices.discard(pred_idx)
                    break
        
        # マッチング結果を表示
        if matched_pairs:
            print(f"\n✓ マッチしたペア ({len(matched_pairs)}組):")
            for pair in matched_pairs:
                status = "完全一致" if pair["is_perfect"] else "部分一致"
                print(f"\n  GT-{pair['gt_idx']+1} ←→ PRED-{pair['pred_idx']+1} [{status}]")
                print(f"    GT:   {pair['gt']['entity']} | {pair['gt']['axis']} / {pair['gt'].get('sub_axis', 'N/A')} / {pair['gt']['polarity']}")
                print(f"    PRED: {pair['pred']['target_label']} | {pair['pred']['axis']} / {pair['pred'].get('sub_axis', 'N/A')} / {pair['pred']['polarity']}")
                
                if not pair["is_axis_match"]:
                    print(f"      ⚠ Axis不一致: GT={pair['gt']['axis']} vs PRED={pair['pred']['axis']}")
                if not pair["is_sub_axis_match"]:
                    print(f"      ⚠ Sub-Axis不一致: GT={pair['gt'].get('sub_axis')} vs PRED={pair['pred'].get('sub_axis')}")
                if not pair["is_polarity_match"]:
                    print(f"      ⚠ Polarity不一致: GT={pair['gt']['polarity']} vs PRED={pair['pred']['polarity']}")
        
        if unmatched_gt_indices:
            print(f"\n✗ マッチしなかったGround Truth ({len(unmatched_gt_indices)}件):")
            for idx in sorted(unmatched_gt_indices):
                gt = ground_truths[idx]
                print(f"  GT-{idx+1}: {gt['entity']} | {gt['axis']} / {gt.get('sub_axis', 'N/A')} / {gt['polarity']}")
        
        if unmatched_pred_indices:
            print(f"\n+ 余分なLLM予測 ({len(unmatched_pred_indices)}件):")
            for idx in sorted(unmatched_pred_indices):
                pred = predictions[idx]
                print(f"  PRED-{idx+1}: {pred['target_label']} | {pred['axis']} / {pred.get('sub_axis', 'N/A')} / {pred['polarity']}")
        
        # 結果を保存
        result = {
            "dialogue_id": dialogue_id,
            "gt_count": len(ground_truths),
            "pred_count": len(predictions),
            "matched_count": len(matched_pairs),
            "unmatched_gt_count": len(unmatched_gt_indices),
            "extra_pred_count": len(unmatched_pred_indices),
            "matched_pairs": matched_pairs
        }
        comparison_results.append(result)
    
    return comparison_results


def evaluate(dataset_path=DATA_FILE, result_dir=RESULT_DIR, extraction_type="zero_shot", max_dialogues=30):
    """
    LLM抽出と評価を実行するメイン関数
    
    Args:
        extraction_type: "zero_shot" または "few_shot"
        max_dialogues: 処理する対話の最大数
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # データの読み込み
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"エラー: {dataset_path} が見つかりません。")
        return
    
    # ステップ1: LLM抽出を実行
    gpt_outputs = run_extraction(dataset, max_dialogues, extraction_type, result_dir=result_dir)
    
    # ステップ2: 評価を実行（datasetをそのまま渡し、evaluate_predictions内でフィルタリング）
    metrics = evaluate_predictions(dataset, gpt_outputs)
    
    # ステップ3: 結果を表示
    print_metrics(metrics)
    
    # ステップ4: 結果を保存
    save_metrics(metrics, extraction_type, timestamp, result_dir=result_dir)

if __name__ == "__main__":
    evaluate()
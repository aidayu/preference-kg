"""Entity Similarity Manual Review Tool

エンティティペアの類似度を人手で確認し、適切な閾値を決定するためのツール。
各ペアに対して「同一エンティティか否か」を判定し、
その結果から最適な閾値を算出する。
"""

import json
import sys
import os
from pathlib import Path

import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preference_kg.evaluation import split_combined_axis
from preference_kg.evaluation.semantic_similarity import compute_entity_similarity


def load_experiment_results(filepath: str) -> dict:
    """実験結果を読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_unique_entity_pairs(experiment_data: dict) -> pd.DataFrame:
    """
    各GTエンティティに対するベストマッチのPredエンティティを抽出し、
    ユニークな（gt_entity, pred_entity）ペアを返す。
    
    Returns:
        DataFrame with columns: dialogue_id, gt_entity, pred_entity, similarity
    """
    all_pairs = []
    
    for entry in experiment_data.get("results", []):
        dialogue_id = entry["dialogue_id"]
        ground_truths = entry.get("ground_truth_annotations", [])
        extracted = entry.get("extracted_preferences", {})
        predictions = extracted.get("preferences", [])
        
        if not ground_truths or not predictions:
            continue
        
        # 各GTに対して最も類似度が高いPredを見つける
        for gt in ground_truths:
            gt_entity = gt.get("entity", "").strip()
            if not gt_entity:
                continue
            
            best_pred = None
            best_similarity = -1
            
            for pred in predictions:
                pred_entity = pred.get("entity", "").strip()
                if not pred_entity:
                    continue
                
                similarity = compute_entity_similarity(gt_entity, pred_entity)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pred = pred_entity
            
            if best_pred:
                all_pairs.append({
                    "dialogue_id": dialogue_id,
                    "gt_entity": gt_entity,
                    "pred_entity": best_pred,
                    "similarity": best_similarity,
                })
    
    # ユニークなペアを抽出（類似度でソート）
    df = pd.DataFrame(all_pairs)
    df = df.drop_duplicates(subset=["gt_entity", "pred_entity"])
    df = df.sort_values("similarity", ascending=False).reset_index(drop=True)
    
    return df


def create_review_csv(df: pd.DataFrame, output_path: str):
    """
    レビュー用CSVを作成する。
    'is_same_entity'列を追加し、人手でY/Nを記入できるようにする。
    """
    review_df = df.copy()
    review_df["is_same_entity"] = ""  # 人手で記入: Y or N
    review_df["notes"] = ""  # コメント欄
    
    review_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"レビュー用CSVを作成しました: {output_path}")
    print(f"総ペア数: {len(review_df)}")
    print("\n'is_same_entity' 列に Y（同一）または N（別物）を記入してください。")


def analyze_reviewed_csv(reviewed_path: str):
    """
    レビュー済みCSVを分析し、最適閾値を算出する。
    """
    df = pd.read_csv(reviewed_path)
    
    # is_same_entity が記入されているかチェック
    reviewed = df[df["is_same_entity"].isin(["Y", "N", "y", "n"])]
    
    if len(reviewed) == 0:
        print("エラー: 'is_same_entity' 列が記入されていません。")
        print("Y（同一エンティティ）または N（別物）を記入してください。")
        return
    
    reviewed = reviewed.copy()
    reviewed["is_same"] = reviewed["is_same_entity"].str.upper() == "Y"
    
    print(f"\n=== レビュー結果分析 ===")
    print(f"レビュー済み: {len(reviewed)} / {len(df)} ペア")
    print(f"同一: {reviewed['is_same'].sum()} ペア")
    print(f"別物: {(~reviewed['is_same']).sum()} ペア")
    
    # 類似度帯ごとの正答率
    print("\n--- 類似度帯ごとの同一率 ---")
    bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["0.0-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    reviewed["sim_bin"] = pd.cut(reviewed["similarity"], bins=bins, labels=labels, include_lowest=True)
    
    for bin_label in labels:
        subset = reviewed[reviewed["sim_bin"] == bin_label]
        if len(subset) > 0:
            same_rate = subset["is_same"].mean() * 100
            print(f"  {bin_label}: {len(subset):3d} ペア, 同一率 {same_rate:.1f}%")
    
    # 最適閾値の算出（同一エンティティの最小類似度）
    same_entities = reviewed[reviewed["is_same"]]
    different_entities = reviewed[~reviewed["is_same"]]
    
    if len(same_entities) > 0:
        min_same_sim = same_entities["similarity"].min()
        print(f"\n同一エンティティの最小類似度: {min_same_sim:.4f}")
    
    if len(different_entities) > 0:
        max_diff_sim = different_entities["similarity"].max()
        print(f"別物エンティティの最大類似度: {max_diff_sim:.4f}")
    
    # 重複がある場合（同一と別物で類似度が重なる）
    if len(same_entities) > 0 and len(different_entities) > 0:
        overlap_min = max(different_entities["similarity"].min(), same_entities["similarity"].min())
        overlap_max = min(different_entities["similarity"].max(), same_entities["similarity"].max())
        
        if overlap_min < overlap_max:
            print(f"\n⚠ 同一/別物の類似度が重複: {overlap_min:.4f} - {overlap_max:.4f}")
            print("  この範囲では閾値による完全な分離は困難です。")
    
    # 推奨閾値の計算（F1最大化）
    print("\n--- 閾値ごとの性能 ---")
    thresholds_to_test = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    best_f1 = 0
    best_threshold = 0
    
    for thresh in thresholds_to_test:
        # 閾値以上をマッチと判定
        predicted_same = reviewed["similarity"] >= thresh
        
        # TP: 実際に同一で、マッチと判定された
        tp = (reviewed["is_same"] & predicted_same).sum()
        # FP: 実際は別物だが、マッチと判定された
        fp = (~reviewed["is_same"] & predicted_same).sum()
        # FN: 実際は同一だが、マッチと判定されなかった
        fn = (reviewed["is_same"] & ~predicted_same).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  閾値 {thresh:.2f}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"\n★ 推奨閾値: {best_threshold:.2f} (F1={best_f1:.3f})")
    
    # 誤分類の詳細
    print("\n--- 誤分類の詳細（閾値 {:.2f}）---".format(best_threshold))
    
    predicted_same = reviewed["similarity"] >= best_threshold
    
    # False Positives: 別物をマッチと判定
    fp_cases = reviewed[~reviewed["is_same"] & predicted_same]
    if len(fp_cases) > 0:
        print(f"\n[FP] 別物だがマッチと判定 ({len(fp_cases)} 件):")
        for _, row in fp_cases.iterrows():
            print(f"  {row['similarity']:.3f}: '{row['gt_entity']}' <-> '{row['pred_entity']}'")
    
    # False Negatives: 同一だがマッチと判定されなかった
    fn_cases = reviewed[reviewed["is_same"] & ~predicted_same]
    if len(fn_cases) > 0:
        print(f"\n[FN] 同一だがマッチと判定されず ({len(fn_cases)} 件):")
        for _, row in fn_cases.iterrows():
            print(f"  {row['similarity']:.3f}: '{row['gt_entity']}' <-> '{row['pred_entity']}'")


def interactive_review(df: pd.DataFrame, output_path: str):
    """
    インタラクティブにペアをレビューする。
    
    操作:
        y: 同一エンティティ
        n: 別物
        s: スキップ
        q: 終了（進捗は保存される）
        b: 1つ戻る
    """
    print("\n=== インタラクティブレビュー ===")
    print("操作: [y]同一  [n]別物  [s]スキップ  [b]戻る  [q]終了保存")
    print("-" * 50)
    
    results = df.copy()
    if "is_same_entity" not in results.columns:
        results["is_same_entity"] = ""
    if "notes" not in results.columns:
        results["notes"] = ""
    
    # 既存の結果を読み込む
    if Path(output_path).exists():
        existing = pd.read_csv(output_path)
        if "is_same_entity" in existing.columns:
            # マージ
            for idx, row in existing.iterrows():
                mask = (results["gt_entity"] == row["gt_entity"]) & (results["pred_entity"] == row["pred_entity"])
                if mask.any() and pd.notna(row.get("is_same_entity")) and row.get("is_same_entity") != "":
                    results.loc[mask, "is_same_entity"] = row["is_same_entity"]
    
    # 未レビューの開始位置を探す
    start_idx = 0
    for i, row in results.iterrows():
        if row["is_same_entity"] in ["", None] or pd.isna(row["is_same_entity"]):
            start_idx = i
            break
    else:
        start_idx = len(results)
    
    reviewed_count = results["is_same_entity"].apply(lambda x: x in ["Y", "N"]).sum()
    print(f"進捗: {reviewed_count}/{len(results)} 件レビュー済み\n")
    
    i = start_idx
    while i < len(results):
        row = results.iloc[i]
        
        # 表示
        print(f"\n[{i+1}/{len(results)}] 類似度: {row['similarity']:.3f}")
        print(f"  GT:   '{row['gt_entity']}'")
        print(f"  Pred: '{row['pred_entity']}'")
        
        # 既存の判定があれば表示
        if row["is_same_entity"] in ["Y", "N"]:
            print(f"  現在の判定: {row['is_same_entity']}")
        
        # 入力待ち
        try:
            key = input("  >>> ").strip().lower()
        except EOFError:
            break
        
        if key == "y":
            results.iloc[i, results.columns.get_loc("is_same_entity")] = "Y"
            i += 1
        elif key == "n":
            results.iloc[i, results.columns.get_loc("is_same_entity")] = "N"
            i += 1
        elif key == "s":
            i += 1
        elif key == "b":
            if i > 0:
                i -= 1
        elif key == "q":
            break
        else:
            print("  ※ y/n/s/b/q のいずれかを入力してください")
    
    # 保存
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    reviewed_count = results["is_same_entity"].apply(lambda x: x in ["Y", "N"]).sum()
    print(f"\n保存しました: {output_path}")
    print(f"レビュー済み: {reviewed_count}/{len(results)} 件")
    
    if reviewed_count == len(results):
        print("\n全件レビュー完了！分析を実行します...")
        analyze_reviewed_csv(output_path)


def main():
    import argparse
    
    output_dir = project_root / "preference_kg/results/evaluations/gpt-4o/20260112_194507"
    similarity_csv_path = output_dir / "entity_similarity_analysis.csv"
    review_csv_path = output_dir / "entity_pairs_for_review.csv"
    
    parser = argparse.ArgumentParser(description="Entity Similarity Manual Review Tool")
    parser.add_argument("mode", nargs="?", default="review", 
                        choices=["review", "analyze"],
                        help="review: インタラクティブレビュー, analyze: 結果分析")
    parser.add_argument("--min", type=float, default=0.0,
                        help="類似度の下限 (例: 0.6)")
    parser.add_argument("--max", type=float, default=1.0,
                        help="類似度の上限 (例: 0.9)")
    parser.add_argument("--redo", action="store_true",
                        help="指定範囲の既存アノテーションをクリアして再レビュー")
    
    args = parser.parse_args()
    
    print("=== Entity Similarity Manual Review Tool ===\n")
    
    if args.mode == "analyze":
        analyze_reviewed_csv(str(review_csv_path))
    else:
        print(f"類似度データ読み込み中: {similarity_csv_path}")
        df = pd.read_csv(similarity_csv_path)
        
        # 類似度範囲でフィルタ
        if args.min > 0.0 or args.max < 1.0:
            original_count = len(df)
            df = df[(df["similarity"] >= args.min) & (df["similarity"] <= args.max)]
            print(f"フィルタ: 類似度 {args.min:.2f} - {args.max:.2f}")
            print(f"対象ペア: {len(df)} / {original_count}")
        else:
            print(f"総ペア数: {len(df)}")
        
        print(f"類似度範囲: {df['similarity'].min():.3f} - {df['similarity'].max():.3f}")
        
        # 類似度順でソート（高い順）
        df = df.sort_values("similarity", ascending=False).reset_index(drop=True)
        
        # --redo オプションで既存アノテーションをクリア
        if args.redo:
            print("\n⚠ 指定範囲の既存アノテーションをクリアします")
            if Path(review_csv_path).exists():
                existing = pd.read_csv(review_csv_path)
                # 指定範囲のアノテーションをクリア
                mask = (existing["similarity"] >= args.min) & (existing["similarity"] <= args.max)
                existing.loc[mask, "is_same_entity"] = ""
                existing.to_csv(review_csv_path, index=False, encoding="utf-8-sig")
                print(f"  クリア済み: {mask.sum()} 件")
        
        # インタラクティブレビュー開始
        interactive_review(df, str(review_csv_path))


if __name__ == "__main__":
    main()



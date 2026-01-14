"""Entity Similarity Analysis Script

GPT-4oの実験結果を分析し、意味的類似度計算の閾値妥当性を検証する。
すべてのGT-Pred Entity ペアに対して類似度を計算し、
マッチング結果と比較対象の情報を記録する。
"""

import json
import sys
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preference_kg.evaluation import split_combined_axis
from preference_kg.evaluation.semantic_similarity import (
    compute_entity_similarity,
    ENTITY_SIMILARITY_THRESHOLD,
    get_cache_info,
)


def load_experiment_results(filepath: str) -> dict:
    """実験結果を読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_entity_similarities(experiment_data: dict) -> pd.DataFrame:
    """
    すべてのGT-Predペアの意味的類似度を計算し、DataFrameとして返す。
    
    Returns:
        DataFrame with columns:
        - dialogue_id
        - gt_entity
        - pred_entity
        - similarity
        - is_match (閾値以上か)
        - gt_axis, gt_sub_axis, gt_polarity
        - pred_axis, pred_sub_axis, pred_polarity
    """
    results = []
    
    for entry in experiment_data.get("results", []):
        dialogue_id = entry["dialogue_id"]
        ground_truths = entry.get("ground_truth_annotations", [])
        extracted = entry.get("extracted_preferences", {})
        predictions = extracted.get("preferences", [])
        
        if not ground_truths or not predictions:
            continue
        
        # すべてのGT-Predペアを計算
        for gt in ground_truths:
            gt_entity = gt.get("entity", "").strip()
            if not gt_entity:
                continue
                
            for pred in predictions:
                pred_entity = pred.get("entity", "").strip()
                if not pred_entity:
                    continue
                
                # 類似度計算
                similarity = compute_entity_similarity(gt_entity, pred_entity)
                
                # axisを分割
                pred_axis, pred_sub_axis = split_combined_axis(pred.get("combined_axis", ""))
                
                results.append({
                    "dialogue_id": dialogue_id,
                    "gt_entity": gt_entity,
                    "pred_entity": pred_entity,
                    "similarity": similarity,
                    "is_match": similarity >= ENTITY_SIMILARITY_THRESHOLD,
                    "gt_axis": gt.get("axis", ""),
                    "gt_sub_axis": gt.get("sub_axis", ""),
                    "gt_polarity": gt.get("polarity", ""),
                    "pred_axis": pred_axis,
                    "pred_sub_axis": pred_sub_axis,
                    "pred_polarity": pred.get("polarity", ""),
                })
    
    return pd.DataFrame(results)


def analyze_best_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    各GTエンティティに対して最も類似度が高いPredを特定する。
    
    Returns:
        DataFrame with best match for each (dialogue_id, gt_entity) pair
    """
    # 各GTエンティティに対して最大類似度のペアを取得
    idx = df.groupby(["dialogue_id", "gt_entity"])["similarity"].idxmax()
    best_matches = df.loc[idx].copy()
    best_matches["is_best_match"] = True
    
    return best_matches


def plot_similarity_distribution(df: pd.DataFrame, output_path: str = None):
    """類似度分布をヒストグラムで可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 全ペアの分布
    ax1 = axes[0]
    ax1.hist(df["similarity"], bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=ENTITY_SIMILARITY_THRESHOLD, color='red', linestyle='--', 
                label=f'Threshold ({ENTITY_SIMILARITY_THRESHOLD})')
    ax1.set_xlabel("Semantic Similarity")
    ax1.set_ylabel("Count")
    ax1.set_title("All GT-Pred Entity Pairs")
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # ベストマッチのみ
    best_df = analyze_best_matches(df)
    ax2 = axes[1]
    ax2.hist(best_df["similarity"], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(x=ENTITY_SIMILARITY_THRESHOLD, color='red', linestyle='--',
                label=f'Threshold ({ENTITY_SIMILARITY_THRESHOLD})')
    ax2.set_xlabel("Semantic Similarity")
    ax2.set_ylabel("Count")
    ax2.set_title("Best Match per GT Entity")
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    
    plt.show()


def plot_threshold_analysis(df: pd.DataFrame, output_path: str = None):
    """
    異なる閾値での精度・再現率をプロットし、最適閾値を分析
    """
    best_df = analyze_best_matches(df)
    
    thresholds = np.arange(0.3, 1.0, 0.05)
    tp_counts = []
    fn_counts = []
    precisions = []
    recalls = []
    
    total_gt = len(best_df)
    
    for thresh in thresholds:
        # 閾値以上のベストマッチをTP、未満をFN
        matches = best_df[best_df["similarity"] >= thresh]
        tp = len(matches)
        fn = total_gt - tp
        
        # すべての閾値以上ペアをカウント（FP推定用）
        all_above = len(df[df["similarity"] >= thresh])
        
        precision = tp / all_above if all_above > 0 else 0
        recall = tp / total_gt if total_gt > 0 else 0
        
        tp_counts.append(tp)
        fn_counts.append(fn)
        precisions.append(precision)
        recalls.append(recall)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # TP/FN counts
    ax1 = axes[0]
    ax1.plot(thresholds, tp_counts, 'g-o', label='True Positives', markersize=4)
    ax1.plot(thresholds, fn_counts, 'r-o', label='False Negatives', markersize=4)
    ax1.axvline(x=ENTITY_SIMILARITY_THRESHOLD, color='blue', linestyle='--',
                label=f'Current Threshold ({ENTITY_SIMILARITY_THRESHOLD})')
    ax1.set_xlabel("Similarity Threshold")
    ax1.set_ylabel("Count")
    ax1.set_title("TP/FN vs Threshold")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Precision/Recall curve
    ax2 = axes[1]
    ax2.plot(thresholds, precisions, 'b-o', label='Precision', markersize=4)
    ax2.plot(thresholds, recalls, 'orange', marker='o', label='Recall', markersize=4)
    
    # F1 score
    f1_scores = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precisions, recalls)]
    ax2.plot(thresholds, f1_scores, 'g-o', label='F1', markersize=4)
    
    ax2.axvline(x=ENTITY_SIMILARITY_THRESHOLD, color='red', linestyle='--',
                label=f'Current Threshold ({ENTITY_SIMILARITY_THRESHOLD})')
    ax2.set_xlabel("Similarity Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision/Recall/F1 vs Threshold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    
    plt.show()
    
    # 最適閾値を計算
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"\n=== Optimal Threshold Analysis ===")
    print(f"Best F1 Score: {f1_scores[best_idx]:.4f} at threshold {best_threshold:.2f}")
    print(f"Current Threshold: {ENTITY_SIMILARITY_THRESHOLD}")
    
    return thresholds, f1_scores


def print_similarity_statistics(df: pd.DataFrame):
    """類似度の統計情報を表示"""
    print("=== Similarity Statistics ===")
    print(f"Total GT-Pred pairs: {len(df)}")
    print(f"\nAll pairs:")
    print(f"  Mean: {df['similarity'].mean():.4f}")
    print(f"  Std:  {df['similarity'].std():.4f}")
    print(f"  Min:  {df['similarity'].min():.4f}")
    print(f"  Max:  {df['similarity'].max():.4f}")
    
    # 閾値に基づくカウント
    n_match = len(df[df["is_match"]])
    n_no_match = len(df) - n_match
    print(f"\nThreshold {ENTITY_SIMILARITY_THRESHOLD}:")
    print(f"  Matching pairs: {n_match} ({n_match/len(df)*100:.1f}%)")
    print(f"  Non-matching:   {n_no_match} ({n_no_match/len(df)*100:.1f}%)")
    
    # ベストマッチの統計
    best_df = analyze_best_matches(df)
    print(f"\nBest matches per GT entity:")
    print(f"  Total GT entities: {len(best_df)}")
    print(f"  Mean similarity: {best_df['similarity'].mean():.4f}")
    n_best_match = len(best_df[best_df["similarity"] >= ENTITY_SIMILARITY_THRESHOLD])
    print(f"  Above threshold: {n_best_match} ({n_best_match/len(best_df)*100:.1f}%)")
    
    # エンベディングキャッシュ情報
    cache_info = get_cache_info()
    print(f"\nEmbedding cache: {cache_info}")


def show_example_pairs(df: pd.DataFrame, n: int = 20):
    """類似度の例を表示（閾値付近、高い、低い）"""
    print("\n=== Example Entity Pairs ===")
    
    # 閾値付近（0.6-0.8）
    near_threshold = df[(df["similarity"] >= 0.6) & (df["similarity"] < 0.85)].sort_values("similarity", ascending=False)
    print(f"\n--- Near Threshold (0.6-0.85): {len(near_threshold)} pairs ---")
    for _, row in near_threshold.head(n).iterrows():
        match_str = "✓" if row["is_match"] else "✗"
        print(f"  {match_str} {row['similarity']:.3f}: '{row['gt_entity']}' <-> '{row['pred_entity']}'")
    
    # 高い類似度（0.85+、完全一致以外）
    high_sim = df[(df["similarity"] >= 0.85) & (df["similarity"] < 1.0)].sort_values("similarity", ascending=False)
    print(f"\n--- High Similarity (0.85-1.0): {len(high_sim)} pairs ---")
    for _, row in high_sim.head(n).iterrows():
        print(f"  ✓ {row['similarity']:.3f}: '{row['gt_entity']}' <-> '{row['pred_entity']}'")
    
    # 完全一致
    exact = df[df["similarity"] == 1.0]
    print(f"\n--- Exact Match (1.0): {len(exact)} pairs ---")
    
    # 低い類似度で見逃している可能性があるペア
    low_sim = df[(df["similarity"] >= 0.5) & (df["similarity"] < ENTITY_SIMILARITY_THRESHOLD)]
    low_sim = low_sim.sort_values("similarity", ascending=False)
    print(f"\n--- Missed matches (0.5 to {ENTITY_SIMILARITY_THRESHOLD}): {len(low_sim)} pairs ---")
    for _, row in low_sim.head(n).iterrows():
        print(f"  ✗ {row['similarity']:.3f}: '{row['gt_entity']}' <-> '{row['pred_entity']}'")


def main():
    # 実験結果のパス
    experiment_path = project_root / "preference_kg/results/experiments/gpt-4o/experiment_results_20260112_194507.json"
    output_dir = project_root / "preference_kg/results/evaluations/gpt-4o/20260112_194507"
    
    print(f"Loading: {experiment_path}")
    experiment_data = load_experiment_results(str(experiment_path))
    
    print(f"Analyzing {len(experiment_data.get('results', []))} dialogues...")
    
    # 類似度分析
    df = analyze_entity_similarities(experiment_data)
    
    # 統計表示
    print_similarity_statistics(df)
    
    # 例を表示
    show_example_pairs(df)
    
    # 結果を保存
    output_csv = output_dir / "entity_similarity_analysis.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    
    # 可視化
    plot_similarity_distribution(df, str(output_dir / "similarity_distribution.png"))
    plot_threshold_analysis(df, str(output_dir / "threshold_analysis.png"))
    
    return df


if __name__ == "__main__":
    df = main()

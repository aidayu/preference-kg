"""マッチングモジュール

GT（正解）とPred（予測）の嗜好オブジェクト間の最適マッチングを行う。
Hungarian algorithmを使用して、スコア合計が最大となる1対1マッチングを計算する。
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .normalizers import normalize_sub_axis, normalize_context, normalize_intensity


def compute_matching_score(gt: dict, pred: dict) -> float:
    """
    GTとPred嗜好オブジェクト間の類似度スコアを計算する。
    
    スコア配分:
    - Entity部分一致: 1.0
    - Axis一致: 1.0
    - Sub-axis一致: 1.0
    - Polarity一致: 1.0
    - Intensity一致: 0.5
    - Context重複あり: 0.5
    
    Args:
        gt: 正解の嗜好オブジェクト
        pred: 予測の嗜好オブジェクト
    
    Returns:
        0.0-5.0のスコア（高いほど良いマッチ）
    """
    score = 0.0
    
    # Entity部分一致チェック（必須条件）
    gt_entity = gt.get("entity", "").lower().strip()
    pred_entity = pred.get("entity", "").lower().strip()
    
    if not gt_entity or not pred_entity:
        return 0.0
    
    if gt_entity in pred_entity or pred_entity in gt_entity:
        score += 1.0
    else:
        # Entityが一致しない場合はマッチ不可
        return 0.0
    
    # Axis一致
    if gt.get("axis") == pred.get("axis"):
        score += 1.0
    
    # Sub-axis一致（正規化して比較）
    gt_sub_axis = normalize_sub_axis(gt.get("sub_axis"))
    pred_sub_axis = normalize_sub_axis(pred.get("sub_axis"))
    if gt_sub_axis == pred_sub_axis:
        score += 1.0
    
    # Polarity一致
    if gt.get("polarity") == pred.get("polarity"):
        score += 1.0
    
    # Intensity一致（正規化して比較）
    gt_intensity = normalize_intensity(gt.get("intensity"))
    pred_intensity = normalize_intensity(pred.get("intensity"))
    if gt_intensity and pred_intensity and gt_intensity == pred_intensity:
        score += 0.5
    
    # Context重複チェック
    gt_context = normalize_context(gt.get("context", []))
    pred_context = normalize_context(pred.get("context_tags", []))
    
    if len(gt_context) == 0 and len(pred_context) == 0:
        score += 0.5
    elif len(gt_context) > 0 and len(pred_context) > 0:
        if len(gt_context & pred_context) > 0:
            score += 0.5
    
    return score


def find_optimal_matching(
    ground_truths: list[dict],
    predictions: list[dict],
) -> list[tuple[int, int | None, dict | None, float]]:
    """
    Hungarian algorithmを使用してGTとPred間の最適マッチングを見つける。
    
    Args:
        ground_truths: 正解の嗜好オブジェクトリスト
        predictions: 予測の嗜好オブジェクトリスト
    
    Returns:
        マッチング結果のリスト: [(gt_idx, pred_idx or None, pred or None, score), ...]
        pred_idx=None の場合はマッチなし（MISSING）
    """
    n_gt = len(ground_truths)
    n_pred = len(predictions)
    
    if n_gt == 0:
        return []
    
    if n_pred == 0:
        # 予測がない場合、すべてのGTはマッチなし
        return [(i, None, None, 0.0) for i in range(n_gt)]
    
    # スコア行列を作成（GTがrow, Predがcolumn）
    score_matrix = np.zeros((n_gt, n_pred))
    
    for i, gt in enumerate(ground_truths):
        for j, pred in enumerate(predictions):
            score_matrix[i, j] = compute_matching_score(gt, pred)
    
    # Hungarian algorithmはコスト最小化なので、スコアを負にする
    cost_matrix = -score_matrix
    
    # 最適割り当てを計算
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 結果を構築
    results = []
    matched_gt_indices = set()
    
    for gt_idx, pred_idx in zip(row_ind, col_ind):
        score = score_matrix[gt_idx, pred_idx]
        
        if score > 0:  # entity一致がある場合のみマッチとみなす
            results.append((gt_idx, pred_idx, predictions[pred_idx], score))
            matched_gt_indices.add(gt_idx)
        else:
            results.append((gt_idx, None, None, 0.0))
            matched_gt_indices.add(gt_idx)
    
    # マッチングに含まれなかったGTを追加
    for i in range(n_gt):
        if i not in matched_gt_indices:
            results.append((i, None, None, 0.0))
    
    # GT indexでソート
    results.sort(key=lambda x: x[0])
    
    return results


def get_unmatched_predictions(
    predictions: list[dict],
    matching_results: list[tuple[int, int | None, dict | None, float]],
) -> list[tuple[int, dict]]:
    """
    マッチングされなかった予測を取得する（False Positive検出用）。
    
    Args:
        predictions: 予測の嗜好オブジェクトリスト
        matching_results: find_optimal_matchingの結果
    
    Returns:
        マッチされなかった予測: [(pred_idx, pred), ...]
    """
    matched_pred_indices = {
        result[1] for result in matching_results if result[1] is not None
    }
    
    return [
        (i, pred) for i, pred in enumerate(predictions)
        if i not in matched_pred_indices
    ]

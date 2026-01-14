"""評価モジュール

嗜好抽出結果の評価に関する機能を提供
- normalizers: 正規化関数
- metrics: 評価指標計算
- matching: GT-Pred間の最適マッチング
- dialogue_evaluator: 対話単位評価
- aggregators: Micro/Macro/Weighted F1集計
- reporter: 結果出力
"""

from .normalizers import (
    split_combined_axis,
    normalize_sub_axis,
    normalize_context,
    normalize_intensity,
)
from .metrics import (
    augment_with_ancestors,
    compute_hierarchical_metrics,
    compute_f1,
    compute_basic_metrics,
)
from .matching import (
    compute_matching_score,
    find_optimal_matching,
    get_unmatched_predictions,
)
from .dialogue_evaluator import (
    DialogueResult,
    evaluate_dialogue,
)
from .aggregators import (
    AggregatedMetrics,
    aggregate_metrics,
    aggregate_hierarchical_metrics,
    aggregate_all_metrics,
)
from .reporter import (
    save_evaluation_results,
    print_evaluation_summary,
)

__all__ = [
    # normalizers
    "split_combined_axis",
    "normalize_sub_axis", 
    "normalize_context",
    "normalize_intensity",
    # metrics
    "augment_with_ancestors",
    "compute_hierarchical_metrics",
    "compute_f1",
    "compute_basic_metrics",
    # matching
    "compute_matching_score",
    "find_optimal_matching",
    "get_unmatched_predictions",
    # dialogue_evaluator
    "DialogueResult",
    "evaluate_dialogue",
    # aggregators
    "AggregatedMetrics",
    "aggregate_metrics",
    "aggregate_hierarchical_metrics",
    "aggregate_all_metrics",
    # reporter
    "save_evaluation_results",
    "print_evaluation_summary",
]

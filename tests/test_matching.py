"""マッチングモジュールのテスト

Hungarian algorithmを使用した最適マッチングのテスト
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preference_kg"))

from evaluation.matching import compute_matching_score, find_optimal_matching, get_unmatched_predictions


class TestComputeMatchingScore:
    """compute_matching_score関数のテスト"""
    
    def test_exact_match(self):
        """完全一致の場合"""
        gt = {
            "entity": "ラーメン",
            "axis": "liking",
            "sub_axis": "aesthetic/sensory",
            "polarity": "positive",
            "intensity": "high",
            "context": [],
        }
        pred = {
            "entity": "ラーメン",
            "axis": "liking",
            "sub_axis": "aesthetic_sensory",
            "polarity": "positive",
            "intensity": "high",
            "context_tags": [],
        }
        score = compute_matching_score(gt, pred)
        assert score == 5.0  # 1 + 1 + 1 + 1 + 0.5 + 0.5
    
    def test_partial_entity_match(self):
        """エンティティの部分一致"""
        gt = {"entity": "ラーメン", "axis": "liking", "polarity": "positive"}
        pred = {"entity": "豚骨ラーメン", "axis": "liking", "polarity": "positive"}
        score = compute_matching_score(gt, pred)
        assert score >= 1.0  # entity match
    
    def test_entity_no_match(self):
        """エンティティが一致しない場合"""
        gt = {"entity": "ラーメン", "axis": "liking"}
        pred = {"entity": "カレー", "axis": "liking"}
        score = compute_matching_score(gt, pred)
        assert score == 0.0
    
    def test_axis_mismatch(self):
        """axisが異なる場合"""
        gt = {"entity": "ラーメン", "axis": "liking", "polarity": "positive"}
        pred = {"entity": "ラーメン", "axis": "wanting", "polarity": "positive"}
        score = compute_matching_score(gt, pred)
        # entity (1) + sub_axis None match (1) + polarity (1) + context empty (0.5) = 3.5
        assert score == 3.5


class TestFindOptimalMatching:
    """find_optimal_matching関数のテスト"""
    
    def test_simple_matching(self):
        """単純な1対1マッチング"""
        gts = [{"entity": "ラーメン", "axis": "liking", "polarity": "positive"}]
        preds = [{"entity": "ラーメン", "axis": "liking", "polarity": "positive"}]
        
        results = find_optimal_matching(gts, preds)
        
        assert len(results) == 1
        assert results[0][0] == 0  # gt_idx
        assert results[0][1] == 0  # pred_idx
        assert results[0][2] is not None  # match
        assert results[0][3] > 0  # score
    
    def test_same_entity_different_axes(self):
        """同一エンティティで異なるaxisの場合 - 順序問題のテスト"""
        gts = [
            {"entity": "ラーメン", "axis": "liking", "sub_axis": "aesthetic/sensory", "polarity": "positive"},
            {"entity": "ラーメン", "axis": "wanting", "sub_axis": "goal", "polarity": "positive"},
        ]
        # predの順序がgtと逆
        preds = [
            {"entity": "ラーメン", "axis": "wanting", "sub_axis": "goal", "polarity": "positive"},
            {"entity": "ラーメン", "axis": "liking", "sub_axis": "aesthetic_sensory", "polarity": "positive"},
        ]
        
        results = find_optimal_matching(gts, preds)
        
        # 最適マッチングにより、正しいペアリングがされるべき
        assert len(results) == 2
        
        # GT0 (liking) は Pred1 (liking) とマッチすべき
        gt0_result = results[0]
        assert gt0_result[0] == 0
        assert gt0_result[2]["axis"] == "liking"
        
        # GT1 (wanting) は Pred0 (wanting) とマッチすべき
        gt1_result = results[1]
        assert gt1_result[0] == 1
        assert gt1_result[2]["axis"] == "wanting"
    
    def test_more_gts_than_preds(self):
        """GTがPredより多い場合"""
        gts = [
            {"entity": "ラーメン", "axis": "liking", "polarity": "positive"},
            {"entity": "カレー", "axis": "liking", "polarity": "positive"},
        ]
        preds = [
            {"entity": "ラーメン", "axis": "liking", "polarity": "positive"},
        ]
        
        results = find_optimal_matching(gts, preds)
        
        assert len(results) == 2
        # ラーメンはマッチ
        assert any(r[2] is not None for r in results)
        # カレーはマッチなし
        assert any(r[2] is None for r in results)
    
    def test_empty_predictions(self):
        """予測が空の場合"""
        gts = [{"entity": "ラーメン", "axis": "liking", "polarity": "positive"}]
        preds = []
        
        results = find_optimal_matching(gts, preds)
        
        assert len(results) == 1
        assert results[0][1] is None
        assert results[0][2] is None
        assert results[0][3] == 0.0
    
    def test_empty_ground_truths(self):
        """GTが空の場合"""
        gts = []
        preds = [{"entity": "ラーメン", "axis": "liking", "polarity": "positive"}]
        
        results = find_optimal_matching(gts, preds)
        
        assert len(results) == 0


class TestGetUnmatchedPredictions:
    """get_unmatched_predictions関数のテスト"""
    
    def test_gets_unmatched(self):
        """マッチしなかった予測を取得"""
        preds = [
            {"entity": "ラーメン", "axis": "liking"},
            {"entity": "カレー", "axis": "liking"},
        ]
        matching_results = [
            (0, 0, preds[0], 3.0),  # ラーメンがマッチ
        ]
        
        unmatched = get_unmatched_predictions(preds, matching_results)
        
        assert len(unmatched) == 1
        assert unmatched[0][0] == 1  # pred_idx
        assert unmatched[0][1]["entity"] == "カレー"


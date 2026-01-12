"""Multi-step preference extraction experiment.

This script runs the 3-step preference extraction pipeline on the annotated dataset
and compares results with ground truth annotations.
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from tqdm import tqdm

from extractors import MultiStepPreferenceExtractor

# 環境変数読み込み
load_dotenv()

# パス設定
DATASET_PATH = "/home/y-aida/Programs/preference-kg/data/interim/dailydialog_annotated_20260108_fined2.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

# Few-shot用のdialogue_id（比較用に同じIDを使用）
FEW_SHOT_IDS = [0, 18, 46]


def load_dataset():
    """データセットを読み込む"""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(debug_mode: bool = False, limit: int | None = None):
    """
    Multi-step抽出実験を実行する

    Args:
        debug_mode: Trueの場合、各ステップの中間結果も保存
        limit: 処理する対話数の上限（Noneで全件処理）
    """
    print("=== Multi-Step 嗜好抽出実験開始 ===")
    print(f"データセット: {DATASET_PATH}")
    print(f"Few-shot dialogue_ids: {FEW_SHOT_IDS}")

    # データセット読み込み
    print("\n[1/4] データセット読み込み中...")
    dataset = load_dataset()
    print(f"総対話数: {len(dataset)}")

    # タイムスタンプを先に生成（ステップ出力ディレクトリ用）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ステップごとの出力ディレクトリ
    step_output_dir = os.path.join(OUTPUT_DIR, f"multistep_steps_{timestamp}")
    os.makedirs(step_output_dir, exist_ok=True)
    print(f"ステップ出力ディレクトリ: {step_output_dir}")
    
    # パイプライン初期化
    print("\n[2/4] Multi-step パイプライン初期化中...")
    extractor = MultiStepPreferenceExtractor(model="gpt-4o-mini", output_dir=step_output_dir)

    # テスト対象の対話を取得（Few-shot例を除く）
    test_dialogues = [d for d in dataset if d["dialogue_id"] not in FEW_SHOT_IDS]

    if limit:
        test_dialogues = test_dialogues[:limit]

    print(f"テスト対話数: {len(test_dialogues)}")

    # 抽出実行
    print("\n[3/4] Multi-step 嗜好抽出実行中...")
    results = []

    for dialogue_data in tqdm(test_dialogues, desc="Processing dialogues"):
        dialogue_id = dialogue_data["dialogue_id"]
        dialogue_text = dialogue_data["original_dialogue"]

        # 抽出実行
        if debug_mode:
            extraction_result = extractor.extract_with_debug(dialogue_text, dialogue_id)
        else:
            extraction_result = extractor.extract(dialogue_text, dialogue_id)

        # 結果構築
        result_with_annotation = {
            "dialogue_id": dialogue_id,
            "original_dialogue": dialogue_text,
            "translated_dialogue": dialogue_data.get("translated_dialogue", ""),
            "ground_truth_annotations": dialogue_data.get("annotations", []),
            "extracted_preferences": extraction_result,
        }

        results.append(result_with_annotation)

    # 結果保存
    print("\n[4/4] 結果保存中...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    suffix = "_debug" if debug_mode else ""
    output_path = os.path.join(OUTPUT_DIR, f"multistep_results_{timestamp}{suffix}.json")

    experiment_metadata = {
        "experiment_info": {
            "timestamp": timestamp,
            "dataset_path": DATASET_PATH,
            "few_shot_ids": FEW_SHOT_IDS,
            "total_test_dialogues": len(test_dialogues),
            "model": "gpt-4o-mini",
            "method": "multi-step (3 steps)",
            "debug_mode": debug_mode,
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 実験完了！")
    print(f"結果保存先: {output_path}")
    print(f"処理対話数: {len(results)}")

    # 簡易統計
    total_extracted = sum(
        len(r["extracted_preferences"].get("preferences", []))
        for r in results
    )
    total_ground_truth = sum(len(r["ground_truth_annotations"]) for r in results)
    print(f"\n統計:")
    print(f"  - 抽出された嗜好数: {total_extracted}")
    print(f"  - Ground truth嗜好数: {total_ground_truth}")

    return experiment_metadata


def run_single_test(dialogue: str):
    """単一対話でテスト（デバッグ用）"""
    print("=== Single Dialogue Test ===")
    print(f"Input: {dialogue[:100]}...")

    extractor = MultiStepPreferenceExtractor(model="gpt-4o-mini")
    result = extractor.extract_with_debug(dialogue)

    print("\n--- Step 1: Entities ---")
    for e in result["step1_entities"]:
        print(f"  - {e['entity']}: {e['reasoning'][:50]}...")

    print("\n--- Step 2: Axes ---")
    for a in result["step2_axes"]:
        print(f"  - {a['entity']}: {[x['combined_axis'] for x in a['axes']]}")

    print("\n--- Step 3: Preferences ---")
    for p in result["final_preferences"]:
        print(f"  - {p['entity']} | {p['combined_axis']} | {p['polarity']} | {p['intensity']}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-step preference extraction experiment")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dialogues")
    parser.add_argument("--test", type=str, default=None, help="Test with single dialogue")

    args = parser.parse_args()

    if args.test:
        run_single_test(args.test)
    else:
        run_experiment(debug_mode=args.debug, limit=args.limit)

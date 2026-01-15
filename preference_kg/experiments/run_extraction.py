import os
import json
from openai import OpenAI
import openai
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

# 1. APIキーの設定
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# 2. ファイルパスの設定
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../../prompt_templates")
DATASET_PATH = "/home/y-aida/Programs/preference-kg/data/interim/dailydialog_annotated_20260108_fined2.json"
FEW_SHOT_PROMPT_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "few_shot_extract_template_cot.txt")
SCHEMA_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "schema_template_cot.json")
AXIS_DEFINITIONS_PATH = os.path.join(TEMPLATE_DIR, "axis_definitions_en.json")

# モデル設定
MODEL_NAME = "gemma3:27b"

# Few-shot用のdialogue_id
FEW_SHOT_IDS = [0, 18, 46]

# 3. データセットの読み込み
def load_dataset():
    """データセットを読み込む"""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# 4. テンプレートの読み込み
def load_few_shot_prompt_template():
    """Few-shotプロンプトテンプレートを読み込む"""
    with open(FEW_SHOT_PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

# def load_axis_definitions():
#     """軸定義を読み込む"""
#     with open(AXIS_DEFINITIONS_PATH, "r", encoding="utf-8") as f:
#         return json.load(f)

def load_schema_template():
    """スキーマテンプレートを読み込む"""
    with open(SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# 5. Few-shot例の作成
def create_few_shot_examples(dataset, few_shot_ids):
    """
    データセットからFew-shot例を作成する
    
    Args:
        dataset: 全データセット
        few_shot_ids: Few-shot用のdialogue_idリスト
    
    Returns:
        Few-shot例のテキスト
    """
    examples_text = ""
    
    for idx, dialogue_id in enumerate(few_shot_ids, 1):
        # dialogue_idに対応するデータを取得
        dialogue_data = next((d for d in dataset if d["dialogue_id"] == dialogue_id), None)
        
        if dialogue_data is None:
            print(f"Warning: dialogue_id {dialogue_id} not found in dataset")
            continue
        
        # 対話テキスト
        dialogue_text = dialogue_data["original_dialogue"]
        
        # アノテーションから抽出結果を構築
        preferences = []
        for ann in dialogue_data.get("annotations", []):
            # combined_axisの形式に変換（例: "liking__identification"）
            axis = ann.get("axis", "")
            sub_axis = ann.get("sub_axis", "")
            combined_axis = f"{axis}__{sub_axis}"
            
            # Note: explicitnessフィールドはfew-shot例から除外する
            # （モデルにexplicitnessの概念を学習させないため）
            preference = {
                "reasoning": f"Entity '{ann.get('entity', '')}' is mentioned with {ann.get('polarity', '')} sentiment",
                "combined_axis": combined_axis,
                "entity": ann.get("entity", ""),
                "original_mention": ann.get("entity", ""),
                "context_tags": ann.get("context", []) if ann.get("context") != ["None"] else [],
                "polarity": ann.get("polarity", "neutral"),
                "intensity": ann.get("intensity", "medium")
                # explicitnessは意図的に除外
            }
            preferences.append(preference)
        
        extraction = {
            "dialogue_id": dialogue_id,
            "user_id": "user",
            "preferences": preferences
        }
        
        # Few-shot例のフォーマット
        examples_text += f"\n### Example {idx}:\n"
        examples_text += f"**Dialogue:**\n{dialogue_text}\n\n"
        examples_text += f"**Extraction:**\n{json.dumps(extraction, indent=2, ensure_ascii=False)}\n"
    
    return examples_text

# 6. 嗜好抽出関数
def extract_preferences_with_few_shot(dialogue_text, dialogue_id, system_prompt, schema):
    """
    Few-shotプロンプトを使用して対話から嗜好を抽出する
    
    Args:
        dialogue_text: 対話テキスト
        dialogue_id: 対話ID
        system_prompt: システムプロンプト
        schema: JSONスキーマ
    
    Returns:
        抽出結果（JSON）
    """
    try:
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="gemma3:27b"
        )

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": f"Extract preference from this dialogue.\n\n{dialogue_text}"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": schema
            },
            temperature=0
        )
        
        result = json.loads(completion.choices[0].message.content)
        return result
    
    except Exception as e:
        print(f"Error extracting preferences for dialogue_id {dialogue_id}: {e}")
        return {
            "dialogue_id": dialogue_id,
            "user_id": "user",
            "preferences": [],
            "error": str(e)
        }

# 7. メイン実験関数
def run_experiment():
    """実験を実行する"""
    print("=== 嗜好抽出実験開始 ===")
    print(f"データセット: {DATASET_PATH}")
    print(f"Few-shot dialogue_ids: {FEW_SHOT_IDS}")
    
    # データセット読み込み
    print("\n[1/5] データセット読み込み中...")
    dataset = load_dataset()
    print(f"総対話数: {len(dataset)}")
    
    # Few-shot例の作成
    print("\n[2/5] Few-shot例作成中...")
    few_shot_examples_text = create_few_shot_examples(dataset, FEW_SHOT_IDS)
    
    # プロンプトテンプレート読み込み
    print("\n[3/5] プロンプトテンプレート読み込み中...")
    prompt_template = load_few_shot_prompt_template()
    system_prompt = prompt_template.replace("{few_shot_examples}", few_shot_examples_text)
    
    # スキーマ読み込み
    schema = load_schema_template()
    
    # テスト対象の対話を取得（Few-shot例を除く）
    test_dialogues = [d for d in dataset if d["dialogue_id"] not in FEW_SHOT_IDS]
    print(f"テスト対話数: {len(test_dialogues)}")
    
    # 嗜好抽出実行
    print("\n[4/5] 嗜好抽出実行中...")
    results = []
    
    for dialogue_data in tqdm(test_dialogues, desc="Processing dialogues"):
        dialogue_id = dialogue_data["dialogue_id"]
        dialogue_text = dialogue_data["original_dialogue"]
        
        # 抽出実行
        extraction_result = extract_preferences_with_few_shot(
            dialogue_text, 
            dialogue_id, 
            system_prompt, 
            schema
        )
        
        # 元のアノテーションも保持
        result_with_annotation = {
            "dialogue_id": dialogue_id,
            "original_dialogue": dialogue_text,
            "translated_dialogue": dialogue_data.get("translated_dialogue", ""),
            "ground_truth_annotations": dialogue_data.get("annotations", []),
            "extracted_preferences": extraction_result
        }
        
        results.append(result_with_annotation)
    
    # 結果保存
    print("\n[5/5] 結果保存中...")
    output_dir = os.path.join(os.path.dirname(__file__), f"../results/experiments/{MODEL_NAME}")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"experiment_results_{timestamp}.json")
    
    experiment_metadata = {
        "experiment_info": {
            "timestamp": timestamp,
            "dataset_path": DATASET_PATH,
            "few_shot_ids": FEW_SHOT_IDS,
            "total_test_dialogues": len(test_dialogues),
            "model": MODEL_NAME,
            "prompt_template": FEW_SHOT_PROMPT_TEMPLATE_PATH,
            "schema_template": SCHEMA_TEMPLATE_PATH,
            "axis_definitions": AXIS_DEFINITIONS_PATH
        },
        "results": results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 実験完了！")
    print(f"結果保存先: {output_path}")
    print(f"処理対話数: {len(results)}")
    
    # 簡易統計
    total_extracted = sum(len(r["extracted_preferences"].get("preferences", [])) for r in results)
    total_ground_truth = sum(len(r["ground_truth_annotations"]) for r in results)
    print(f"\n統計:")
    print(f"  - 抽出された嗜好数: {total_extracted}")
    print(f"  - Ground truth嗜好数: {total_ground_truth}")
    
    return experiment_metadata

if __name__ == "__main__":
    run_experiment()

import os
import json
import openai
from dotenv import load_dotenv

# 1. APIキーの設定
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# 2. テンプレートファイルのパス
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "prompt_templates")
PROMPT_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "extract_template.txt")
FEW_SHOT_PROMPT_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "few_shot_extract_template.txt")
SCHEMA_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "schema_template_context.json")
AXIS_DEFINITIONS_PATH = os.path.join(TEMPLATE_DIR, "axis_definitions.json")
FEW_SHOT_EXAMPLES_PATH = os.path.join(TEMPLATE_DIR, "few_shot_examples.json")

# 3. テンプレートの読み込み
def load_prompt_template():
    """プロンプトテンプレートを読み込む"""
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_few_shot_prompt_template():
    """Few-shotプロンプトテンプレートを読み込む"""
    with open(FEW_SHOT_PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_axis_definitions():
    """軸定義を読み込む"""
    with open(AXIS_DEFINITIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_few_shot_examples():
    """Few-shot例を読み込む"""
    with open(FEW_SHOT_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_schema_template():
    """スキーマテンプレートを読み込み、軸定義を埋め込む"""
    with open(SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    # 軸定義を読み込む
    axis_defs = load_axis_definitions()
    
    # スキーマ内の各軸の定義を動的に更新
    items = schema["schema"]["properties"]["preferences"]["items"]["anyOf"]
    
    # Liking
    liking_desc = "\n".join([f"- {k}: {v}" for k, v in axis_defs["liking"].items()])
    items[0]["properties"]["sub_axis"]["description"] = f"Must be one of:\n{liking_desc}"
    
    # Wanting
    wanting_desc = "\n".join([f"- {k}: {v}" for k, v in axis_defs["wanting"].items()])
    items[1]["properties"]["sub_axis"]["description"] = f"Must be one of:\n{wanting_desc}"
    
    # Need
    need_desc = "\n".join([f"- {k}: {v}" for k, v in axis_defs["need"].items()])
    items[2]["properties"]["sub_axis"]["description"] = f"Must be one of:\n{need_desc}"
    
    return schema

# テンプレートの読み込み
SYSTEM_PROMPT = load_prompt_template()
FEW_SHOT_SYSTEM_PROMPT_TEMPLATE = load_few_shot_prompt_template()
SCHEMA = load_schema_template()

def zero_shot_extract_preferences(dialogue_text: str):
    """対話からユーザーの嗜好を抽出する（Zero-shot）"""
    print("--- Zero-shot抽出開始 ---")
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",  # Structured Outputs対応モデル
        messages=[
            {
                "role": "system", 
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": f"Extract preference from this dialogue.\n\n{dialogue_text}"
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": SCHEMA
        }
    )
    
    # 文字列として返ってくるJSONをパースする
    result = json.loads(completion.choices[0].message.content)
    return result

def few_shot_extract_preferences(dialogue_text: str):
    """対話からユーザーの嗜好を抽出する（Few-shot）"""
    print("--- Few-shot抽出開始 ---")
    
    # Few-shot例を読み込んで整形
    examples = load_few_shot_examples()
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"\n### Example {i}:\n"
        examples_text += f"**Dialogue:**\n{ex['dialogue']}\n\n"
        examples_text += f"**Extraction:**\n{json.dumps(ex['extraction'], indent=2, ensure_ascii=False)}\n"
    
    # プロンプトにFew-shot例を埋め込む
    system_prompt = FEW_SHOT_SYSTEM_PROMPT_TEMPLATE.replace("{few_shot_examples}", examples_text)
    
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
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
            "json_schema": SCHEMA
        }
    )
    
    result = json.loads(completion.choices[0].message.content)
    return result

# --- テスト実行用データ ---
dummy_dialogue = """
System: こんにちは。今日はどんな気分ですか？
User: 実は最近、新しいコーヒーメーカーが欲しくて探してるんだよね。(Turn 1)
System: おお、いいですね。こだわりはありますか？
User: とにかくデザインがおしゃれなやつが好きだな。味も大事だけど、見た目でテンション上がるやつがいい。(Turn 2)
"""

if __name__ == "__main__":
    # Zero-shot実行
    print("=== Zero-shot Extraction ===")
    output_zeroshot = zero_shot_extract_preferences(dummy_dialogue)
    
    # Few-shot実行
    print("\n=== Few-shot Extraction ===")
    output_fewshot = few_shot_extract_preferences(dummy_dialogue)
    
    # JSONファイルとして保存
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Zero-shot結果
    zeroshot_path = os.path.join(output_dir, "preference_extraction_zeroshot_result.json")
    with open(zeroshot_path, "w", encoding="utf-8") as f:
        json.dump(output_zeroshot, f, indent=2, ensure_ascii=False)
    print(f"\nZero-shot結果を保存: {zeroshot_path}")
    
    # Few-shot結果
    fewshot_path = os.path.join(output_dir, "preference_extraction_fewshot_result.json")
    with open(fewshot_path, "w", encoding="utf-8") as f:
        json.dump(output_fewshot, f, indent=2, ensure_ascii=False)
    print(f"Few-shot結果を保存: {fewshot_path}")
    
    print("\n=== Zero-shot Output ===")
    print(json.dumps(output_zeroshot, indent=2, ensure_ascii=False))
    print("\n=== Few-shot Output ===")
    print(json.dumps(output_fewshot, indent=2, ensure_ascii=False))
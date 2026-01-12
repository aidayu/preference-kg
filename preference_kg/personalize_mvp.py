import json
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# さっき抽出されたJSON（本当はDBから取ってくる想定）
# target_labelが変でも、とりあえずシステムが動くか確認する
USER_PROFILE = {
  "user_id": "user_001",
  "preferences": [
    {
      "target_label": "aesthetic/sensory", # 本来は "coffee maker" であるべき箇所
      "axis": "liking",
      "sub_axis": "aesthetic/sensory",
      "reasoning": "ユーザーは見た目重視"
    },
    {
      "target_label": "functional", # 本来は "coffee maker" であるべき箇所
      "axis": "need",
      "sub_axis": "functional",
      "reasoning": "機能も大事"
    }
  ]
}

def generate_personalized_response(user_input: str, profile: dict):
    # 嗜好データをプロンプトに埋め込むためのテキスト変換
    # 簡易的に "Target: Axis(Sub-Axis)" の形式にする
    memory_text = ""
    for pref in profile["preferences"]:
        memory_text += f"- User likes/wants '{pref['target_label']}' ({pref['axis']}: {pref['sub_axis']})\n"

    print(f"--- 注入するメモリ(嗜好) ---\n{memory_text}---------------------------")

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""
あなたは優秀なアシスタントです。以下のユーザーの嗜好情報を踏まえて、親身に回答してください。
特にユーザーが過去に高く評価したものや、重視しているポイントに触れると効果的です。

[User Preferences]
{memory_text}
"""
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    )
    return completion.choices[0].message.content

# --- テスト実行 ---
# 新しい会話：ユーザーが具体的な商品を提案してほしいと言ってきた
new_input = "デロンギのエスプレッソマシンと、バルミューダのやつ、どっちがいいと思う？"

response = generate_personalized_response(new_input, USER_PROFILE)
print(f"User: {new_input}")
print(f"AI: {response}")
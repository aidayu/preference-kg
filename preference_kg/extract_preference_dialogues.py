#!/usr/bin/env python3
"""
指定したdialogue_idのデータを抽出し、contains_preferenceキーを削除する
"""
import json

# 抽出対象のdialogue_id
dialogue_ids_with_preference = [
    2, 5, 6, 13, 14, 17, 18, 22, 23, 24,
    28, 34, 40, 45, 47, 55, 62, 65, 71, 72,
    73, 76, 86, 89, 102, 104, 106, 124, 127, 137,
    138, 141, 144, 145, 152, 153, 154, 157, 158, 170,
    171, 176, 177, 178, 185, 202, 206, 208, 214, 222
]

# 入力・出力ファイルパス
input_file = "../data/raw/DailyDialog/dailydialog_all_data.json"
output_file = "../data/raw/DailyDialog/dailydialog_trial3.json"

# データ読み込み
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 指定されたIDのデータを抽出し、contains_preferenceキーを削除
extracted_data = []
for item in data:
    if item['dialogue_id'] in dialogue_ids_with_preference:
        # contains_preferenceキーを削除（存在する場合）
        if 'contains_preference' in item:
            del item['contains_preference']
        extracted_data.append(item)

# 保存
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, indent=4, ensure_ascii=False)

print(f"抽出完了: {output_file}")
print(f"抽出件数: {len(extracted_data)}件")

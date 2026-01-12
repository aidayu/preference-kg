#!/usr/bin/env python3
"""
対話データのcontains_preferenceフィールドをアノテーションするツール
"""
import json
import sys
import os


def load_json(filepath):
    """JSONファイルを読み込む"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(filepath, data):
    """JSONファイルを保存する"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def display_dialogue(dialogue_text):
    """対話を見やすく表示する"""
    print("\n" + "=" * 70)
    print(dialogue_text)
    print("=" * 70)


def get_preference_input():
    """ユーザーからの入力を取得"""
    while True:
        choice = input("\n嗜好を含んでいますか？ [1: Yes(true) / 0: No(false) / s: Skip / q: Quit]: ").strip().lower()
        
        if choice == '1':
            return True
        elif choice == '0':
            return False
        elif choice == 's':
            return None  # スキップ
        elif choice == 'q':
            return 'quit'
        else:
            print("無効な入力です。1, 0, s, または q を入力してください。")


def annotate_preferences(filepath, skip_annotated=True):
    """
    対話データをアノテーションする
    
    Args:
        filepath: JSONファイルのパス
        skip_annotated: 既にアノテーション済みのものをスキップするか
    """
    # データを読み込む
    data = load_json(filepath)
    
    total = len(data)
    annotated_count = sum(1 for item in data if item.get('contains_preference') is not None)
    
    print(f"\n総データ数: {total}")
    print(f"アノテーション済み: {annotated_count}")
    print(f"未アノテーション: {total - annotated_count}")
    
    if skip_annotated:
        print("\n※既にアノテーション済みのデータはスキップします")
    
    # アノテーション開始
    modified = False
    
    for idx, item in enumerate(data):
        # 既にアノテーション済みの場合
        if skip_annotated and item.get('contains_preference') is not None:
            continue
        
        # 対話情報を表示
        print(f"\n[{idx + 1}/{total}] dialogue_id: {item['dialogue_id']} (original_index: {item['original_index']})")
        display_dialogue(item['original_dialogue'])
        
        # 現在の値を表示
        current_value = item.get('contains_preference')
        if current_value is not None:
            print(f"現在の値: {current_value}")
        
        # 入力を取得
        result = get_preference_input()
        
        if result == 'quit':
            print("\nアノテーションを終了します...")
            break
        elif result is None:
            print("スキップしました")
            continue
        else:
            item['contains_preference'] = result
            modified = True
            print(f"✓ 設定しました: {result}")
    
    # 保存
    if modified:
        save_json(filepath, data)
        print(f"\n変更を保存しました: {filepath}")
    else:
        print("\n変更はありませんでした")
    
    # 最終統計
    final_annotated = sum(1 for item in data if item.get('contains_preference') is not None)
    print(f"\nアノテーション完了数: {final_annotated}/{total}")


def main():
    if len(sys.argv) < 2:
        print("使用方法: python annotate_preference.py <jsonファイルのパス>")
        print("オプション: --all (既にアノテーション済みのデータも含める)")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません: {filepath}")
        sys.exit(1)
    
    # オプション確認
    skip_annotated = '--all' not in sys.argv
    
    try:
        annotate_preferences(filepath, skip_annotated)
    except KeyboardInterrupt:
        print("\n\nアノテーションを中断しました")
        sys.exit(0)


if __name__ == '__main__':
    main()

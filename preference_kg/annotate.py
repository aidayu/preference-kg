"""
アノテーションツール: JSONデータセットの空欄を対話的に埋めるプログラム
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from loguru import logger


class AnnotationTool:
    """JSONデータセットのアノテーションを対話的に行うツール"""
    
    def __init__(self, input_path: Path, output_path: Path = None):
        """
        Args:
            input_path: 入力JSONファイルのパス
            output_path: 出力JSONファイルのパス（指定しない場合は自動生成）
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else self._generate_output_path()
        self.data: List[Dict[str, Any]] = []
        self.annotation_fields = [
            'entity',
            'aspect', 
            'polarity',
            'intensity',
            'classification',
            'sub_classification',
            'context',
            'explicitness'
        ]
        
        # 選択肢の定義
        self.choices = {
            'polarity': ['positive', 'neutral', 'negative'],
            'intensity': ['low', 'mid', 'high'],
            'classification': ['liking', 'wanting', 'need'],
            'explicitness': ['explicit', 'implicit']
        }
        
        # classificationの値に応じたsub_classificationの選択肢
        self.sub_classification_choices = {
            'liking': ['aesthetic/sensory', 'stimulation', 'identification'],
            'wanting': ['interest', 'goal'],
            'need': ['functional', 'social', 'personal']
        }
        
        # 現在のclassificationの値を保持（sub_classificationの選択肢決定に使用）
        self.current_classification = None
    
    def _generate_output_path(self) -> Path:
        """出力パスを自動生成"""
        # data/processed/ に保存
        output_dir = self.input_path.parent.parent.parent / "processed" / "annotated"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名に_annotated_タイムスタンプを追加
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.input_path.stem}_annotated_{timestamp}.json"
        return output_dir / filename
    
    def load_data(self):
        """JSONファイルを読み込む"""
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"データを読み込みました: {self.input_path}")
            logger.info(f"総文章数: {len(self.data)}")
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {e}")
            sys.exit(1)
    
    def save_data(self):
        """アノテーション済みデータを保存"""
        try:
            # 親ディレクトリを作成
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
            
            logger.success(f"アノテーション済みデータを保存しました: {self.output_path}")
            
            # ファイルサイズを表示
            file_size = self.output_path.stat().st_size / 1024
            logger.info(f"ファイルサイズ: {file_size:.2f} KB")
        except Exception as e:
            logger.error(f"ファイル保存エラー: {e}")
    
    def get_input(self, prompt: str, field_name: str) -> str:
        """標準入力から値を取得"""
        print(f"\n{prompt}", end=" ")
        sys.stdout.flush()
        value = input().strip()
        return value
    
    def get_choice_input(self, field_name: str, choices: List[str]) -> str:
        """選択肢から値を取得"""
        print(f"\n[選択] {field_name}:")
        for idx, choice in enumerate(choices, 1):
            print(f"  {idx}. {choice}")
        
        while True:
            print(f"選択してください (1-{len(choices)}): ", end="")
            sys.stdout.flush()
            user_input = input().strip()
            
            # 数字の入力をチェック
            try:
                choice_idx = int(user_input)
                if 1 <= choice_idx <= len(choices):
                    selected_value = choices[choice_idx - 1]
                    print(f"  → {field_name}: {selected_value}")
                    return selected_value
                else:
                    print(f"  エラー: 1から{len(choices)}の間の数字を入力してください")
            except ValueError:
                print(f"  エラー: 数字を入力してください")
    
    def annotate_field(self, field_name: str, current_value: Any) -> Any:
        """フィールドの値を取得または入力"""
        # 値が空欄（空文字列またはNone）の場合
        if not current_value or current_value == "":
            # フィールドに応じて選択肢入力または自由入力
            if field_name in self.choices:
                # 選択肢がある場合
                new_value = self.get_choice_input(field_name, self.choices[field_name])
                
                # classificationの値を保存（sub_classificationで使用）
                if field_name == 'classification':
                    self.current_classification = new_value
                
                return new_value
            
            elif field_name == 'sub_classification':
                # sub_classificationはclassificationの値に応じて選択肢が変わる
                if self.current_classification in self.sub_classification_choices:
                    choices = self.sub_classification_choices[self.current_classification]
                    new_value = self.get_choice_input(field_name, choices)
                    return new_value
                else:
                    # classificationが未設定または不明な場合は自由入力
                    new_value = self.get_input(
                        f"[入力] {field_name}:",
                        field_name
                    )
                    print(f"  → {field_name}: {new_value}")
                    return new_value
            
            else:
                # 選択肢がない場合は自由入力
                new_value = self.get_input(
                    f"[入力] {field_name}:",
                    field_name
                )
                print(f"  → {field_name}: {new_value}")
                return new_value
        else:
            # 既に値がある場合はそのまま表示
            print(f"  → {field_name}: {current_value}")
            
            # classificationの値を保存（後続のsub_classificationで使用）
            if field_name == 'classification':
                self.current_classification = current_value
            
            return current_value
    
    def process_annotation(self, annotation: Dict[str, Any], anno_idx: int) -> Dict[str, Any]:
        """1つのアノテーションを処理"""
        print(f"\n--- アノテーション {anno_idx + 1} ---")
        
        # current_classificationをリセット
        self.current_classification = None
        
        # 各フィールドを順番に処理
        for field in self.annotation_fields:
            current_value = annotation.get(field, "")
            annotation[field] = self.annotate_field(field, current_value)
        
        # timestampを自動で設定
        annotation['timestamp'] = datetime.now().isoformat()
        print(f"  → timestamp: {annotation['timestamp']} (自動設定)")
        
        return annotation
    
    def process_sentence(self, sentence_data: Dict[str, Any], sent_idx: int):
        """1つの文章を処理"""
        print("\n" + "=" * 80)
        print(f"文章 {sent_idx + 1}/{len(self.data)} (ID: {sentence_data.get('sentence_id', 'N/A')})")
        print("=" * 80)
        
        # 翻訳テキストを表示
        translated_text = sentence_data.get('translated_text', '')
        original_text = sentence_data.get('original_text', '')
        
        print(f"\n【原文】 {original_text}")
        print(f"【翻訳】 {translated_text}")
        
        # annotationsがあるかチェック
        annotations = sentence_data.get('annotations', [])
        
        if not annotations or len(annotations) == 0:
            print("\n→ アノテーションなし。次の文章へ。")
            return
        
        # 各アノテーションを処理
        for anno_idx, annotation in enumerate(annotations):
            sentence_data['annotations'][anno_idx] = self.process_annotation(
                annotation, 
                anno_idx
            )
    
    def run(self):
        """アノテーション処理を実行"""
        logger.info("アノテーションツールを開始します")
        
        # データ読み込み
        self.load_data()
        
        print("\n" + "=" * 80)
        print("アノテーション開始")
        print("=" * 80)
        print("空欄のフィールドには値を入力してください。")
        print("既に値があるフィールドはそのまま表示されます。")
        print("Ctrl+C で中断できます。")
        
        try:
            # 各文章を処理
            for sent_idx, sentence_data in enumerate(self.data):
                self.process_sentence(sentence_data, sent_idx)
            
            print("\n" + "=" * 80)
            print("アノテーション完了")
            print("=" * 80)
            
            # データを保存
            self.save_data()
            
        except KeyboardInterrupt:
            print("\n\n中断されました。")
            
            # 途中までのデータを保存するか確認
            save_partial = input("\n途中までのデータを保存しますか？ (y/n): ").strip().lower()
            if save_partial == 'y':
                self.save_data()
                logger.info("途中までのデータを保存しました")
            else:
                logger.info("データは保存されませんでした")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="JSONデータセットのアノテーションツール"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='入力JSONファイルのパス'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='出力JSONファイルのパス（省略時は自動生成）'
    )
    
    args = parser.parse_args()
    
    # アノテーションツールを実行
    tool = AnnotationTool(args.input_file, args.output)
    tool.run()


if __name__ == "__main__":
    main()

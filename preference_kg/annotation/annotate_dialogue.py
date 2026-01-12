"""
対話データセット用アノテーションツール: 会話から嗜好を抽出してアノテーション
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt, IntPrompt, Confirm


class DialogueAnnotationTool:
    """対話データセットのアノテーションを対話的に行うツール"""
    
    def __init__(self, input_path: Path = None, output_path: Path = None):
        """
        Args:
            input_path: 入力JSONファイルのパス（省略可能、デフォルト: dailydialog_trial.json）
            output_path: 出力JSONファイルのパス（指定しない場合は自動生成）
        """
        if input_path is None:
            input_path = Path("data/raw/DailyDialog/dailydialog_trial.json")
        if output_path is None:
            output_path = Path("data/processed/annotated/dailydialog_annotated.json")
        self.input_path = Path(input_path) if input_path else None
        self.output_path = Path(output_path) if output_path else self._generate_output_path()
        self.data: List[Dict[str, Any]] = []
        
        # Rich console
        self.console = Console()
        
        # 現在の対話情報（固定表示用）
        self.current_dialogue_header = ""
        self.current_progress_info = ""
        
        # アノテーションフィールド
        self.annotation_fields = [
            'entity',
            'aspect', 
            'classification',
            'sub_classification',
            'polarity',
            'intensity',
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
        
        # 現在のclassificationの値を保持
        self.current_classification = None
    
    def clear_screen(self):
        """画面をクリア"""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def print_dialogue_panel(self):
        """対話パネルを表示"""
        if self.current_dialogue_header:
            panel = Panel(
                self.current_dialogue_header,
                title=f"[bold cyan]{self.current_progress_info}[/bold cyan]",
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(panel)
    
    def set_dialogue_header(self, dialogue_data: Dict[str, Any], dialogue_idx: int):
        """対話のヘッダー情報を設定"""
        # 進捗情報
        self.current_progress_info = f"対話 {dialogue_idx + 1}/{len(self.data)} (ID: {dialogue_data.get('dialogue_id', 'N/A')})"
        
        # 原文と和訳を両方追加
        header_parts = []
        
        original_dialogue = dialogue_data.get('original_dialogue', '')
        if original_dialogue:
            header_parts.append("[bold]【原文】[/bold]")
            header_parts.append(original_dialogue)
        
        translated_dialogue = dialogue_data.get('translated_dialogue', '')
        if translated_dialogue:
            if header_parts:
                header_parts.append("")  # 空行を追加
            header_parts.append("[bold]【和訳】[/bold]")
            header_parts.append(translated_dialogue)
        
        if not original_dialogue and not translated_dialogue and 'utterances' in dialogue_data:
            # utterances形式の場合
            header_parts.append("[bold]【対話内容】[/bold]")
            for utt in dialogue_data['utterances']:
                header_parts.append(f"{utt['speaker']}: {utt['text']}")
        
        if header_parts:
            self.current_dialogue_header = "\n".join(header_parts)
        else:
            self.current_dialogue_header = "(対話内容なし)"
    
    def clear_dialogue_header(self):
        """対話ヘッダーをクリア"""
        self.current_dialogue_header = ""
        self.current_progress_info = ""
    
    def _generate_output_path(self) -> Path:
        """出力パスを自動生成"""
        output_dir = Path("data/processed/annotated")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dialogue_annotated_{timestamp}.json"
        return output_dir / filename
    
    def parse_dialogue_text(self, dialogue_text: str) -> List[Dict[str, str]]:
        """
        プレーンテキスト形式の対話を解析
        
        Args:
            dialogue_text: 対話テキスト（"A: text\nB: text" 形式）
        
        Returns:
            発話のリスト [{"speaker": "A", "text": "..."}, ...]
        """
        lines = dialogue_text.strip().split('\n')
        utterances = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # "Speaker: text" 形式を解析
            if ':' in line:
                parts = line.split(':', 1)
                speaker = parts[0].strip()
                text = parts[1].strip()
                utterances.append({
                    "speaker": speaker,
                    "text": text
                })
        
        return utterances
    
    def load_data_from_file(self):
        """JSONファイルから既存データを読み込む"""
        if not self.input_path or not self.input_path.exists():
            logger.warning("入力ファイルが指定されていないか存在しません")
            return
        
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"データを読み込みました: {self.input_path}")
            logger.info(f"総会話数: {len(self.data)}")
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {e}")
    
    def save_data(self):
        """アノテーション済みデータを保存"""
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
            
            logger.success(f"アノテーション済みデータを保存しました: {self.output_path}")
            
            file_size = self.output_path.stat().st_size / 1024
            logger.info(f"ファイルサイズ: {file_size:.2f} KB")
        except Exception as e:
            logger.error(f"ファイル保存エラー: {e}")
    
    def get_input(self, prompt: str, allow_empty: bool = False) -> str:
        """標準入力から値を取得
        
        Args:
            prompt: 入力プロンプト
            allow_empty: 空欄を許容するかどうか
        """
        while True:
            value = Prompt.ask(f"[bold yellow]{prompt}[/bold yellow]", default="" if allow_empty else ...)
            
            if value or allow_empty:
                return value
            else:
                self.console.print("  [red]エラー: 値を入力してください（空欄は許可されていません）[/red]")
    
    def get_choice_input(self, field_name: str, choices: List[str]) -> str:
        """選択肢から値を取得"""
        while True:
            self.console.print(f"\n[bold green][選択] {field_name}:[/bold green]")
            for idx, choice in enumerate(choices, 1):
                self.console.print(f"  [cyan]{idx}.[/cyan] {choice}")
            
            try:
                choice_idx = IntPrompt.ask(f"選択してください (1-{len(choices)})")
                if 1 <= choice_idx <= len(choices):
                    selected_value = choices[choice_idx - 1]
                    self.console.print(f"  [dim]→ {field_name}: {selected_value}[/dim]")
                    return selected_value
                else:
                    self.console.print(f"  [red]エラー: 1から{len(choices)}の間の数字を入力してください[/red]")
            except Exception:
                self.console.print(f"  [red]エラー: 数字を入力してください[/red]")
    
    def annotate_field(self, field_name: str) -> Any:
        """フィールドの値を入力"""
        if field_name in self.choices:
            new_value = self.get_choice_input(field_name, self.choices[field_name])
            
            if field_name == 'classification':
                self.current_classification = new_value
            
            return new_value
        
        elif field_name == 'sub_classification':
            if self.current_classification in self.sub_classification_choices:
                choices = self.sub_classification_choices[self.current_classification]
                return self.get_choice_input(field_name, choices)
            else:
                new_value = self.get_input(f"[入力] {field_name}:")
                return new_value
        
        else:
            # aspectフィールドのみ空欄を許容
            allow_empty = (field_name == 'aspect')
            new_value = self.get_input(f"[入力] {field_name}:", allow_empty=allow_empty)
            return new_value
    
    def create_annotation(self) -> Dict[str, Any]:
        """1つのアノテーションを作成"""
        annotation = {}
        
        self.current_classification = None
        
        for field in self.annotation_fields:
            annotation[field] = self.annotate_field(field)
        
        # timestampを自動で設定
        annotation['timestamp'] = datetime.now().isoformat()
        
        return annotation
    
    def input_dialogue_manually(self) -> Dict[str, Any]:
        """対話を手動で入力"""
        print("\n" + "=" * 80)
        print("対話を入力してください")
        print("=" * 80)
        print("形式: 「話者: 発言内容」を1行ずつ入力")
        print("例: A: Say , Jim , how about going for a few beers after dinner ?")
        print("空行を入力すると終了します")
        print("-" * 80)
        
        dialogue_lines = []
        line_num = 1
        
        while True:
            line = input(f"[{line_num}] ").strip()
            
            if not line:
                if dialogue_lines:
                    break
                else:
                    print("最低1行は入力してください")
                    continue
            
            dialogue_lines.append(line)
            line_num += 1
        
        # 対話テキストを解析
        dialogue_text = '\n'.join(dialogue_lines)
        utterances = self.parse_dialogue_text(dialogue_text)
        
        return {
            "dialogue_id": len(self.data) + 1,
            "dialogue_text": dialogue_text,
            "utterances": utterances,
            "annotations": []
        }
    
    def process_dialogue(self, dialogue_data: Dict[str, Any], dialogue_idx: int):
        """1つの対話を処理"""
        # 対話ヘッダーを設定
        self.set_dialogue_header(dialogue_data, dialogue_idx)
        
        # 画面クリアしてパネル表示
        self.clear_screen()
        self.print_dialogue_panel()
        
        # annotations フィールドを初期化
        if 'annotations' not in dialogue_data:
            dialogue_data['annotations'] = []
        
        # 嗜好を1つずつ抽出
        annotation_count = 0
        first_question = True
        
        while True:
            self.console.print("\n" + "-" * 60)
            if first_question:
                question = "この対話に嗜好は含まれますか？"
                first_question = False
            else:
                question = "まだ嗜好は含まれますか？"
            
            has_preference = Confirm.ask(f"[bold]{question}[/bold]")
            
            if not has_preference:
                # 最終結果を表示
                self.console.print("\n" + "-" * 60)
                if annotation_count == 0:
                    self.console.print("[yellow]→ 嗜好なし[/yellow]")
                else:
                    self.console.print(f"[green]→ 合計 {annotation_count} 件の嗜好を抽出しました[/green]")
                self.console.print("-" * 60)
                break
            
            # 嗜好を1つ抽出
            annotation_count += 1
            self.console.print(f"\n[bold magenta]--- 嗜好 {annotation_count} ---[/bold magenta]")
            annotation = self.create_annotation()
            dialogue_data['annotations'].append(annotation)
        
        # 対話ヘッダーをクリア（次の対話へ）
        self.clear_dialogue_header()
    
    def set_dialogue_header_with_progress(self, dialogue_data: Dict[str, Any], data_idx: int, 
                                          progress_idx: int, total_unannotated: int):
        """進捗表示付きで対話のヘッダー情報を設定"""
        # 進捗情報
        self.current_progress_info = f"対話 {progress_idx + 1}/{total_unannotated} (全体ID: {dialogue_data.get('dialogue_id', data_idx + 1)})"
        
        # 原文と和訳を両方追加
        header_parts = []
        
        original_dialogue = dialogue_data.get('original_dialogue', '')
        if original_dialogue:
            header_parts.append("[bold]【原文】[/bold]")
            header_parts.append(original_dialogue)
        
        translated_dialogue = dialogue_data.get('translated_dialogue', '')
        if translated_dialogue:
            if header_parts:
                header_parts.append("")  # 空行を追加
            header_parts.append("[bold]【和訳】[/bold]")
            header_parts.append(translated_dialogue)
        
        if not original_dialogue and not translated_dialogue and 'utterances' in dialogue_data:
            # utterances形式の場合
            header_parts.append("[bold]【対話内容】[/bold]")
            for utt in dialogue_data['utterances']:
                header_parts.append(f"{utt['speaker']}: {utt['text']}")
        
        if header_parts:
            self.current_dialogue_header = "\n".join(header_parts)
        else:
            self.current_dialogue_header = "(対話内容なし)"
    
    def process_dialogue_without_header(self, dialogue_data: Dict[str, Any]):
        """ヘッダー設定済みの対話を処理"""
        # 画面クリアしてパネル表示
        self.clear_screen()
        self.print_dialogue_panel()
        
        # annotations フィールドを初期化
        if 'annotations' not in dialogue_data:
            dialogue_data['annotations'] = []
        
        # 嗜好を1つずつ抽出
        annotation_count = 0
        first_question = True
        
        while True:
            self.console.print("\n" + "-" * 60)
            if first_question:
                question = "この対話に嗜好は含まれますか？"
                first_question = False
            else:
                question = "まだ嗜好は含まれますか？"
            
            has_preference = Confirm.ask(f"[bold]{question}[/bold]")
            
            if not has_preference:
                # 最終結果を表示
                self.console.print("\n" + "-" * 60)
                if annotation_count == 0:
                    self.console.print("[yellow]→ 嗜好なし[/yellow]")
                else:
                    self.console.print(f"[green]→ 合計 {annotation_count} 件の嗜好を抽出しました[/green]")
                self.console.print("-" * 60)
                break
            
            # 嗜好を1つ抽出
            annotation_count += 1
            self.console.print(f"\n[bold magenta]--- 嗜好 {annotation_count} ---[/bold magenta]")
            annotation = self.create_annotation()
            dialogue_data['annotations'].append(annotation)
        
        # 対話ヘッダーをクリア（次の対話へ）
        self.clear_dialogue_header()
    
    def run_interactive(self):
        """対話的なアノテーション処理を実行"""
        logger.info("対話データセット アノテーションツールを開始します")
        
        # データを読み込み
        self.load_data_from_file()
        
        if not self.data:
            logger.error("データが読み込まれませんでした")
            return
        
        # アノテーション済みとアノテーション未済の対話を分類
        unannotated_indices = []
        annotated_count = 0
        for idx, dialogue_data in enumerate(self.data):
            annotations = dialogue_data.get('annotations', [])
            if annotations and len(annotations) > 0:
                annotated_count += 1
            else:
                unannotated_indices.append(idx)
        
        # 開始メッセージを表示
        self.console.print("\n" + "=" * 60, style="cyan")
        self.console.print("[bold cyan]対話データセット アノテーション[/bold cyan]")
        self.console.print("=" * 60, style="cyan")
        self.console.print("各対話から嗜好を抽出してアノテーションします。")
        self.console.print("- 嗜好は0個以上含まれます")
        self.console.print("- aspectフィールドのみ空欄を許可")
        self.console.print("- [bold red]Ctrl+C[/bold red] で中断できます")
        self.console.print("=" * 60, style="cyan")
        self.console.print(f"\n[bold]総対話数:[/bold] {len(self.data)}")
        self.console.print(f"[bold]アノテーション済み:[/bold] {annotated_count} 件（スキップ）")
        self.console.print(f"[bold]アノテーション対象:[/bold] {len(unannotated_indices)} 件")
        self.console.print("=" * 60, style="cyan")
        
        if not unannotated_indices:
            logger.info("すべての対話がアノテーション済みです")
            return
        
        # Enterで開始
        Prompt.ask("\n[dim]Enterキーを押して開始[/dim]", default="")
        
        try:
            # 未アノテーションの対話のみを処理
            for progress_idx, data_idx in enumerate(unannotated_indices):
                dialogue_data = self.data[data_idx]
                # 進捗表示用にカスタムヘッダーを設定
                self.set_dialogue_header_with_progress(
                    dialogue_data, data_idx, progress_idx, len(unannotated_indices)
                )
                self.process_dialogue_without_header(dialogue_data)
            
            self.console.print("\n" + "=" * 60, style="green")
            self.console.print("[bold green]アノテーション完了[/bold green]")
            self.console.print("=" * 60, style="green")
            self.console.print(f"[bold]総対話数:[/bold] {len(self.data)}")
            
            total_annotations = sum(len(d.get('annotations', [])) for d in self.data)
            self.console.print(f"[bold]総アノテーション数:[/bold] {total_annotations}")
            
            # データを保存
            self.save_data()
            
        except KeyboardInterrupt:
            self.console.print("\n\n[bold red]中断されました。[/bold red]")
            
            if self.data:
                save_partial = Confirm.ask("途中までのデータを保存しますか？")
                if save_partial:
                    self.save_data()
                    logger.info("途中までのデータを保存しました")
                else:
                    logger.info("データは保存されませんでした")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="対話データセットのアノテーションツール"
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='data/raw/DailyDialog/dailydialog_trial.json',
        help='入力JSONファイルのパス（デフォルト: data/raw/DailyDialog/dailydialog_trial.json）'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='出力JSONファイルのパス（省略時は自動生成）'
    )
    
    args = parser.parse_args()
    
    # アノテーションツールを実行
    tool = DialogueAnnotationTool(args.input, args.output)
    tool.run_interactive()


if __name__ == "__main__":
    main()

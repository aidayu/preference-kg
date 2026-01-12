from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from preference_kg.huggingface_loader import load_hf_dataset, save_dataset_locally
from preference_kg.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# rawフォルダ内のデータセット用サブディレクトリ
DATASET_RAW_DIR = RAW_DATA_DIR / "empathetic_dialogues_llm"

@app.command()
def main():
    """
    Hugging Face からすべてのsplit（train, validation, test）を取得して
    raw/ フォルダに別々のファイルとして保存
    """
    splits = ["train", "valid", "test"]
    
    for split in splits:
        try:
            logger.info(f"Processing split: {split}")
            
            # Hugging Face からデータセットを取得
            dataset = load_hf_dataset("Estwld/empathetic_dialogues_llm", split=split)
            
            # raw/ フォルダに保存
            output_path = DATASET_RAW_DIR / f"{split}.parquet"
            save_dataset_locally(dataset, output_path)
            
        except Exception as e:
            logger.error(f"Error processing split {split}: {e}")
            continue
    
    logger.success("All splits downloaded and saved to raw/")


@app.command()
def extract_sample(split: str = "train", num_samples: int = 5):
    """
    rawフォルダから指定されたsplitの最初のN個をサンプルとして抽出
    
    Args:
        split: 対象のsplit（"train", "validation", "test"）
        num_samples: 抽出する件数
    """
    input_path = DATASET_RAW_DIR / f"{split}.parquet"
    output_jsonl = PROCESSED_DATA_DIR / f"empathetic_dialogues_llm_{split}_example.jsonl"
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    logger.info(f"Extracting {num_samples} samples from {split} split...")
    
    # parquetファイルを読み込み
    import pandas as pd
    df = pd.read_parquet(input_path)
    df_sample = df.head(num_samples)
    
    # JSONLに変換して保存
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for _, row in df_sample.iterrows():
            f.write(row.to_json(force_ascii=False) + '\n')
    
    logger.success(f"Sample file created: {output_jsonl}")


if __name__ == "__main__":
    app()

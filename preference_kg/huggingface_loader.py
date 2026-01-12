# preference_kg/huggingface_loader.py
from datasets import load_dataset
from pathlib import Path
from loguru import logger
from preference_kg.config import RAW_DATA_DIR

def load_hf_dataset(dataset_name: str, split: str = "train", cache_dir: Path = None):
    """
    Hugging Face からデータセットを読み込む
    
    Args:
        dataset_name: Hugging Face のデータセット名
        split: 取得する分割（"train", "validation", "test"）
        cache_dir: キャッシュディレクトリ
    
    Returns:
        Dataset オブジェクト
    """
    if cache_dir is None:
        cache_dir = RAW_DATA_DIR / "huggingface_cache"
    
    logger.info(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir))
    logger.success(f"Dataset loaded successfully. Size: {len(dataset)}")
    
    return dataset

def save_dataset_locally(dataset, output_path: Path):
    """
    データセットをローカルにparquet形式で保存
    
    Args:
        dataset: Hugging Face Dataset オブジェクト
        output_path: 保存先パス（.parquet拡張子）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving dataset to {output_path}...")
    dataset.to_parquet(output_path)
    
    # ファイルサイズを確認
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB単位
    logger.success(f"Dataset saved to {output_path} ({file_size:.2f} MB)")
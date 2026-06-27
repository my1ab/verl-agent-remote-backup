"""
Standalone script to download Search-R1 dataset Parquet files from HuggingFace.
Saves raw files to a specified directory without any processing.

Usage:
    python examples/data_preprocess/download_search_r1_dataset.py
    python examples/data_preprocess/download_search_r1_dataset.py --hf_repo_id PeterJinGo/nq_hotpotqa_train --local_dir ~/data/searchR1_raw
"""
import argparse
import logging
import os

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(hf_repo_id, local_dir):
    """Download train.parquet and test.parquet from HuggingFace dataset repo."""
    local_save_dir = os.path.expanduser(local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    downloaded_files = []

    for split in ["train", "test"]:
        parquet_filename = f"{split}.parquet"
        logger.info(f"Downloading {parquet_filename} from {hf_repo_id}")

        try:
            local_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=parquet_filename,
                repo_type="dataset",
                local_dir=local_save_dir,
                local_dir_use_symlinks=False,
            )
            logger.info(f"Downloaded to {local_path}")
            downloaded_files.append(local_path)
        except EntryNotFoundError:
            logger.warning(f"{parquet_filename} not found in repository {hf_repo_id}")
        except Exception as e:
            logger.error(f"Error downloading {split} split: {e}")

    if not downloaded_files:
        logger.warning("No files were downloaded")
    else:
        logger.info(f"Successfully downloaded {len(downloaded_files)} files to {local_save_dir}")
        for f in downloaded_files:
            logger.info(f"  {f}")

    return downloaded_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Search-R1 dataset from HuggingFace.")
    parser.add_argument(
        "--hf_repo_id",
        default="PeterJinGo/nq_hotpotqa_train",
        help="HuggingFace dataset repository ID.",
    )
    parser.add_argument(
        "--local_dir",
        default="~/data/searchR1_raw",
        help="Local directory to save the downloaded Parquet files.",
    )
    args = parser.parse_args()
    download_dataset(args.hf_repo_id, args.local_dir)
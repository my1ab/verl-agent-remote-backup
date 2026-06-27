"""
ModelScope 下载脚本（对应 searchr1_download.py 的 HuggingFace 版本）
下载内容：
  - yamseyoung/wiki-18-e5-index : part_aa, part_ab (E5 索引文件)
  - yamseyoung/wiki-18-corpus  : wiki-18.jsonl.gz (语料数据)

依赖安装：
  pip install modelscope

使用方式：
  python searchr1_download_ms.py --local_dir /path/to/save
"""

import argparse
import os
from modelscope.hub.file_download import dataset_file_download

parser = argparse.ArgumentParser(description="Download files from ModelScope dataset repository.")
parser.add_argument("--local_dir", type=str, required=True, help="本地保存目录")
args = parser.parse_args()

os.makedirs(args.local_dir, exist_ok=True)

# 1. 下载 E5 索引分片
print("=" * 60)
print("下载 wiki-18-e5-index (索引文件)...")
dataset_id_index = "yamseyoung/wiki-18-e5-index"
for file in ["part_aa", "part_ab"]:
    print(f"  下载 {file} ...")
    dataset_file_download(
        dataset_id=dataset_id_index,
        file_path=file,
        local_dir=args.local_dir,
    )
    print(f"  {file} 下载完成")

# 2. 下载语料数据
print("=" * 60)
print("下载 wiki-18-corpus (语料数据)...")
dataset_file_download(
    dataset_id="yamseyoung/wiki-18-corpus",
    file_path="wiki-18.jsonl.gz",
    local_dir=args.local_dir,
)
print("  wiki-18.jsonl.gz 下载完成")

print("=" * 60)
print(f"所有文件已下载到: {args.local_dir}")
print("文件列表:")
for f in os.listdir(args.local_dir):
    file_path = os.path.join(args.local_dir, f)
    size_gb = os.path.getsize(file_path) / (1024 ** 3)
    print(f"  {f}  ({size_gb:.2f} GB)")

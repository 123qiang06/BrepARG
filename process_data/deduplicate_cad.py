import os
import pickle 
import argparse
from tqdm import tqdm
from hashlib import sha256
from convert_utils import *


def load_pkl_paths(data_folder):
    """
    Recursively find all pkl files under the given folder.
    
    Args:
        data_folder: Path to the data folder.
    
    Returns:
        list: List of all pkl file paths.
    """
    pkl_files = []
    print(f"正在递归搜索目录: {data_folder}")
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    print(f"找到 {len(pkl_files)} 个 pkl 文件")
    return pkl_files


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='data/abc_parsed', help="Path to the data folder")
parser.add_argument("--bit",  type=int, default=6, help="Deduplication precision (bit)")
parser.add_argument(
    "--option",
    type=str,
    choices=["abc", "deepcad", "furniture"],
    default="abc",
    help="Select dataset type [abc/deepcad/furniture] (default: abc)",
)
args = parser.parse_args()

if args.option == 'deepcad': 
    OUTPUT = f'data/deepcad_data_split_{args.bit}bit.pkl'
elif args.option == 'abc': 
    OUTPUT = f'data/abc_data_split_{args.bit}bit.pkl'
else:
    OUTPUT = f'data/furniture_data_split_{args.bit}bit.pkl'

# Load all pkl file paths
all_pkl_paths = load_pkl_paths(args.data)

# Split dataset: 90% train, 5% val, 5% test
import random
random.seed(42)
random.shuffle(all_pkl_paths)

total_count = len(all_pkl_paths)
train_count = int(total_count * 0.9)
val_count = int(total_count * 0.95)

train_all = all_pkl_paths[:train_count]
val_path = all_pkl_paths[train_count:val_count]
test_path = all_pkl_paths[val_count:]

print(f"\nDataset split:")
print(f"  Train: {len(train_all)} files")
print(f"  Val:   {len(val_path)} files")
print(f"  Test:  {len(test_path)} files")

# Deduplicate training set only
print(f"\nStart deduplicating training set...")
train_path = []
unique_hash = set()
total = 0

for path_idx, pkl_path in tqdm(enumerate(train_all), total=len(train_all), desc="Deduplicating train set"):
    total += 1

    # Load pkl data
    try:
        with open(pkl_path, "rb") as file:
            data = pickle.load(file)
    except Exception as e:
        print(f"Failed to read {pkl_path}: {e}")
        continue

    # Check if surf_wcs exists
    if 'surf_wcs' not in data:
        continue

    # Hash sampled surface points
    surfs_wcs = data['surf_wcs']
    surf_hash_total = []
    for surf in surfs_wcs:
        np_bit = real2bit(surf, n_bits=args.bit).reshape(-1, 3)
        data_hash = sha256(np_bit.tobytes()).hexdigest()
        surf_hash_total.append(data_hash)
    surf_hash_total = sorted(surf_hash_total)
    data_hash = '_'.join(surf_hash_total)

    # Save non-duplicate shapes
    prev_len = len(unique_hash)
    unique_hash.add(data_hash)  
    if prev_len < len(unique_hash):
        train_path.append(pkl_path)
        
    if path_idx % 2000 == 0:
        print(f"Deduplication rate: {len(unique_hash)/total:.2%}")

# Save deduplicated path list
print(f"\nTraining set deduplication finished:")
print(f"  Before: {len(train_all)} files")
print(f"  After: {len(train_path)} files")
print(f"  Duplicates removed: {len(train_all) - len(train_path)} files")
print(f"  Retention rate: {len(train_path)/len(train_all):.2%}")

print(f"\nFinal dataset statistics:")
print(f"  Train: {len(train_path)} files (deduplicated)")
print(f"  Val:   {len(val_path)} files")
print(f"  Test:  {len(test_path)} files")
print(f"  Total: {len(train_path) + len(val_path) + len(test_path)} files")

data_path = {
    'train': train_path,
    'val': val_path,
    'test': test_path,
}
with open(OUTPUT, "wb") as tf:
    pickle.dump(data_path, tf)

print(f"\nResult saved to: {OUTPUT}")


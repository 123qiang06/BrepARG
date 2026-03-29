import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from hashlib import sha256


def real2bit(data, n_bits=6, min_range=-1, max_range=1):
    """
    Quantize a real-valued array to integers with n_bits precision.
    Assumes the data lies within [min_range, max_range].
    Args:
        data (np.ndarray): Array to quantize, shape (N, ...).
        n_bits (int): Number of quantization bits.
        min_range (float): Lower bound of value range.
        max_range (float): Upper bound of value range.
    Returns:
        np.ndarray: Integer array with same shape as data, values in [0, 2**n_bits-1].
    """
    range_quantize = 2 ** n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize)
    return data_quantize.astype(np.int64)


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate NCS surfaces or edges from the training data"
    )
    parser.add_argument(
        "--data_list",
        type=str,
        default="data/abc_data_split_6bit.pkl",
        help="Pickle file containing train/test path lists",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["face", "edge"],
        default="face",
        help="Choose 'face' to deduplicate surface NCS or 'edge' to deduplicate edge NCS",
    )
    parser.add_argument(
        "--bit",
        type=int,
        default=6,
        help="Number of quantization bits (default 6)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pkl file path (default: data/{dataset_type}_parsed_unique_{surfaces|edges}.pkl)",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["furniture", "deepcad", "abc"],
        default="abc",
        help="Dataset type: 'furniture' | 'deepcad' | 'abc'",
    )
    args = parser.parse_args()

    # 读取数据列表
    print(f"Loading data list from {args.data_list}...")
    with open(args.data_list, "rb") as f:
        dataset = pickle.load(f)

    pkl_files = dataset["train"]
    print(f"Found {len(pkl_files)} training files")

    # 根据 mode 选择键
    key = "edge_ncs" if args.mode == "edge" else "surf_ncs"
    print(f"Processing {args.mode} NCS data using key '{key}'...")

    unique_hash = set()
    unique_data = []

    processed_files = 0
    skipped_files = 0

    for pkl_file in tqdm(pkl_files, desc="Processing files"):
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {pkl_file}: {e}")
            skipped_files += 1
            continue

        if key not in data:
            print(f"Key '{key}' not found in {pkl_file}")
            skipped_files += 1
            continue

        arrs = data[key]
        if arrs is None or len(arrs) == 0:
            print(f"Empty or invalid {key} data in {pkl_file}")
            skipped_files += 1
            continue

        processed_files += 1

        # 量化并去重
        bits_arr = real2bit(arrs, n_bits=args.bit)
        for bit_repr, real_val in zip(bits_arr, arrs):
            flat_bits = bit_repr.reshape(-1)
            h = sha256(flat_bits.tobytes()).hexdigest()
            if h not in unique_hash:
                unique_hash.add(h)
                unique_data.append(real_val)

    # 目标输出路径
    if args.output:
        out_path = args.output
    else:
        if args.mode == "edge":
            out_path = f"data/{args.dataset_type}_parsed_unique_edges.pkl"
        else:
            out_path = f"data/{args.dataset_type}_parsed_unique_surfaces.pkl"

    # 一次性保存
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(unique_data, f)

    print(f"\nProcessed {processed_files} files successfully, skipped {skipped_files} files")
    print(f"Saved {len(unique_data)} unique NCS {args.mode} items to {out_path}")

    # 统计信息
    if len(unique_data) > 0:
        data_array = np.array(unique_data)
        print(f"Data shape: {data_array.shape}")
        print(f"Data range: [{data_array.min():.4f}, {data_array.max():.4f}]")
        print(f"Data mean: {data_array.mean():.4f}")
        print(f"Data std: {data_array.std():.4f}")


if __name__ == "__main__":
    main()

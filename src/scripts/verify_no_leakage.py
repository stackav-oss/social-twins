"""A script that to verify there are no repeated scenarios in the dataset splits."""

import json
from collections import Counter
from itertools import product
from pathlib import Path

from natsort import natsorted


def run(base_data_path: Path, splits: list[str], meta_path: Path) -> None:
    assert base_data_path.exists(), f"Base data path {base_data_path} does not exist!"

    file_lists = {}
    for split in splits:
        file_lists[split] = []
        print(f"Running {split} split")
        split_path = base_data_path / split
        split_shards = natsorted([subdir for subdir in split_path.iterdir() if subdir.is_dir()])
        for shard_path in split_shards:
            shard_files = [file.name for file in shard_path.iterdir() if file.is_file()]
            print(f"\tFound {len(shard_files)} scenarios in {shard_path}")
            file_lists[split] += shard_files

    # Check that ther arent repeated scenarios within and across splits
    scenario_duplicates = {}
    for split_a, split_b in list(product(splits, splits)):
        key = f"{split_a}_{split_b}"
        print(f"Comparing splits: {split_a} and {split_b}")
        list_a = file_lists[split_a]
        list_b = file_lists[split_b]
        if split_a == split_b:
            duplicates = [item for item, cnt in Counter(list_a).items() if cnt > 1]
        else:
            duplicates = list(set(list_a) & set(list_b))
        num_duplicates = len(duplicates)
        print(f"\t...there are {num_duplicates} intersections.")
        scenario_duplicates[key] = {"num_duplicates": num_duplicates, "scenario_duplicates": duplicates}

    meta_path.mkdir(parents=True, exist_ok=True)
    output_filepath = meta_path / f"{base_data_path.name}_duplicates.json"
    print(f"Saving data selection summary to: {output_filepath}")
    with output_filepath.open("w") as f:
        json.dump(scenario_duplicates, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_data_path", type=Path, default="/datasets/waymo/processed/mini_causal", help="Paths to the input data."
    )
    parser.add_argument(
        "--splits",
        type=list[str],
        nargs="+",
        default=["training", "validation", "testing"],
        help="Dataset splits to check",
    )
    parser.add_argument("--meta_path", type=Path, default="./meta", help="Paths to the input data.")
    args = parser.parse_args()
    run(**vars(args))

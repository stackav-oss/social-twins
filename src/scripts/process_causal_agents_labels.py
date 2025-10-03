"""A script that processes causal agent TF record into JSON files."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from scripts import causal_agents_pb2


def run(causal_labels_filepath: Path, processed_labels_path: Path, summary_filepath: Path) -> None:
    assert causal_labels_filepath.exists(), f"Causal labels path {causal_labels_filepath} does not exist!"
    processed_labels_path.mkdir(parents=True, exist_ok=True)

    causal_labels = tf.data.TFRecordDataset(causal_labels_filepath, compression_type="")
    print("\n\nProcessing causal agents labels...")
    total_scenarios = 0
    data_summary = {
        "scenario_ids": [],
        "num_scenarios": 0,
    }
    for data in tqdm(causal_labels):
        total_scenarios += 1
        labels = causal_agents_pb2.CausalLabels()
        labels.ParseFromString(bytearray(data.numpy()))

        label_info = {}
        scenario_id = labels.scenario_id
        data_summary["scenario_ids"].append(scenario_id)
        label_info["scenario_id"] = scenario_id
        label_info["labeler_results"] = {}
        labeler_results = list(labels.labeler_results)

        causal_agent_ids = []
        for n, labeler in enumerate(labeler_results):
            labeler_key = f"labeler_{n}"
            causal_ids = list(labeler.causal_agent_ids)
            label_info["labeler_results"][labeler_key] = causal_ids
            causal_agent_ids += causal_ids

        causal_agent_ids = np.array(causal_agent_ids)
        unique_ids, counts = np.unique(causal_agent_ids, return_counts=True)
        label_info["causal_ids"] = unique_ids.tolist()
        label_info["labeler_votes"] = counts.tolist()

        # Save label info
        label_info_filepath = processed_labels_path / f"{scenario_id}.json"
        with label_info_filepath.open("w") as f:
            json.dump(label_info, f, indent=4)
    data_summary["num_scenarios"] = total_scenarios
    print(f"Processed labels from {total_scenarios} scenarios")

    with summary_filepath.open("w") as f:
        json.dump(data_summary, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--causal_labels_filepath",
        type=Path,
        default="/datasets/waymo/causal_agents/causal_labels.tfrecord",
        help="Paths to the input data.",
    )
    parser.add_argument(
        "--processed_labels_path",
        type=Path,
        default="/datasets/waymo/causal_agents/processed_labels",
        help="Paths to the processed data.",
    )
    parser.add_argument(
        "--summary_filepath",
        type=Path,
        default="/datasets/waymo/causal_agents/summary.json",
        help="Paths to the processed data.",
    )
    args = parser.parse_args()

    run(**vars(args))

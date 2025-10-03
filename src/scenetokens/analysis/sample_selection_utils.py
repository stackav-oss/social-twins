"""Utility functions to perform model analysis. See 'docs/ANALYSIS.md' for details on usage."""

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from scenetokens.analysis.model_analysis_utils import get_scenario_classes_best_mode
from scenetokens.schemas import output_schemas as output


def random_selection(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """A sample selection strategy that randomly keeps a specified percentage of all scenarios.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a list of model outputs per scenario.

    Returns:
        dict[str, Any]: a dictionary containing the IDs of the samples (scenarios) to keep or drop for training.
    """
    scenario_ids, _, _, _ = get_scenario_classes_best_mode(model_outputs)
    scenario_ids = scenario_ids.tolist()
    num_scenarios = len(scenario_ids)
    random.seed(config.seed)
    random.shuffle(scenario_ids)

    min_scenarios_to_keep = int(config.min_percentage * num_scenarios)
    keep = scenario_ids[:min_scenarios_to_keep]
    drop = scenario_ids[min_scenarios_to_keep:]
    return {"keep": keep, "num_to_keep": len(keep), "drop": drop, "num_to_drop": len(drop)}


def random_selection_per_class(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """A sample selection strategy that randomly keeps a specified percentage of the scenarios for each class that has
    more than a desired minimum percentage.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a list of model outputs per scenario.

    Returns:
        dict[str, Any]: a dictionary containing the IDs of the samples (scenarios) to keep or drop for training.
    """
    scenario_ids, scenario_classes, _, _ = get_scenario_classes_best_mode(model_outputs)
    num_scenarios, num_modes = scenario_classes.shape
    classes_dict = {"scenario_id": scenario_ids, "scenario_class": scenario_classes[:, 0]}
    classes_df = pd.DataFrame(classes_dict)
    percentage_per_class = (classes_df["scenario_class"].value_counts() / num_scenarios).to_frame(name="percentage")
    min_scenarios_to_keep = int(config.min_percentage_per_class * num_scenarios)

    selected_samples = {}
    for _, row in percentage_per_class.iterrows():
        scenario_class = row.name
        scenario_ids_in_class = classes_df["scenario_id"][classes_df["scenario_class"] == scenario_class].tolist()
        percentage = row.percentage
        if percentage > config.min_percentage_per_class:
            # Select samples to drop
            random.seed(config.seed)
            random.shuffle(scenario_ids_in_class)
            keep = scenario_ids_in_class[:min_scenarios_to_keep]
            drop = scenario_ids_in_class[min_scenarios_to_keep:]
            selected_samples[scenario_class] = {
                "keep": keep,
                "num_to_keep": len(keep),
                "drop": drop,
                "num_to_drop": len(drop),
            }
        else:
            selected_samples[scenario_class] = {
                "keep": scenario_ids_in_class,
                "num_to_keep": len(scenario_ids_in_class),
                "drop": [],
            }

    # Get combined list
    keep = []
    drop = []
    for samples in selected_samples.values():
        keep += samples["keep"]
        drop += samples["drop"]
    selected_samples["keep"] = keep
    selected_samples["drop"] = drop
    selected_samples["num_to_keep"] = len(selected_samples["keep"])
    selected_samples["num_to_drop"] = len(selected_samples["drop"])
    return selected_samples


def run_sample_selection(config: DictConfig, model_outputs: dict[str, output.ModelOutput], output_path: Path) -> None:
    """Wrapper function which runs a specified sample selection strategy. A sample selection strategy produces a
    dictionary containing the a set of training scenarios to keep and to drop.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a list of model outputs per scenario.
        output_path (Path): output path where visualization will be saved to.
    """
    selection_strategy = config.selection_strategy
    match selection_strategy:
        case "random_per_class":
            sample_selection = random_selection_per_class(config, model_outputs)
        case "random":
            sample_selection = random_selection(config, model_outputs)
        case _:
            error_message = f"Unsupported selection strategy: {selection_strategy}"
            raise ValueError(error_message)

    output_filepath = output_path / f"{selection_strategy}_sample_selection.json"
    with output_filepath.open("w") as f:
        json.dump(sample_selection, f, indent=2)

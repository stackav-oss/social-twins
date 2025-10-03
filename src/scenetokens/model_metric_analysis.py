"""Models Metrics Analysis Script.

Example usage:

    uv run -m scenetokens.model_metric_analysis group_name=[name]

See `docs/ANALYSIS.md` for more argument details.
"""

import itertools
import random
from pathlib import Path
from time import time

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import pyrootutils
import seaborn as sns
from omegaconf import DictConfig

from scenetokens import utils


log = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="model_metric_analysis.yaml")
def main(config: DictConfig) -> float | None:
    """Hydra's entrypoint for running scenario analysis training."""
    log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
    utils.print_config_tree(config, resolve=True, save_to_file=False)
    random.seed(config.seed)

    start = time()
    output_path = Path(f"{config.output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Read the metrics from the comparison group csv
    metrics_df = pd.read_csv(config.group_file)

    # Each metric in the dataframe follows the pattern 'split/metric_type' or 'split/subset/metric_type'. Below, we
    # filter the metrics by split and metric
    for split, metric in itertools.product(config.splits_to_compare, config.metrics_to_compare):
        log.info("Comparing split: %s and metric: %s", split, metric)

        # Get a sub dataframe that contains only the desired metric from the desired split
        filtered_df = metrics_df[metrics_df.Metric.str.contains(split) & metrics_df.Metric.str.contains(metric)]
        if len(filtered_df) == 0:
            log.info("Missing metric %s; skipping analysis.", metric)
            continue

        if metric == "missRate":
            filtered_df = filtered_df[~filtered_df.Metric.str.contains(metric + "6")]

        # Ad-hoc renaming 'split/dataset-subset-subsplit/metric' -> 'subset'
        filtered_df.Metric = filtered_df.Metric.str.extract(r"(?<=/)([^/]+)(?=/)", expand=False)
        filtered_df.Metric = filtered_df.Metric.str.extract(r"(?<=-)([^/]+)(?=-)", expand=False)
        df_melted = filtered_df.melt(id_vars=["Metric"], var_name="Model", value_name="Value")

        # Create barplot comparing the model groups
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Metric", y="Value", hue="Model", data=df_melted, palette=config.palette)

        # Scale the y-axis to highlight the metric differences. This just shows the y-range from 10% less than the min
        # metric value to 10% more than the max metric value.
        min_value, max_value = df_melted.min().Value, df_melted.max().Value
        plt.ylim(min_value - 0.1 * min_value, max(max_value + 0.1 * max_value, 1.0))

        plt.title(f"Split: {split}, Metric: {metric}")
        plt.xlabel("Metrics by Subset")
        plt.ylabel("Metrics Values")
        plt.legend(title="Models", title_fontsize="12", fontsize="10", loc="upper left")

        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Show the plot
        output_filepath = f"{output_path}/{split}_{metric}.png"
        plt.savefig(output_filepath, dpi=200)

    log.info("Total time: %s second", time() - start)
    log.info("Process completed!")


if __name__ == "__main__":
    main()

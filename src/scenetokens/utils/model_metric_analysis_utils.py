from itertools import product
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from scenetokens.utils.constants import SMALL_EPSILON


def group_analysis(config: DictConfig, log: Logger) -> None:
    """Loads a CSV containing model metrics from a specified group and produces per-metric barplots with model-to-model
    comparisons.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging anslysis information.
    """
    base_path = Path(config.base_path)
    output_path = base_path / config.group_name
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(config.group_file)

    # Each metric in the dataframe follows the pattern 'split/metric_type' or 'split/subset/metric_type'. Below, we
    # filter the metrics by split and metric
    for split, metric in product(config.splits_to_compare, config.metrics_to_compare):
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


def sample_selection_analysis(config: DictConfig, log: Logger) -> None:
    """Loads a CSV containing all model metrics downlaoded from MLflow from produces per-metric barplots with sample
    selection comparisons.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging anslysis information.
    """
    base_path = Path(config.base_path)
    output_path = base_path / "sample_selection_analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    # The 'all_runs.csv' file contains the metrics for all experiments and is exported from MLflow
    metrics_df = pd.read_csv(base_path / "all_runs.csv")

    # Models to analyze
    models = ["scenetf", "wayformer", "st-student", "st-teacher", "st-teacher-u"]
    sample_selection_strategies = ["none", "uniform-random", "token-random", "token-jaccsim", "token-jaccgum"]
    subset = ["waymo-mini-causal-", "waymo-remove-noncausal-"]

    for split, metric, model in product(config.splits_to_compare, config.metrics_to_compare, models):
        log.info("Comparing split: %s, metric: %s, model: %s", split, metric, model)

        # select columns that contain the split string
        metric_cols = [column for column in metrics_df.columns if metric in column]
        subset_cols = [column for column in metric_cols if any(sub in column for sub in subset)]
        valid_cols = ["Name"] + [column for column in subset_cols if split in column]
        if not valid_cols:
            log.info("No columns found for split %s; skipping.", split)
            continue

        # filter rows by model and sample selection strategy, keep only the split-related columns + model_name
        filtered_df = metrics_df[valid_cols].copy()
        if filtered_df.empty:
            log.info("Missing data; skipping")
            continue

        strategy_models = [model] + [
            model + f"_{ss_strategy}" for ss_strategy in sample_selection_strategies if ss_strategy != "none"
        ]
        filtered_df = filtered_df[filtered_df["Name"].isin(strategy_models)]
        # preserve the desired model order by making "Name" a categorical with the strategy_models ordering
        filtered_df["Name"] = pd.Categorical(filtered_df["Name"], categories=strategy_models, ordered=True)
        filtered_df = filtered_df.sort_values("Name").reset_index(drop=True)

        # Melt for seaborn plotting
        cols_to_melt = [col for col in filtered_df.columns if col != "Name"]
        df_melted = filtered_df.melt(id_vars=["Name"], value_vars=cols_to_melt, var_name="Metric", value_name="Value")
        if metric == "missRate":
            df_melted = df_melted[~df_melted.Metric.str.contains(metric + "6")]

        # skip if all values are NaN
        if df_melted["Value"].isna().all():
            log.info("All values are NaN for split=%s, metric=%s, model=%s; skipping.", split, metric, model)
            continue

        # Create barplot comparing the model groups
        plt.subplots(figsize=(10, 6))
        sns.barplot(x="Metric", y="Value", hue="Name", data=df_melted, palette=config.palette)

        min_value, max_value = df_melted.min().Value, df_melted.max().Value
        plt.ylim(min_value - 0.1 * min_value, max(max_value + 0.1 * max_value, 1.0))

        plt.title(f"Sample Selection Analysis for {model}")
        plt.xlabel("Metrics")
        plt.ylabel("Metrics Values")
        plt.legend(title="Models", title_fontsize="12", fontsize="10", loc="upper left")

        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Show the plot
        output_filepath = output_path / f"{split}_{model}_{metric}.png"
        plt.savefig(output_filepath, dpi=200)
        plt.close()


def model_to_model_analysis(config: DictConfig, log: Logger) -> None:  # noqa: PLR0915
    """Loads a CSV containing all model metrics downlaoded from MLflow from produces per-metric barplots with model to
    model comparisons and generalization assessments.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging anslysis information.
    """
    base_path = Path(config.base_path)
    output_path = base_path / "model_to_model_analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    # The 'all_runs.csv' file contains the metrics for all experiments and is exported from MLflow
    metrics_df = pd.read_csv(base_path / "all_runs.csv")

    # Models to analyze
    models = ["scenetf", "wayformer", "st-student", "st-teacher", "st-teacher-u"]
    sample_selection_strategies = ["none", "uniform-random", "token-random", "token-jaccsim", "token-jaccgum"]
    subset = ["waymo-mini-causal-", "waymo-remove-noncausal-"]

    for metric in config.metrics_to_compare:
        generalization = pd.DataFrame({"models": models})

        for split, ss_strategy in product(config.splits_to_compare, sample_selection_strategies):
            log.info("Comparing split: %s, metric: %s, sample selection strategy: %s", split, metric, ss_strategy)

            # select columns that contain the split string
            metric_cols = [column for column in metrics_df.columns if metric in column]
            subset_cols = [column for column in metric_cols if any(sub in column for sub in subset)]
            valid_cols = ["Name"] + [column for column in subset_cols if split in column]
            if not valid_cols:
                log.info("No columns found for split %s; skipping.", split)
                continue

            # filter rows by model and sample selection strategy, keep only the split-related columns + model_name
            filtered_df = metrics_df[valid_cols].copy()
            if filtered_df.empty:
                log.info("Missing data; skipping")
                continue

            strategy_models = [model + f"_{ss_strategy}" for model in models] if ss_strategy != "none" else models
            filtered_df = filtered_df[filtered_df["Name"].isin(strategy_models)]
            # preserve the desired model order by making "Name" a categorical with the strategy_models ordering
            filtered_df["Name"] = pd.Categorical(filtered_df["Name"], categories=strategy_models, ordered=True)
            filtered_df = filtered_df.sort_values("Name").reset_index(drop=True)
            if filtered_df.empty:
                log.info("No rows match models for strategy %s; skipping.", ss_strategy or "none")
                continue

            # Compute generalization values
            unperturbed_values = filtered_df[filtered_df.columns[1]] + SMALL_EPSILON
            perturbed_values = filtered_df[filtered_df.columns[2]]
            generalization_values = 100 * (perturbed_values - unperturbed_values) / unperturbed_values.abs()
            generalization[ss_strategy] = generalization_values

            # Melt for seaborn plotting
            cols_to_melt = [col for col in filtered_df.columns if col != "Name"]
            df_melted = filtered_df.melt(
                id_vars=["Name"], value_vars=cols_to_melt, var_name="Metric", value_name="Value"
            )
            if metric == "missRate":
                df_melted = df_melted[~df_melted.Metric.str.contains(metric + "6")]

            # Create barplot comparing the model groups
            plt.subplots(figsize=(10, 6))
            sns.barplot(x="Metric", y="Value", hue="Name", data=df_melted, palette=config.palette)

            min_value, max_value = df_melted.min().Value, df_melted.max().Value
            plt.ylim(min_value - 0.1 * min_value, max(max_value + 0.1 * max_value, 1.0))
            # plt.yscale("log")
            # plt.ylim(bottom=0.0, top=10)

            plt.title(f"Sample Selection Strategy: {ss_strategy}")
            plt.xlabel("Metrics")
            plt.ylabel("Metrics Values")
            plt.legend(title="Models", title_fontsize="12", fontsize="10", loc="upper left")

            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()

            # Show the plot
            output_filepath = output_path / f"{split}_{metric}_ss-{ss_strategy}.png"
            plt.savefig(output_filepath, dpi=200)
            plt.close()

        # Generalization plot
        plt.subplots(figsize=(10, 6))

        generalization_melted = generalization.melt(
            id_vars=["models"], var_name="Strategy", value_name="Generalization"
        )
        sns.barplot(x="Strategy", y="Generalization", hue="models", data=generalization_melted, palette=config.palette)

        min_value, max_value = generalization_melted.min().Generalization, generalization_melted.max().Generalization
        plt.ylim(min_value - 0.1 * min_value, max(max_value + 0.1 * max_value, 1.0))

        plt.title(f"Generalization: {metric}")
        plt.xlabel("Strategy")
        plt.ylabel("Generalization")
        plt.legend(title="Models", title_fontsize="12", fontsize="10", loc="upper left")

        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Show the plot
        output_filepath = output_path / f"generalization_{split}_{metric}.png"
        plt.savefig(output_filepath, dpi=200)
        plt.close()

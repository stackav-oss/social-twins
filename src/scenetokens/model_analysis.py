"""Scenario Analysis Script.

Example usage:

    uv run -m scenetokens.model_analysis analysis=tokenization num_batches=[x] num_scenarios=[y]

See `docs/ANALYSIS.md` and 'configs/analysis.yaml' for more argument details.
"""

import random
from pathlib import Path
from time import time

import hydra
import pyrootutils
from omegaconf import DictConfig

from scenetokens import utils


log = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="analysis.yaml")
def main(config: DictConfig) -> float | None:
    """Hydra's entrypoint for running scenario analysis training."""
    log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
    utils.print_config_tree(config, resolve=True, save_to_file=False)
    random.seed(config.seed)

    start = time()
    output_path = Path(config.paths.analysis_path) / f"{config.split}_model-analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    # Loads scenario tokenized information
    log.info("Loading batches from %s", config.paths.batch_cache_path)
    batches = utils.load_batches(
        config.paths.batch_cache_path, config.num_batches, config.num_scenarios, config.seed, config.split
    )

    # Run analyses

    # Produces a histogram over the tokenized scenarios, representing token-utilization.
    if config.run_scenario_class_distribution:
        log.info("Running tokenization distribution analysis...")
        utils.plot_scenario_class_distribution(config, batches, output_path)

    # Produces a consistency analysis over tokenized groups.
    if config.run_token_consistency_analysis:
        log.info("Running tokenization consistency distribution analysis...")
        consistency_matrix = utils.compute_token_consistency_matrix(config, batches)
        utils.plot_heatmap(
            consistency_matrix,
            title="Token Asignment Consistency",
            x_label="Group Mode",
            y_label="Token Group",
            cbar_label="Average Consistency Index",
            output_filepath=output_path / f"token_consistency_matrix_{config.consistency_measure}.png",
        )

    # Producess a distribution analysis over tokenized groups.
    if config.run_group_uniqueness_analysis:
        log.info("Running group uniqueness distribution analysis...")
        group_uniqueness_index, group_uniqueness_counts = utils.compute_group_uniqueness(config, batches)
        utils.plot_uniqueness_index(config, group_uniqueness_index, output_path)
        utils.plot_heatmap(
            group_uniqueness_counts,
            title="Group Uniqueness Heatmap",
            x_label="Group",
            y_label="Vocabulary",
            cbar_label="Uniqueness Index",
            output_filepath=output_path / f"group_uniqueness_counts_{config.normalize_counts}.png",
        )

    if config.run_intergroup_analysis:
        log.info("Running intergroup distribution analysis...")
        intergroup_uniqueness = utils.compute_intergroup_uniqueness(config, batches)
        utils.plot_heatmap(
            intergroup_uniqueness,
            title="Intergroup Uniqueness Heatmap",
            x_label="Group",
            y_label="Group",
            cbar_label="Uniqueness Index",
            output_filepath=output_path / "intergroup_uniqueness.png",
        )

    # Produces a visualization of the scenario embeddings using dimensionality reduction methods.
    if config.run_dim_reduction_analysis:
        log.info("Running dimensionality reduction analysis...")
        dim_reduction_result = utils.compute_dimensionality_reduction(config, batches, output_path)
        utils.plot_manifold_by_tokens(config, dim_reduction_result, batches, output_path)

    # This analysis produces a per-class score analysis over the scenarios.
    if config.run_score_analysis:
        log.info("Running score analysis...")
        score_analysis = utils.compute_score_analysis(config, batches, output_path)
        if config.viz_scored_scenarios:
            utils.plot_tokenized_scenarios_by_score_percentile(config, batches, score_analysis, output_path)

    log.info("Total time: %s second", time() - start)
    log.info("Process completed!")


if __name__ == "__main__":
    main()

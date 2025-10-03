"""Scenario Analysis Script.

Example usage:

    uv run -m scenetokens.model_analysis num_batches=[x] num_scenarios=[y]

See `docs/ANALYSIS.md` for more argument details.
"""

import random
from pathlib import Path
from time import time

import hydra
import pyrootutils
from omegaconf import DictConfig

from scenetokens import analysis, utils


log = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="model_analysis.yaml")
def main(config: DictConfig) -> float | None:
    """Hydra's entrypoint for running scenario analysis training."""
    log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
    utils.print_config_tree(config, resolve=True, save_to_file=False)
    random.seed(config.seed)

    start = time()
    output_path = Path(f"{config.output_path}/{config.split}_model-analysis")
    output_path.mkdir(parents=True, exist_ok=True)

    # Loads scenario tokenized information
    batches = utils.load_batches(
        config.batches_path, config.num_batches, config.num_scenarios, config.seed, config.split
    )

    # Run analyses

    # Produces a histogram over the tokenized scenarios, representing token-utilization.
    if config.run_distribution_analysis:
        log.info("Running tokenization distribution analysis...")
        analysis.plot_scenario_class_distribution(config, batches, output_path)

    # Produces a visualization of the scenario embeddings using dimensionality reduction methods.
    dim_reduction_result = None
    if config.run_dim_reduction_analysis:
        log.info("Running dimensionality reduction analysis...")
        dim_reduction_result = analysis.compute_dimensionality_reduction(config, batches, output_path)
        analysis.plot_manifold_by_tokens(config, dim_reduction_result, batches, output_path)

    # This analysis produces a per-class score analysis over the scenarios.
    if config.run_score_analysis:
        log.info("Running score analysis...")
        analysis.compute_score_analysis(config, batches, output_path)

    log.info("Total time: %s second", time() - start)
    log.info("Process completed!")


if __name__ == "__main__":
    main()

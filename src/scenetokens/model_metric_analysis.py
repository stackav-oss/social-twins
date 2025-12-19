"""Models Metrics Analysis Script.

Example usage:

    uv run -m scenetokens.model_metric_analysis group_name=[name]

See `docs/ANALYSIS.md` and 'configs/model_metric_analysis.yaml' for more argument details.
"""

import random
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
    random.seed(config.seed)
    start = time()

    if config.metric_analysis_type == "group_analysis":
        # Read the metrics from the comparison group csv
        utils.group_analysis(config, log)
    elif config.metric_analysis_type == "sample_selection_analysis":
        utils.sample_selection_analysis(config, log)
        utils.model_to_model_analysis(config, log)

    log.info("Total time: %s second", time() - start)
    log.info("Process completed!")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
